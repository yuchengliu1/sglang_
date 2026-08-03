#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <cstdint>

#include "common.h"
#include "vec.h"

// Fused GEMM + relu + per-head weighted reduction, defined in gemm.cpp.
extern void fused_linear_relu_reduce(
    at::Tensor& out,
    at::Tensor& q,
    at::Tensor& q_scale,
    at::Tensor& k,
    at::Tensor& k_scale,
    bool is_vnni);

namespace {

constexpr int64_t kBlockSize = 64;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kHeadDimWithScaleBytes = 132;
constexpr int64_t kScaleOffsetBytes = kBlockSize * kHeadDim;
constexpr int64_t kBlockBytes = kBlockSize * kHeadDimWithScaleBytes;
// AMX weight-packing granularity required by convert_weight_packed (TILE_N in gemm.h).
constexpr int64_t kTileN = 16;

// Efficient fp8_e4m3fn -> bf16 conversion using CVT_FP8_TO_BF16 from vec.h.
// Processes 32 elements per AVX512 iteration (256-bit load, 512-bit store).
void fp8_to_bf16(at::BFloat16* __restrict__ dst, const uint8_t* __restrict__ src, int64_t n) {
#if defined(CPU_CAPABILITY_AVX512)
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m512bh v = CVT_FP8_TO_BF16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i)));
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), (__m512i)v);
  }
  for (; i < n; ++i) {
    c10::Float8_e4m3fn x;
    x.x = src[i];
    dst[i] = at::BFloat16(static_cast<float>(x));
  }
#else
  for (int64_t i = 0; i < n; ++i) {
    c10::Float8_e4m3fn x;
    x.x = src[i];
    dst[i] = at::BFloat16(static_cast<float>(x));
  }
#endif
}

template <typename T>
inline int64_t load_int(const T* ptr, int64_t idx) {
  return static_cast<int64_t>(ptr[idx]);
}

// Gather one batch row's paged K (fp8 -> bf16) and per-token k_scale into
// contiguous buffers, padded up to a multiple of kTileN so the AMX
// weight-packing path in fused_linear_relu_reduce is always usable. The
// padding rows are packed but never read since fused_linear_relu_reduce
// only produces k_scale.size(0) == seq_len output columns.
template <typename page_t>
void gather_row_kv(
    const uint8_t* __restrict__ cache_ptr,
    const page_t* __restrict__ page_row,
    int64_t pages_per_batch,
    int64_t num_blocks,
    int64_t seq_len,
    at::BFloat16* __restrict__ k_bf16_row,
    float* __restrict__ k_scale_row) {
  for (int64_t token = 0; token < seq_len; ++token) {
    const int64_t logical_page = token / kBlockSize;
    const int64_t token_in_page = token % kBlockSize;
    TORCH_CHECK(logical_page < pages_per_batch, "page_table does not cover seq_len");

    const int64_t physical_page = load_int(page_row, logical_page);
    TORCH_CHECK(physical_page >= 0 && physical_page < num_blocks, "page_table contains an invalid page index");

    const uint8_t* block = cache_ptr + physical_page * kBlockBytes;
    const uint8_t* k_token = block + token_in_page * kHeadDim;
    const float* scale_ptr = reinterpret_cast<const float*>(block + kScaleOffsetBytes);

    fp8_to_bf16(k_bf16_row + token * kHeadDim, k_token, kHeadDim);
    k_scale_row[token] = scale_ptr[token_in_page];
  }
}

template <typename seq_t, typename page_t>
void fp8_paged_mqa_logits_cpu_impl(
    const at::Tensor& q_fp8,
    const at::Tensor& kvcache_fp8,
    const at::Tensor& weight_f32,
    const at::Tensor& seq_lens,
    const at::Tensor& page_table,
    at::Tensor& logits,
    int64_t max_seq_len) {
  const int64_t batch_size = q_fp8.size(0);
  const int64_t num_heads = q_fp8.size(2);
  const int64_t num_blocks = kvcache_fp8.size(0);
  const int64_t pages_per_batch = page_table.size(1);

  const auto* q_ptr = reinterpret_cast<const uint8_t*>(q_fp8.const_data_ptr());
  const auto* cache_ptr = reinterpret_cast<const uint8_t*>(kvcache_fp8.const_data_ptr());
  const auto* seq_ptr = seq_lens.const_data_ptr<seq_t>();
  const auto* page_ptr = page_table.const_data_ptr<page_t>();

  const auto bf16_opts = q_fp8.options().dtype(at::kBFloat16);
  const auto f32_opts = weight_f32.options();

  // Parallelize across batch rows; each row does its own paged-cache gather
  // + a tiny GEMM (M=1) via fused_linear_relu_reduce instead of a scalar
  // per-token dot-product loop.
  at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      const int64_t seq_len = load_int(seq_ptr, b);
      TORCH_CHECK(seq_len >= 0 && seq_len <= max_seq_len, "seq_lens must be in [0, max_seq_len]");
      if (seq_len == 0) {
        continue;
      }

      const int64_t seq_len_pad = ((seq_len + kTileN - 1) / kTileN) * kTileN;

      at::Tensor k_bf16;
      if (seq_len_pad == seq_len) {
        k_bf16 = at::empty({seq_len, kHeadDim}, bf16_opts);
      } else {
        k_bf16 = at::zeros({seq_len_pad, kHeadDim}, bf16_opts);
      }
      auto k_scale = at::empty({seq_len}, f32_opts);

      gather_row_kv<page_t>(
          cache_ptr,
          page_ptr + b * pages_per_batch,
          pages_per_batch,
          num_blocks,
          seq_len,
          k_bf16.data_ptr<at::BFloat16>(),
          k_scale.data_ptr<float>());

      auto q_bf16 = at::empty({1, num_heads, kHeadDim}, bf16_opts);
      fp8_to_bf16(q_bf16.data_ptr<at::BFloat16>(), q_ptr + b * num_heads * kHeadDim, num_heads * kHeadDim);

      at::Tensor q_scale_row = weight_f32.select(0, b).unsqueeze(0).contiguous();  // [1, num_heads]
      auto out_row = at::empty({1, seq_len}, f32_opts);

      fused_linear_relu_reduce(out_row, q_bf16, q_scale_row, k_bf16, k_scale, /*is_vnni=*/false);

      logits.narrow(0, b, 1).narrow(1, 0, seq_len).copy_(out_row);
    }
  });
}

template <typename seq_t>
void dispatch_page_type(
    const at::Tensor& q_fp8,
    const at::Tensor& kvcache_fp8,
    const at::Tensor& weight_f32,
    const at::Tensor& seq_lens,
    const at::Tensor& page_table,
    at::Tensor& logits,
    int64_t max_seq_len) {
  if (page_table.scalar_type() == at::kInt) {
    fp8_paged_mqa_logits_cpu_impl<seq_t, int32_t>(q_fp8, kvcache_fp8, weight_f32, seq_lens, page_table, logits, max_seq_len);
  } else if (page_table.scalar_type() == at::kLong) {
    fp8_paged_mqa_logits_cpu_impl<seq_t, int64_t>(q_fp8, kvcache_fp8, weight_f32, seq_lens, page_table, logits, max_seq_len);
  } else {
    TORCH_CHECK(false, "page_table must be int32 or int64");
  }
}

}  // namespace

at::Tensor fp8_paged_mqa_logits_cpu(
    at::Tensor& q_fp8,
    at::Tensor& kvcache_fp8,
    at::Tensor& weight,
    at::Tensor& seq_lens,
    at::Tensor& page_table,
    int64_t max_seq_len,
    bool clean_logits) {
  TORCH_CHECK(!clean_logits, "fp8_paged_mqa_logits_cpu only supports clean_logits == false");
  CHECK_INPUT(q_fp8);
  CHECK_INPUT(kvcache_fp8);
  CHECK_INPUT(weight);
  CHECK_INPUT(seq_lens);
  CHECK_INPUT(page_table);
  TORCH_CHECK(q_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn, "q_fp8 must be torch.float8_e4m3fn");
  TORCH_CHECK(kvcache_fp8.scalar_type() == at::kByte, "kvcache_fp8 must be torch.uint8 storage");

  // The checks are aligned with fp8_paged_mqa_logits_torch in indexer.py.
  TORCH_CHECK(q_fp8.dim() == 4, "q_fp8 must have shape [batch, 1, heads, 128]");
  TORCH_CHECK(q_fp8.size(1) == 1, "q_fp8 second dimension must be 1");
  TORCH_CHECK(q_fp8.size(3) == kHeadDim, "q_fp8 head_dim must be 128");
  TORCH_CHECK(kvcache_fp8.dim() == 4, "kvcache_fp8 must have shape [blocks, 64, 1, 132]");
  TORCH_CHECK(kvcache_fp8.size(1) == kBlockSize, "kvcache_fp8 block size must be 64");
  TORCH_CHECK(kvcache_fp8.size(2) == 1, "kvcache_fp8 num kv heads must be 1");
  TORCH_CHECK(kvcache_fp8.size(3) == kHeadDimWithScaleBytes, "kvcache_fp8 last dimension must be 132 bytes");

  const int64_t batch_size = q_fp8.size(0);
  const int64_t num_heads = q_fp8.size(2);
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({batch_size, num_heads}), "weight must have shape [batch, heads]");
  TORCH_CHECK(seq_lens.sizes() == at::IntArrayRef({batch_size}), "seq_lens must have shape [batch]");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) == batch_size, "page_table must have shape [batch, pages]");
  TORCH_CHECK(max_seq_len >= 0, "max_seq_len must be non-negative");

  // fused_linear_relu_reduce requires fp32 q_scale/weight; upcast once for
  // the whole batch (CPU forward always produces fp32 weights already, so
  // this is normally a no-op aside from the dtype check).
  at::Tensor weight_f32 = weight.scalar_type() == at::kFloat ? weight : weight.to(at::kFloat);

  auto logits = at::empty({batch_size, max_seq_len}, q_fp8.options().dtype(at::kFloat));

  if (seq_lens.scalar_type() == at::kInt) {
    dispatch_page_type<int32_t>(q_fp8, kvcache_fp8, weight_f32, seq_lens, page_table, logits, max_seq_len);
  } else if (seq_lens.scalar_type() == at::kLong) {
    dispatch_page_type<int64_t>(q_fp8, kvcache_fp8, weight_f32, seq_lens, page_table, logits, max_seq_len);
  } else {
    TORCH_CHECK(false, "seq_lens must be int32 or int64");
  }

  return logits;
}
