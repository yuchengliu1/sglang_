#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "common.h"
#include "vec.h"

// Ragged (non-paged) fp8 MQA logits: q and k are both flat/concatenated across
// all requests in the batch. Query token `i` may only attend to key tokens in
// the half-open range [ks[i], ke[i)) of the shared k buffer (causal, and
// scoped to its own request via the request-local offset baked into ks/ke).
//
// This is a GEMM (Q @ K^T per head, relu, per-head weighted reduce, scale by
// k_scale), so — like fp8_index.cpp / fp8_paged_mqa_logits_cpu — it is
// computed via convert-to-bf16 + fused_linear_relu_reduce (the AMX/vector GEMM
// kernel in gemm.cpp) instead of a scalar per-token dot-product loop.
//
// clean_logits=False (the common fast path): this computes a full dense
// [num_q, num_k] matrix; the caller (topk_transform with ks=ks) is responsible
// for only selecting within [ks[i], ke[i)) per row, so entries outside that
// range may hold unmasked (but never read) values.
//
// clean_logits=True: entries outside [ks[i], ke[i)) are explicitly filled with
// -inf after the GEMM, matching the CUDA/HIP deep_gemm/aiter semantics, for
// callers that read the raw logits without going through topk_transform's
// ks-aware masking.

namespace {

constexpr int64_t kHeadDim = 128;
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

// Fill entries outside [ks[i], ke[i)) with -inf for every row i, in place.
void clean_logits_range(at::Tensor& logits, const at::Tensor& ks, const at::Tensor& ke) {
  const int64_t num_q = logits.size(0);
  const int64_t num_k = logits.size(1);
  float* __restrict__ logits_ptr = logits.data_ptr<float>();
  const int32_t* __restrict__ ks_ptr = ks.const_data_ptr<int32_t>();
  const int32_t* __restrict__ ke_ptr = ke.const_data_ptr<int32_t>();
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();

  at::parallel_for(0, num_q, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      float* __restrict__ row = logits_ptr + i * num_k;
      const int64_t k_start = std::clamp<int64_t>(ks_ptr[i], 0, num_k);
      const int64_t k_end = std::clamp<int64_t>(ke_ptr[i], 0, num_k);
      std::fill(row, row + k_start, neg_inf);
      std::fill(row + std::max(k_start, k_end), row + num_k, neg_inf);
    }
  });
}

}  // namespace

// Fused GEMM + relu + per-head weighted reduction, defined in gemm.cpp.
extern void fused_linear_relu_reduce(
    at::Tensor& out,
    at::Tensor& q,
    at::Tensor& q_scale,
    at::Tensor& k,
    at::Tensor& k_scale,
    bool is_vnni);

at::Tensor fp8_mqa_logits_cpu(
    at::Tensor& q_fp8,
    at::Tensor& k_fp8,
    at::Tensor& k_scale,
    at::Tensor& weight,
    at::Tensor& ks,
    at::Tensor& ke,
    bool clean_logits) {
  CHECK_INPUT(q_fp8);
  CHECK_INPUT(k_fp8);
  CHECK_INPUT(k_scale);
  CHECK_INPUT(weight);
  CHECK_INPUT(ks);
  CHECK_INPUT(ke);
  TORCH_CHECK(q_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn, "q_fp8 must be torch.float8_e4m3fn");
  TORCH_CHECK(k_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn, "k_fp8 must be torch.float8_e4m3fn");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat, "k_scale must be torch.float32");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be torch.float32");

  TORCH_CHECK(q_fp8.dim() == 3, "q_fp8 must have shape [num_q_tokens, num_heads, head_dim]");
  TORCH_CHECK(q_fp8.size(2) == kHeadDim, "q_fp8 head_dim must be 128");
  TORCH_CHECK(k_fp8.dim() == 2 && k_fp8.size(1) == kHeadDim, "k_fp8 must have shape [num_k_tokens, 128]");
  TORCH_CHECK(k_scale.dim() == 1 && k_scale.size(0) == k_fp8.size(0), "k_scale must have shape [num_k_tokens]");

  const int64_t num_q = q_fp8.size(0);
  const int64_t num_heads = q_fp8.size(1);
  const int64_t num_k = k_fp8.size(0);

  TORCH_CHECK(
      weight.dim() == 2 && weight.size(0) == num_q && weight.size(1) == num_heads,
      "weight must have shape [num_q_tokens, num_heads]");
  TORCH_CHECK(ks.dim() == 1 && ks.size(0) == num_q, "ks must have shape [num_q_tokens]");
  TORCH_CHECK(ke.sizes() == ks.sizes(), "ke must have the same shape as ks");
  TORCH_CHECK(ks.scalar_type() == at::kInt, "ks must be torch.int32");
  TORCH_CHECK(ke.scalar_type() == at::kInt, "ke must be torch.int32");

  auto logits = at::empty({num_q, num_k}, weight.options().dtype(at::kFloat));
  if (num_q == 0 || num_k == 0) {
    return logits;
  }

  // AMX weight packing (inside fused_linear_relu_reduce) requires the K row
  // count to be a multiple of kTileN; pad with zero rows when needed. The
  // padding rows are never read since fused_linear_relu_reduce only produces
  // k_scale.size(0) == num_k output columns.
  const int64_t num_k_pad = ((num_k + kTileN - 1) / kTileN) * kTileN;
  const auto bf16_opts = q_fp8.options().dtype(at::kBFloat16);

  auto q_bf16 = at::empty({num_q, num_heads, kHeadDim}, bf16_opts);
  fp8_to_bf16(
      q_bf16.data_ptr<at::BFloat16>(),
      reinterpret_cast<const uint8_t*>(q_fp8.const_data_ptr()),
      num_q * num_heads * kHeadDim);

  at::Tensor k_bf16;
  if (num_k_pad == num_k) {
    k_bf16 = at::empty({num_k, kHeadDim}, bf16_opts);
  } else {
    k_bf16 = at::zeros({num_k_pad, kHeadDim}, bf16_opts);
  }
  fp8_to_bf16(
      k_bf16.data_ptr<at::BFloat16>(), reinterpret_cast<const uint8_t*>(k_fp8.const_data_ptr()), num_k * kHeadDim);

  fused_linear_relu_reduce(logits, q_bf16, weight, k_bf16, k_scale, /*is_vnni=*/false);

  if (clean_logits) {
    clean_logits_range(logits, ks, ke);
  }

  return logits;
}
