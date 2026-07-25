#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <cstdint>

#include "common.h"
#include "vec.h"

namespace {

constexpr int64_t kHeadDim = 128;

inline float fp8_e4m3_to_float(uint8_t v) {
  c10::Float8_e4m3fn x;
  x.x = v;
  return static_cast<float>(x);
}

inline float dot_fp8_128_scalar(const uint8_t* q, const uint8_t* k) {
  float dot = 0.0f;
  for (int64_t d = 0; d < kHeadDim; ++d) {
    dot += fp8_e4m3_to_float(q[d]) * fp8_e4m3_to_float(k[d]);
  }
  return dot;
}

#if defined(CPU_CAPABILITY_AVX512)
inline float dot_fp8_128(const uint8_t* q, const uint8_t* k) {
  __m512 acc = _mm512_setzero_ps();
  for (int64_t d = 0; d < kHeadDim; d += 32) {
    const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d));
    const __m256i k8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + d));
    acc = _mm512_dpbf16_ps(acc, CVT_FP8_TO_BF16(q8), CVT_FP8_TO_BF16(k8));
  }
  return _mm512_reduce_add_ps(acc);
}
#else
inline float dot_fp8_128(const uint8_t* q, const uint8_t* k) {
  return dot_fp8_128_scalar(q, k);
}
#endif

template <typename T>
inline int64_t load_int(const T* ptr, int64_t idx) {
  return static_cast<int64_t>(ptr[idx]);
}

template <typename T>
inline float load_weight(const T* ptr, int64_t idx) {
  return static_cast<float>(ptr[idx]);
}

// Ragged (non-paged) fp8 MQA logits: q and k are both flat/concatenated across
// all requests in the batch. Query token `i` may only attend to key tokens in
// the half-open range [ks[i], ke[i)) of the shared k buffer (causal, and
// scoped to its own request via the request-local offset baked into ks/ke).
template <typename index_t, typename weight_t>
void fp8_mqa_logits_cpu_impl(
    const at::Tensor& q_fp8,
    const at::Tensor& k_fp8,
    const at::Tensor& k_scale,
    const at::Tensor& weight,
    const at::Tensor& ks,
    const at::Tensor& ke,
    at::Tensor& logits) {
  const int64_t num_q = q_fp8.size(0);
  const int64_t num_heads = q_fp8.size(1);
  const int64_t num_k = k_fp8.size(0);

  const auto* q_ptr = reinterpret_cast<const uint8_t*>(q_fp8.const_data_ptr());
  const auto* k_ptr = reinterpret_cast<const uint8_t*>(k_fp8.const_data_ptr());
  const auto* k_scale_ptr = k_scale.const_data_ptr<float>();
  const auto* weight_ptr = weight.const_data_ptr<weight_t>();
  const auto* ks_ptr = ks.const_data_ptr<index_t>();
  const auto* ke_ptr = ke.const_data_ptr<index_t>();
  auto* out_ptr = logits.data_ptr<float>();

  const int64_t grain_size = std::max<int64_t>(GRAIN_SIZE / std::max<int64_t>(num_heads, 1), 1);
  at::parallel_for(0, num_q, grain_size, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t k_start = load_int(ks_ptr, i);
      const int64_t k_end = load_int(ke_ptr, i);
      TORCH_CHECK(
          k_start >= 0 && k_end <= num_k && k_start <= k_end,
          "ks/ke out of range in fp8_mqa_logits_cpu");

      const uint8_t* q_row_base = q_ptr + i * num_heads * kHeadDim;
      const weight_t* w_row = weight_ptr + i * num_heads;
      float* out_row = out_ptr + i * num_k;

      for (int64_t k = k_start; k < k_end; ++k) {
        const uint8_t* k_row = k_ptr + k * kHeadDim;
        float score_sum = 0.0f;
        for (int64_t h = 0; h < num_heads; ++h) {
          const uint8_t* q_head = q_row_base + h * kHeadDim;
          float dot = dot_fp8_128(q_head, k_row);
          dot = std::max(dot, 0.0f);
          score_sum += dot * load_weight(w_row, h);
        }
        out_row[k] = score_sum * k_scale_ptr[k];
      }
    }
  });
}

template <typename index_t>
void dispatch_weight_type(
    const at::Tensor& q_fp8,
    const at::Tensor& k_fp8,
    const at::Tensor& k_scale,
    const at::Tensor& weight,
    const at::Tensor& ks,
    const at::Tensor& ke,
    at::Tensor& logits) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, weight.scalar_type(), "fp8_mqa_logits_cpu_weight", [&] {
        fp8_mqa_logits_cpu_impl<index_t, scalar_t>(q_fp8, k_fp8, k_scale, weight, ks, ke, logits);
      });
}

}  // namespace

at::Tensor fp8_mqa_logits_cpu(
    at::Tensor& q_fp8,
    at::Tensor& k_fp8,
    at::Tensor& k_scale,
    at::Tensor& weight,
    at::Tensor& ks,
    at::Tensor& ke,
    bool clean_logits) {
  TORCH_CHECK(!clean_logits, "fp8_mqa_logits_cpu only supports clean_logits == false");
  CHECK_INPUT(q_fp8);
  CHECK_INPUT(k_fp8);
  CHECK_INPUT(k_scale);
  CHECK_INPUT(weight);
  CHECK_INPUT(ks);
  CHECK_INPUT(ke);
  TORCH_CHECK(q_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn, "q_fp8 must be torch.float8_e4m3fn");
  TORCH_CHECK(k_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn, "k_fp8 must be torch.float8_e4m3fn");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat, "k_scale must be torch.float32");

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

  // Zero-init so any (masked-out) region outside [ks[i], ke[i)) is
  // deterministic rather than reading uninitialized heap memory; downstream
  // topk_transform only trusts the [ks[i], ke[i)) slice of each row anyway.
  auto logits = at::zeros({num_q, num_k}, k_scale.options().dtype(at::kFloat));
  if (num_q == 0 || num_k == 0) {
    return logits;
  }

  if (ks.scalar_type() == at::kInt) {
    TORCH_CHECK(ke.scalar_type() == at::kInt, "ks and ke must have the same dtype");
    dispatch_weight_type<int32_t>(q_fp8, k_fp8, k_scale, weight, ks, ke, logits);
  } else if (ks.scalar_type() == at::kLong) {
    TORCH_CHECK(ke.scalar_type() == at::kLong, "ks and ke must have the same dtype");
    dispatch_weight_type<int64_t>(q_fp8, k_fp8, k_scale, weight, ks, ke, logits);
  } else {
    TORCH_CHECK(false, "ks/ke must be int32 or int64");
  }

  return logits;
}
