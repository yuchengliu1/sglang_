#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <cstdint>

#include "common.h"
#include "vec.h"

// DSA indexer FP8 index score (ragged, single-batch loop path).
//
// Mirrors the semantics of the tilelang/CUDA ``fp8_index`` kernel and the
// PyTorch reference in dsa/cpu_kernel.py:
//   1) fp8 q @ fp8 k -> fp32 logits          (per head)
//   2) relu(logits) * q_s (head gate)        (per head)
//   3) sum over heads -> logits_sum
//   4) logits_sum * k_s (per-token scale)    -> index_score
//
// Shapes (all contiguous):
//   q   : [B, M, H, D]  float8_e4m3fn
//   q_s : [B, M, H]     float32
//   k   : [B, N, D]     float8_e4m3fn
//   k_s : [B, N]        float32
//   out : [B, M, N]     float32

// Fused GEMM + relu + per-head weighted reduction, defined in gemm.cpp.
extern void fused_linear_relu_reduce(
    at::Tensor& out,
    at::Tensor& q,
    at::Tensor& q_scale,
    at::Tensor& k,
    at::Tensor& k_scale,
    bool is_vnni);

// AMX weight packing requires the GEMM output dimension (here the number of
// key tokens N) to be a multiple of TILE_N (16). Pad the key rows up so the
// packed path is always taken; the padding rows are never read since
// fused_linear_relu_reduce only produces k_scale.size(0) output columns.
constexpr int64_t kTileN = 16;

// Efficient fp8_e4m3fn → bf16 conversion using CVT_FP8_TO_BF16 from vec.h.
// Processes 32 elements per AVX512 iteration (256-bit load, 512-bit store).
// Avoids the torch .to() path which routes through float32 intermediates.
static void fp8_to_bf16(
    at::BFloat16* __restrict__ dst,
    const uint8_t* __restrict__ src,
    int64_t n) {
#if defined(CPU_CAPABILITY_AVX512)
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m512bh v = CVT_FP8_TO_BF16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i)));
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), (__m512i)v);
  }
  // scalar tail (< 32 elements)
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

at::Tensor fp8_index_cpu(at::Tensor& q, at::Tensor& q_s, at::Tensor& k, at::Tensor& k_s) {
  CHECK_INPUT(q);
  CHECK_INPUT(q_s);
  CHECK_INPUT(k);
  CHECK_INPUT(k_s);

  TORCH_CHECK(q.scalar_type() == at::ScalarType::Float8_e4m3fn, "q must be torch.float8_e4m3fn");
  TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn, "k must be torch.float8_e4m3fn");
  TORCH_CHECK(q_s.scalar_type() == at::kFloat, "q_s must be torch.float32");
  TORCH_CHECK(k_s.scalar_type() == at::kFloat, "k_s must be torch.float32");

  TORCH_CHECK(q.dim() == 4, "q must have shape [B, M, H, D]");
  TORCH_CHECK(k.dim() == 3, "k must have shape [B, N, D]");
  TORCH_CHECK(q_s.dim() == 3, "q_s must have shape [B, M, H]");
  TORCH_CHECK(k_s.dim() == 2, "k_s must have shape [B, N]");

  const int64_t B = q.size(0);
  // Caller always passes B=1 (one batch item per forward_indexer iteration).
  TORCH_CHECK(B == 1, "fp8_index_cpu only supports B=1, got B=", B);

  const int64_t M = q.size(1);
  const int64_t H = q.size(2);
  const int64_t D = q.size(3);
  const int64_t N = k.size(1);

  TORCH_CHECK(k.size(0) == 1 && k.size(2) == D, "k must have shape [1, N, D] matching q");
  TORCH_CHECK(
      q_s.size(0) == 1 && q_s.size(1) == M && q_s.size(2) == H,
      "q_s must have shape [1, M, H] matching q");
  TORCH_CHECK(k_s.size(0) == 1 && k_s.size(1) == N, "k_s must have shape [1, N] matching k");

  auto out = at::empty({1, M, N}, q.options().dtype(at::kFloat));
  if (M == 0 || N == 0) {
    return out;
  }

  const int64_t N_pad = ((N + kTileN - 1) / kTileN) * kTileN;
  const auto bf16_opts = q.options().dtype(at::kBFloat16);

  // Q[0]: [1, M, H, D] fp8 → [M, H, D] bf16.
  // B=1 + contiguous: q.data_ptr() == q[0].data_ptr(), no select needed.
  auto q_bf16 = at::empty({M, H, D}, bf16_opts);
  fp8_to_bf16(
      q_bf16.data_ptr<at::BFloat16>(),
      reinterpret_cast<const uint8_t*>(q.const_data_ptr()),
      M * H * D);

  // K[0]: [1, N, D] fp8 → [N_pad, D] bf16.
  // If N_pad > N the extra rows must be zero (padding); otherwise no zeroing needed.
  at::Tensor k_bf16;
  if (N_pad == N) {
    k_bf16 = at::empty({N, D}, bf16_opts);
  } else {
    k_bf16 = at::zeros({N_pad, D}, bf16_opts);
  }
  fp8_to_bf16(
      k_bf16.data_ptr<at::BFloat16>(),
      reinterpret_cast<const uint8_t*>(k.const_data_ptr()),
      N * D);

  // Views into q_s/k_s/out — no data copies.
  at::Tensor q_scale = q_s.select(0, 0);  // [M, H]
  at::Tensor k_scale = k_s.select(0, 0);  // [N]
  at::Tensor out_b   = out.select(0, 0);  // [M, N]
  fused_linear_relu_reduce(out_b, q_bf16, q_scale, k_bf16, k_scale, /*is_vnni=*/false);

  return out;
}
