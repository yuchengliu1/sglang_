#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "common.h"
#include "vec.h"

// Fused GEMM + relu + per-head weighted reduction, defined in gemm.cpp.
template <bool parallelize_internally, bool is_vnni>
void fused_linear_relu_reduce(
    at::Tensor& out,
    at::Tensor& q,
    at::Tensor& q_scale,
    at::Tensor& k,
    at::Tensor& k_scale);

namespace {

constexpr int64_t kBlockSize = 64;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kHeadDimWithScaleBytes = 132;
constexpr int64_t kScaleOffsetBytes = kBlockSize * kHeadDim;
constexpr int64_t kBlockBytes = kBlockSize * kHeadDimWithScaleBytes;
// AMX weight-packing granularity required by convert_weight_packed (TILE_N in gemm.h).
constexpr int64_t kTileN = 16;
// Target KV-chunk tasks per thread when adaptively sizing kv_chunk_size in schedule_paged_mqa_tasks.
constexpr int64_t kTasksPerThread = 4;

template <typename T>
inline int64_t load_int(const T* ptr, int64_t idx) {
  return static_cast<int64_t>(ptr[idx]);
}

// One row's KV range split into a kv_chunk_size-wide slice; the scheduling
// unit below, so a single long row (including batch_size == 1) can be spread
// across multiple threads instead of being pinned to one.
struct PagedMqaTask {
  int64_t row;
  int64_t kv_start;
  int64_t kv_len;
};

// Splits every row's [0, seq_len) KV range into kv_chunk_size-wide tasks and
// greedily assigns them to at::get_num_threads() cost-balanced groups (cost =
// task kv_len), mirroring the cost_prefix walk in gemm.cpp's
// schedule_mqa_row_tiles. Unlike balancing whole rows across threads, this
// also spreads a single long row across threads, which whole-row balancing
// cannot do. kv_chunk_size is picked adaptively (see below) and adjacent
// same-row tasks landing in the same group are coalesced back into one task,
// so a fine chunk size only costs cheap bookkeeping, never extra
// fused_linear_relu_reduce/gather calls, whenever several chunks of the same
// row end up sharing a thread.
template <typename seq_t>
std::vector<std::vector<PagedMqaTask>> schedule_paged_mqa_tasks(
    const seq_t* __restrict__ seq_ptr, int64_t batch_size) {
  const int64_t nth = at::get_num_threads();
  std::vector<std::vector<PagedMqaTask>> groups(nth);

  int64_t total_cost = 0;
  for (int64_t b = 0; b < batch_size; ++b) {
    total_cost += load_int(seq_ptr, b);
  }
  if (total_cost == 0) {
    return groups;
  }

  // Aim for kTasksPerThread chunks per thread on average: fine enough to
  // balance a skewed batch or a single long row, coarse enough to keep the
  // tasks/cost_prefix arrays and the assignment walk below cheap.
  const int64_t kv_chunk_size =
      std::max(kTileN, div_up(div_up(total_cost, nth * kTasksPerThread), kTileN) * kTileN);

  int64_t max_tasks = 0;
  for (int64_t b = 0; b < batch_size; ++b) {
    const int64_t seq_len = load_int(seq_ptr, b);
    max_tasks += div_up(std::max<int64_t>(seq_len, 1), kv_chunk_size);
  }

  std::vector<PagedMqaTask> tasks;
  tasks.reserve(max_tasks);
  std::vector<int64_t> cost_prefix;
  cost_prefix.reserve(max_tasks + 1);
  cost_prefix.push_back(0);
  for (int64_t b = 0; b < batch_size; ++b) {
    const int64_t seq_len = load_int(seq_ptr, b);
    for (int64_t kv_start = 0; kv_start < seq_len; kv_start += kv_chunk_size) {
      const int64_t kv_len = std::min(kv_chunk_size, seq_len - kv_start);
      tasks.push_back({b, kv_start, kv_len});
      cost_prefix.push_back(cost_prefix.back() + kv_len);
    }
  }

  const int64_t num_tasks = static_cast<int64_t>(tasks.size());
  int64_t ith = 0;
  for (int64_t t = 0; t < num_tasks; ++t) {
    while (ith + 1 < nth && cost_prefix[t] >= total_cost * (ith + 1) / nth) {
      ++ith;
    }
    // Same-row-and-same-group tasks are always adjacent in `tasks` (ith
    // never decreases), so checking only the group's last entry detects
    // every mergeable run.
    auto& group = groups[ith];
    if (!group.empty() && group.back().row == tasks[t].row &&
        group.back().kv_start + group.back().kv_len == tasks[t].kv_start) {
      group.back().kv_len += tasks[t].kv_len;
    } else {
      group.push_back(tasks[t]);
    }
  }
  return groups;
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

  for (int64_t b = 0; b < batch_size; ++b) {
    const int64_t seq_len = load_int(seq_ptr, b);
    TORCH_CHECK(seq_len >= 0 && seq_len <= max_seq_len, "seq_lens must be in [0, max_seq_len]");
  }

  // Precompute every row's q (fp8 -> bf16) up front: with the KV-chunked
  // tasks below, one row's chunks can run concurrently on different
  // threads, so each task must only read q_bf16, never lazily compute it.
  auto q_bf16_all = at::empty({batch_size, num_heads, kHeadDim}, bf16_opts);
  at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      fp8_to_bf16(
          q_bf16_all.data_ptr<at::BFloat16>() + b * num_heads * kHeadDim,
          q_ptr + b * num_heads * kHeadDim,
          num_heads * kHeadDim);
    }
  });

  // Balance KV-chunk tasks by cumulative cost across threads so both a batch
  // with skewed seq_lens and a single long row (e.g. batch_size == 1) are
  // spread evenly.
  const auto task_groups = schedule_paged_mqa_tasks(seq_ptr, batch_size);
  const int64_t nth = static_cast<int64_t>(task_groups.size());

  at::parallel_for(0, nth, 1, [&](int64_t begin, int64_t end) {
    for (int64_t ith = begin; ith < end; ++ith) {
      for (const auto& task : task_groups[ith]) {
        const int64_t kv_len_pad = ((task.kv_len + kTileN - 1) / kTileN) * kTileN;

        at::Tensor k_bf16;
        if (kv_len_pad == task.kv_len) {
          k_bf16 = at::empty({task.kv_len, kHeadDim}, bf16_opts);
        } else {
          k_bf16 = at::zeros({kv_len_pad, kHeadDim}, bf16_opts);
        }
        at::Tensor k_scale = at::empty({task.kv_len}, f32_opts);

        // Gather this task's [kv_start, kv_start + kv_len) sub-range of the
        // row's paged K (fp8 -> bf16) and per-token k_scale; k_bf16 is
        // padded up to a multiple of kTileN so the AMX weight-packing path
        // in fused_linear_relu_reduce is always usable, but the padding rows
        // are never read since fused_linear_relu_reduce only produces
        // k_scale.size(0) output columns.
        const page_t* page_row = page_ptr + task.row * pages_per_batch;
        at::BFloat16* k_bf16_row = k_bf16.data_ptr<at::BFloat16>();
        float* k_scale_row = k_scale.data_ptr<float>();
        for (int64_t i = 0; i < task.kv_len; ++i) {
          const int64_t token = task.kv_start + i;
          const int64_t logical_page = token / kBlockSize;
          const int64_t token_in_page = token % kBlockSize;
          TORCH_CHECK(logical_page < pages_per_batch, "page_table does not cover seq_len");

          const int64_t physical_page = load_int(page_row, logical_page);
          TORCH_CHECK(physical_page >= 0 && physical_page < num_blocks, "page_table contains an invalid page index");

          const uint8_t* block = cache_ptr + physical_page * kBlockBytes;
          const uint8_t* k_token = block + token_in_page * kHeadDim;
          const float* scale_ptr = reinterpret_cast<const float*>(block + kScaleOffsetBytes);

          fp8_to_bf16(k_bf16_row + i * kHeadDim, k_token, kHeadDim);
          k_scale_row[i] = scale_ptr[token_in_page];
        }

        auto q_bf16_row = q_bf16_all.narrow(0, task.row, 1);
        at::Tensor q_scale_row = weight_f32.select(0, task.row).unsqueeze(0).contiguous();  // [1, num_heads]
        // dim0 is narrowed to size 1, so this slice stays contiguous and can
        // be written to directly (no temporary buffer + copy_ needed).
        auto out_view = logits.narrow(0, task.row, 1).narrow(1, task.kv_start, task.kv_len);

        fused_linear_relu_reduce<false, false>(out_view, q_bf16_row, q_scale_row, k_bf16, k_scale);
      }
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
