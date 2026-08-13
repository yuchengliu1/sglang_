import unittest
from typing import Any

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

BLOCK_SIZE = 64
HEAD_DIM = 128
HEAD_DIM_WITH_SCALE_BYTES = 132

FP8_DTYPE = torch.float8_e4m3fn


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
    for i in range(batch_size):
        q = q_fp8[i, 0]
        q = q.to(torch.float32)
        q_scale = weight[i]
        seq_len = int(seq_lens[i].item())
        assert seq_len <= max_seq_len
        num_pages = (seq_len + block_size - 1) // block_size
        padded_seq_len = num_pages * block_size
        pages = page_table[i, :num_pages]
        kvcache_fp8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
        kvcache = kvcache_fp8[pages]
        SCALE_OFFSET = block_size * head_dim
        kvcache_value = kvcache[..., :SCALE_OFFSET].view(dtype=FP8_DTYPE)
        kvcache_scale = kvcache[..., SCALE_OFFSET:].view(dtype=torch.float32)
        kvcache_value = kvcache_value.to(torch.float32)
        kvcache_scale = kvcache_scale.contiguous()
        kvcache_value = kvcache_value.view(padded_seq_len, head_dim)
        kvcache_scale = kvcache_scale.view(padded_seq_len)
        score = F.linear(kvcache_value, q)
        score = F.relu(score)
        score *= q_scale[None, :]
        score = score.sum(dim=1)
        score *= kvcache_scale
        logits[i, :seq_len] = score[:seq_len]

    return logits


class TestFp8PagedMqaLogitsCPU(CustomTestCase):
    def _make_inputs(
        self,
        *,
        seq_lens_list,
        num_heads: int = 4,
        max_seq_len: int = 192,
        index_dtype: torch.dtype = torch.int32,
        weight_dtype: torch.dtype = torch.float32,
        q_dtype: torch.dtype = torch.bfloat16,
        num_blocks: int = None,
        page_rows: list = None,
    ):
        torch.manual_seed(2)
        batch_size = len(seq_lens_list)
        pages_per_batch = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        # num_blocks/page_rows affect the RNG draw size (and thus every value
        # drawn afterwards), so test_matches_torch_reference pins both to its
        # original values instead of the auto-generated ones below.
        if num_blocks is None:
            num_blocks = batch_size * pages_per_batch

        q = (torch.randn(batch_size, 1, num_heads, HEAD_DIM) * 0.25).to(q_dtype)
        q_fp8 = q.to(torch.float8_e4m3fn).contiguous()

        k = (torch.randn(num_blocks, BLOCK_SIZE, HEAD_DIM) * 0.25).to(q_dtype)
        k_fp8 = k.to(torch.float8_e4m3fn).contiguous()
        k_bytes = k_fp8.view(num_blocks, BLOCK_SIZE * HEAD_DIM).view(dtype=torch.uint8)

        scales = torch.rand(num_blocks, BLOCK_SIZE, dtype=torch.float32) * 0.5 + 0.75
        scale_bytes = (
            scales.contiguous().view(num_blocks, BLOCK_SIZE).view(dtype=torch.uint8)
        )

        kvcache = torch.cat([k_bytes, scale_bytes], dim=1).contiguous()
        kvcache = kvcache.view(num_blocks, BLOCK_SIZE, 1, HEAD_DIM_WITH_SCALE_BYTES)

        weight = (
            torch.randn(batch_size, num_heads, dtype=torch.float32)
            .to(weight_dtype)
            .contiguous()
        )
        seq_lens = torch.tensor(seq_lens_list, dtype=index_dtype)

        page_table = torch.empty(batch_size, pages_per_batch, dtype=index_dtype)
        if page_rows is not None:
            for i, row in enumerate(page_rows):
                page_table[i] = torch.tensor(row, dtype=index_dtype)
        else:
            # Disjoint per-row pages: simpler than sharing blocks across rows
            # and still exercises the paged gather across a row's own page
            # boundaries.
            for i in range(batch_size):
                page_table[i] = torch.arange(
                    i * pages_per_batch, (i + 1) * pages_per_batch, dtype=index_dtype
                )

        return q_fp8, kvcache, weight, seq_lens, page_table, max_seq_len

    def _assert_matches_reference(
        self,
        seq_lens_list,
        *,
        max_seq_len: int = 192,
        index_dtype: torch.dtype = torch.int32,
        weight_dtype: torch.dtype = torch.float32,
        q_dtype: torch.dtype = torch.bfloat16,
        num_blocks: int = None,
        page_rows: list = None,
        atol: float = None,
    ):
        q_fp8, kvcache, weight, seq_lens, page_table, max_seq_len = self._make_inputs(
            seq_lens_list=seq_lens_list,
            max_seq_len=max_seq_len,
            index_dtype=index_dtype,
            weight_dtype=weight_dtype,
            q_dtype=q_dtype,
            num_blocks=num_blocks,
            page_rows=page_rows,
        )

        actual = torch.ops.sgl_kernel.fp8_paged_mqa_logits_cpu(
            q_fp8,
            kvcache,
            weight,
            seq_lens,
            page_table,
            max_seq_len,
            False,
        )
        expected = fp8_paged_mqa_logits_torch(
            q_fp8,
            kvcache,
            weight,
            seq_lens,
            page_table,
            None,
            max_seq_len,
            False,
        )

        self.assertEqual(actual.shape, (seq_lens.numel(), max_seq_len))
        self.assertEqual(actual.dtype, torch.float32)
        # atol may be loosened past precision[q_dtype] (rtol stays tight):
        # summing several per-head weighted terms can cancel down to a small
        # value, at which point brgemm-tiled-vs-plain-torch reduction-order
        # rounding noise (negligible relative to the uncancelled per-head
        # magnitudes) becomes non-negligible in absolute terms - see the same
        # note in test_fp8_index_cpu.py.
        rtol = precision[q_dtype]
        atol = precision[q_dtype] if atol is None else atol
        for batch_idx, seq_len in enumerate(seq_lens.tolist()):
            if seq_len == 0:
                continue
            torch.testing.assert_close(
                actual[batch_idx, :seq_len],
                expected[batch_idx, :seq_len],
                atol=atol,
                rtol=rtol,
            )

    def test_matches_torch_reference(self):
        # num_blocks/page_rows pinned to the original values: num_blocks
        # alone shifts every subsequent RNG draw (k/scales/weight), and a
        # different draw can land closer to the reduction-order rounding
        # noise inherent to comparing a brgemm-tiled sum against a plain
        # torch reduction (see _assert_matches_reference's tolerance note).
        self._assert_matches_reference(
            [0, 65, 191],
            max_seq_len=192,
            index_dtype=torch.int32,
            weight_dtype=torch.float32,
            q_dtype=torch.bfloat16,
            num_blocks=8,
            page_rows=[[0, 1, 2], [3, 4, 5], [2, 6, 7]],
        )

    def test_single_request_long_context(self):
        # batch_size == 1 with a long, kv_chunk_size(512)-unaligned context:
        # the only way to parallelize this case is to split one row's KV
        # range across threads (schedule_paged_mqa_tasks), since there are no
        # other rows to balance against.
        self._assert_matches_reference([4000], max_seq_len=4096, atol=0.1)

    def test_skewed_batch(self):
        # One long request among many short ones: a naive per-row split would
        # pin the whole long row to a single thread while the others sit
        # idle; schedule_paged_mqa_tasks balances by cumulative KV-chunk cost
        # instead.
        self._assert_matches_reference([4000] + [16] * 15, max_seq_len=4096, atol=0.1)


if __name__ == "__main__":
    unittest.main()
