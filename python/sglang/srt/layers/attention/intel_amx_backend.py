from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import get_dsa_index_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import compute_cu_seqlens
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend, TopkTransformMethod
from sglang.srt.layers.attention.dsa.utils import compute_dsa_seqlens, pad_dsa_cache_seqlens
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel, get_spec

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class IntelAMXAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        import sgl_kernel  # noqa: F401

        super().__init__()
        self.forward_metadata = None
        self.extend_metadata = None
        self.draft_decode_metadata = None
        self.device = model_runner.device
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.max_context_len = model_runner.model_config.context_len

        # full->SWA translated out_cache_loc, computed once per forward (the only
        # set_kv_buffer is in eager forward_extend; decode writes KV in-kernel).
        self.use_sliding_window_kv_pool = (
            isinstance(self.token_to_kv_pool, SWAKVPool)
            and self.token_to_kv_pool.swa_layer_nums > 0
        )
        self.swa_out_cache_loc = None

        self.num_head = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )

        # [NB]: `layer_id` set to 0 for qwen3-next models, as not all attn layers require kv pool
        # using "full_attention_layer_id_mapping" to map which layer needs kv pool
        layer_id = 0
        if hasattr(model_runner.token_to_kv_pool, "full_attention_layer_id_mapping"):
            layer_id = [*model_runner.token_to_kv_pool.full_attention_layer_id_mapping][
                0
            ]
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(
            layer_id
        ).shape[-1]
        self.decode_attention_fwd = torch.ops.sgl_kernel.decode_attention_cpu
        self.extend_attention_fwd = torch.ops.sgl_kernel.extend_attention_cpu

        # Number of KV splits used by decode_attention_cpu; attn_logits is
        # sized [bs, num_head, num_kv_splits, v_head_dim + 1] to match.
        self.num_kv_splits = 8

        self._attn_logits_buffers: dict[tuple[int, int], torch.Tensor] = {}

        # speculative decoding params
        self.num_draft_tokens = get_spec().speculative_num_draft_tokens

        hf_config = model_runner.model_config.hf_config
        if is_deepseek_dsa(hf_config):
            self.dsa_metadata = None
            self.dsa_topk_transform_method = TopkTransformMethod.PAGED
            self.dsa_index_topk = get_dsa_index_topk(hf_config)
            assert isinstance(model_runner.page_size, int)
            self.real_page_size: int = model_runner.page_size
            self.dsa_topk_backend: DSATopKBackend = DSATopKBackend(
                model_runner.server_args.dsa_topk_backend
            )
            self.dsa_topk_indices_already_physical = (
                envs.SGLANG_DSA_FUSE_TOPK.get()
                and (
                    self.dsa_topk_backend.is_sgl_kernel()
                    or self.dsa_topk_backend.is_flashinfer()
                )
            )

    def _build_extend_metadata(self, forward_batch: ForwardBatch):
        """Resolve (seq_lens, extend_seq_lens, extend_start_loc, tree_mask) for
        forward_extend, once per forward pass.

        In TARGET_VERIFY mode the batch carries no extend_* fields, so they are
        derived from spec_info (mirrors the CUDA unified path in
        triton_backend.py); each request extends by exactly num_draft_tokens
        tokens. Outside spec decoding the fields are passed through.
        """
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        tree_mask = None

        if forward_batch.forward_mode.is_target_verify():
            spec_info = forward_batch.spec_info
            if spec_info is None:
                raise RuntimeError(
                    "spec_info is unset in TARGET_VERIFY mode; the extend_* "
                    "metadata can only be derived from spec_info for "
                    "speculative verify batches."
                )
            num_draft_tokens = spec_info.draft_token_num
            extend_seq_lens = torch.full(
                (bs,), num_draft_tokens, dtype=torch.int32, device=self.device
            )
            # Uniform extend lengths: start locations form a plain range.
            extend_start_loc = torch.arange(
                0,
                bs * num_draft_tokens,
                num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            seq_lens = forward_batch.seq_lens + num_draft_tokens
            # Speculative verify with a token tree: each draft token may only
            # attend to its ancestors among the draft tokens (the committed
            # prefix stays fully visible).
            #
            # NOTE: unlike triton_backend.py, which forwards spec_info.custom_mask
            # unconditionally, the mask is gated on tree_topk here. tree_topk == 1
            # means the draft tokens form a simple chain whose visibility is
            # exactly the kernel's built-in causal masking, and skipping the explicit
            # mask lets extend_attention_cpu take its faster mask-free path. EAGLE
            # has tree_topk == topk (> 1 for real trees); NGRAM has tree_topk == -1
            # (irregular tree); both need the mask.
            if spec_info.tree_topk != 1:
                custom_mask = spec_info.custom_mask
                if custom_mask is not None and custom_mask.numel() > 0:
                    tree_mask = custom_mask
        else:
            extend_seq_lens = forward_batch.extend_seq_lens
            extend_start_loc = forward_batch.extend_start_loc

        return seq_lens, extend_seq_lens, extend_start_loc, tree_mask

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""

        bs = forward_batch.batch_size
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = None
            self.extend_metadata = None
        elif forward_batch.forward_mode.is_target_verify():
            max_extend_len = self.num_draft_tokens
            self.extend_metadata = self._build_extend_metadata(forward_batch)
        else:
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
            self.extend_metadata = self._build_extend_metadata(forward_batch)
        self.forward_metadata = (attn_logits, max_extend_len)

        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            self.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )
        else:
            self.swa_out_cache_loc = None

        if hasattr(self, "dsa_index_topk"):
            self._init_dsa_metadata(forward_batch)

    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.real_page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

    def _cal_indexer_k_start_end(
        self,
        forward_batch: ForwardBatch,
        bs_idx,
    ):
        assert bs_idx is None
        if not forward_batch.forward_mode.is_extend_without_speculative():
            return None, None
        if forward_batch.batch_size == 0:
            empty_t = torch.empty(0, dtype=torch.int32, device=self.device)
            return (empty_t, empty_t), empty_t

        # Suppose there are two requests, with extend_seq_len = [3, 2]
        # and seq_lens = [10, 4]
        # The logits matrix looks like this, with * representing the valid logits
        # and - representing the invalid logits:
        #
        #  ********--|----
        #  *********-|----
        #  **********|----
        #  ----------|***-
        #  ----------|****
        #
        # ks = [0, 0, 0, 10, 10]
        # ke = [8, 9, 10, 13, 14]
        ks_list = []
        ke_list = []
        token_to_batch_idx = []

        q_offset = 0
        k_offset = 0

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            ks = torch.full(
                (extend_seq_len,), k_offset, dtype=torch.int32, device=self.device
            )
            kv_len = seq_len
            if forward_batch.forward_mode.is_target_verify():
                assert False, "Target verify mode is not supported on CPU"
            seq_lens_expanded = torch.arange(
                kv_len - extend_seq_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=self.device,
            )
            ke = ks + seq_lens_expanded
            ks_list.append(ks)
            ke_list.append(ke)

            # bi: The index within the selected batch bs_idx. Entries that were not selected are ignored.
            bi = i
            tb = torch.full(
                (extend_seq_len,), bi, dtype=torch.int32, device=self.device
            )
            token_to_batch_idx.append(tb)

            if bs_idx is None or i in bs_idx:  # skip batch not included in bs_idx
                q_offset += extend_seq_len
                k_offset += seq_len

        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)
        token_to_batch_idx = torch.cat(token_to_batch_idx, dim=0)
        return (ks, ke), token_to_batch_idx
    
    def _init_dsa_metadata(self, forward_batch: ForwardBatch):
        from sglang.srt.layers.attention.dsa_backend import DSAMetadata

        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        # speculative decoding is not supported
        draft_token_num = 0

        cache_seqlens_int32 = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        if forward_batch.seq_lens_cpu is not None:
            max_seqlen_k = int(
                forward_batch.seq_lens_cpu.max().item() + draft_token_num
            )
        else:
            # needs_cpu_seq_lens=False nulls the host mirror for spec-v2 relay
            # batches; graph replay uses the static page-table width, so only this
            # eager (e.g. over-capture-bs) fallback needs a length here.
            max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num
        # [b, max_seqlen_k]
        page_table = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        topk_transform_method = TopkTransformMethod.PAGED

        # CP (context parallelism) is not supported
        bs_idx_cpu = None
        # seq_len_cpu of selected sequences
        indexer_seq_lens_cpu = forward_batch.seq_lens_cpu
        indexer_seq_lens = forward_batch.seq_lens

        if forward_batch.forward_mode.is_decode_or_idle():
            extend_seq_lens_cpu = [1] * batch_size
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
            seqlens_expanded = cache_seqlens_int32
        elif forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None
            extend_seq_lens = forward_batch.extend_seq_lens

            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    for qo_len, kv_len in zip(
                        forward_batch.extend_seq_lens_cpu,
                        forward_batch.seq_lens_cpu.tolist(),
                        strict=True,
                    )
                ]
            )

            # forward_mode is never DRAFT_EXTEND (no speculative)
            if (
                any(forward_batch.extend_prefix_lens_cpu) or bs_idx_cpu is not None
            ):
                max_seqlen_q = (
                    max(extend_seq_lens_cpu) if len(extend_seq_lens_cpu) != 0 else 1
                )
                cu_seqlens_q = compute_cu_seqlens(extend_seq_lens.to(torch.int32))
            else:
                max_seqlen_q = max_seqlen_k
                cu_seqlens_q = cu_seqlens_k

            forward_batch.using_mha_one_shot_fp8_dequant = False
        else:
            assert False, f"Unsupported {forward_batch.forward_mode = }"

        indexer_k_start_end, token_to_batch_idx = self._cal_indexer_k_start_end(
            forward_batch, bs_idx_cpu
        )
        # 1D, expanded seqlens (1D means cheap to compute, so always compute it)
        dsa_cache_seqlens_int32 = compute_dsa_seqlens(
            original_seq_lens=seqlens_expanded,
            dsa_index_topk=self.dsa_index_topk,
        )
        dsa_cache_seqlens_int32 = pad_dsa_cache_seqlens(
            forward_batch, dsa_cache_seqlens_int32
        )
        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens_int32)
        dsa_cu_seqlens_q = torch.arange(
            len(dsa_cu_seqlens_k), dtype=torch.int32, device=device
        )

        self._init_flashmla_cpu_indices(
            forward_batch=forward_batch,
            batch_size=batch_size,
            device=device,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            max_seqlen_q=max_seqlen_q,
        )

        self.dsa_metadata = DSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seq_lens_sum=forward_batch.seq_lens_sum,
            page_table_1=page_table,
            page_table_1_flattened=None,
            flashmla_metadata=(
                self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens_int32,
                    seq_len_q=1,
                )
                if False
                else None
            ),
            paged_mqa_schedule_metadata=None,
            dsa_cache_seqlens_int32=dsa_cache_seqlens_int32,
            dsa_cu_seqlens_q=dsa_cu_seqlens_q,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            dsa_seqlens_expanded=seqlens_expanded,
            dsa_extend_seq_lens_list=extend_seq_lens_cpu,
            real_page_table=self._transform_table_1_to_real(page_table),
            dsa_max_seqlen_q=1,
            topk_indices_offset=None,
            indexer_k_start_end=indexer_k_start_end,
            indexer_seq_lens_cpu=indexer_seq_lens_cpu,
            indexer_seq_lens=indexer_seq_lens,
            token_to_batch_idx=token_to_batch_idx,
        )

    def _init_flashmla_cpu_indices(
        self,
        forward_batch: ForwardBatch,
        batch_size: int,
        device: torch.device,
        cache_seqlens_int32: torch.Tensor,
        max_seqlen_k: int,
        page_table: torch.Tensor,
        max_seqlen_q: int,
    ):
        """Precompute (once per forward pass, reused by every layer) the
        ``indices``/``topk_length`` tensors needed to run CPU attention via
        ``flash_mla_with_kvcache_cpu`` instead of extend_attention_cpu /
        decode_attention_cpu. This does *not* perform real DSA top-k
        selection: it simply gathers the whole (causal) KV range for each
        query token through the sparse-kernel interface, using -1 as the
        sentinel for out-of-range / future positions.
        """
        page_table_i32 = page_table.to(torch.int32)
        col_idx = torch.arange(max_seqlen_k, dtype=torch.int32, device=device)

        if forward_batch.forward_mode.is_decode_or_idle():
            mask = col_idx[None, :] < cache_seqlens_int32[:, None]
            indices = page_table_i32.clone()
            indices[~mask] = -1
            self._fmla_indices = indices.unsqueeze(1)  # [bs, 1, max_seqlen_k]
            self._fmla_topk_length = cache_seqlens_int32.to(torch.int32).contiguous()
            self._fmla_s_q = 1
        else:
            prefix_lens_cpu = (
                forward_batch.extend_prefix_lens_cpu
                if forward_batch.extend_prefix_lens_cpu is not None
                else [0] * batch_size
            )
            prefix_lens = torch.tensor(prefix_lens_cpu, dtype=torch.int32, device=device)
            s_idx = torch.arange(max_seqlen_q, dtype=torch.int32, device=device)
            # kv length visible to query row s (0-indexed within the chunk)
            row_len = prefix_lens[:, None] + s_idx[None, :] + 1  # [bs, s_q]
            mask = col_idx[None, None, :] < row_len[:, :, None]  # [bs, s_q, max_seqlen_k]
            indices = (
                page_table_i32[:, None, :]
                .expand(batch_size, max_seqlen_q, max_seqlen_k)
                .clone()
            )
            indices[~mask] = -1
            self._fmla_indices = indices
            self._fmla_topk_length = cache_seqlens_int32.to(torch.int32).contiguous()
            self._fmla_s_q = max_seqlen_q

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        from sglang.srt.layers.attention.dsa_backend import DSAIndexerMetadata

        if not hasattr(self, "dsa_metadata") or self.dsa_metadata is None:
            return None
        return DSAIndexerMetadata(
            attn_metadata=self.dsa_metadata,
            topk_transform_method=self.dsa_topk_transform_method,
            topk_backend=self.dsa_topk_backend,
        )

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        max_extend_len = None
        self.forward_metadata = (attn_logits, max_extend_len)
        self.extend_metadata = None

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if save_kv_cache and k is not None and v is not None:
            # Cross-attention never writes to the SWA pool, so only thread the
            # full->SWA location for non-cross-attention layers.
            swa_loc = None if layer.is_cross_attention else self.swa_out_cache_loc
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(cache_loc, swa_loc), k, v
            )

        # Precomputed once per forward pass in init_forward_metadata (spec
        # verify batches carry no extend_* fields; see _build_extend_metadata).
        seq_lens, extend_seq_lens, extend_start_loc, tree_mask = self.extend_metadata

        if hasattr(self, "dsa_index_topk"):
            return self._forward_flashmla_cpu(
                q, layer, forward_batch, topk_indices=topk_indices
            )

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        _, max_extend_len = self.forward_metadata
        seq_lens = forward_batch.seq_lens
        if seq_lens.dtype != torch.int64:
            seq_lens = seq_lens.to(torch.int64)

        # Gemma4's KV-shared layers pass k=v=None - the layer they share with
        # already wrote their extend K/V to the cache
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k,
            v,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
            layer.is_cross_attention,
            layer.sliding_window_size + 1,
            forward_batch.encoder_lens,
            sinks,
            tree_mask,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _get_real_topk_indices(
        self,
        topk_indices: torch.Tensor,
        forward_batch: ForwardBatch,
        batch_size: int,
        s_q: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the indexer's logical top-k sequence positions into
        physical KV-pool slot ids (``indices``) usable by
        ``flash_mla_with_kvcache_cpu``, plus a matching ``topk_length`` bound.

        ``topk_length`` is only a loop-bound optimization for the kernel (real
        masking uses -1 sentinels inside ``indices``), so it is safe (if a bit
        conservative performance-wise) to always scan the full topk width.
        """
        from sglang.srt.layers.attention.dsa.transform_index import (
            transform_index_page_table_decode,
            transform_index_page_table_prefill,
        )

        assert self.dsa_metadata is not None
        page_table_1 = self.dsa_metadata.page_table_1
        index_topk = topk_indices.shape[-1]

        # With SGLANG_DSA_FUSE_TOPK on a fusion-capable backend (sgl-kernel / flashinfer),
        # the indexer's fused kernel already folds the raw-index -> page-table transform
        # in, so topk_indices here are ALREADY page_size=1 physical slots; re-running the
        # transform below would gather through the page table a second time (mirrors GPU's
        # `_get_fused_topk_page_table` passthrough in dsa_backend.py).
        if s_q == 1:
            indices_phys = (
                topk_indices
                if self.dsa_topk_indices_already_physical
                else transform_index_page_table_decode(
                    page_table=page_table_1,
                    topk_indices=topk_indices,
                    page_size=1,
                )
            )
            indices = indices_phys.unsqueeze(1)  # [bs, 1, index_topk]
        else:
            indices_phys = (
                topk_indices
                if self.dsa_topk_indices_already_physical
                else transform_index_page_table_prefill(
                    page_table=page_table_1,
                    topk_indices=topk_indices,
                    extend_lens_cpu=self.dsa_metadata.dsa_extend_seq_lens_list,
                    page_size=1,
                )
            )
            start_locs = forward_batch.extend_start_loc.tolist()
            ext_lens = forward_batch.extend_seq_lens_cpu
            indices = indices_phys.new_full((batch_size, s_q, index_topk), -1)
            for b in range(batch_size):
                length = ext_lens[b]
                start = start_locs[b]
                indices[b, :length] = indices_phys[start : start + length]

        topk_length = torch.full(
            (batch_size,), index_topk, dtype=torch.int32, device=device
        )
        return indices, topk_length

    def _forward_flashmla_cpu(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run attention (both prefill/extend and decode) through the CPU
        sparse-MLA kernel ``flash_mla_with_kvcache_cpu`` instead of
        extend_attention_cpu / decode_attention_cpu.

        When ``topk_indices`` (the real DSA indexer top-k logical positions)
        is provided, it is transformed into physical KV-pool slot ids and
        used to restrict attention to the real sparse top-k set. Otherwise
        falls back to the dense (fully causal, via the sparse-kernel API)
        ``indices``/``topk_length`` precomputed by ``_init_flashmla_cpu_indices``.
        """
        from sgl_kernel.flash_mla import flash_mla_with_kvcache_cpu

        num_heads = layer.tp_q_head_num
        head_dim_qk = layer.qk_head_dim
        head_dim_v = layer.v_head_dim
        batch_size = forward_batch.batch_size
        s_q = self._fmla_s_q

        q_flat = q.reshape(-1, num_heads, head_dim_qk)
        if q_flat.dtype != torch.bfloat16:
            q_flat = q_flat.to(torch.bfloat16)

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        if k_cache.dtype != torch.bfloat16:
            k_cache = k_cache.to(torch.bfloat16)
        k_cache = k_cache.unsqueeze(0).contiguous()  # [1, capacity, 1, head_dim_qk]

        if topk_indices is not None:
            fmla_indices, fmla_topk_length = self._get_real_topk_indices(
                topk_indices, forward_batch, batch_size, s_q, q_flat.device
            )
        else:
            fmla_indices, fmla_topk_length = self._fmla_indices, self._fmla_topk_length

        if s_q == 1:
            q_view = q_flat.view(batch_size, 1, num_heads, head_dim_qk)
            out, _ = flash_mla_with_kvcache_cpu(
                q=q_view,
                k_cache=k_cache,
                block_table=None,
                cache_seqlens=None,
                head_dim_v=head_dim_v,
                softmax_scale=layer.scaling,
                is_fp8_kvcache=False,
                indices=fmla_indices,
                topk_length=fmla_topk_length,
            )
            o_flat = out.view(-1, num_heads, head_dim_v)
        else:
            start_locs = forward_batch.extend_start_loc.tolist()
            ext_lens = forward_batch.extend_seq_lens_cpu

            q_padded = q_flat.new_zeros((batch_size, s_q, num_heads, head_dim_qk))
            for b in range(batch_size):
                length = ext_lens[b]
                start = start_locs[b]
                q_padded[b, :length] = q_flat[start : start + length]

            out, _ = flash_mla_with_kvcache_cpu(
                q=q_padded,
                k_cache=k_cache,
                block_table=None,
                cache_seqlens=None,
                head_dim_v=head_dim_v,
                softmax_scale=layer.scaling,
                is_fp8_kvcache=False,
                indices=fmla_indices,
                topk_length=fmla_topk_length,
            )

            o_flat = q_flat.new_empty((q_flat.shape[0], num_heads, head_dim_v))
            for b in range(batch_size):
                length = ext_lens[b]
                start = start_locs[b]
                o_flat[start : start + length] = out[b, :length]

        if layer.qk_head_dim != layer.v_head_dim:
            return o_flat.reshape(-1, num_heads * head_dim_v)
        return o_flat

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if hasattr(self, "dsa_index_topk"):
            if save_kv_cache and k is not None and v is not None:
                self.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            return self._forward_flashmla_cpu(
                q, layer, forward_batch, topk_indices=topk_indices
            )

        if self.draft_decode_metadata is not None:
            req_to_token, seq_lens, req_pool_indices = self.draft_decode_metadata
        else:
            req_to_token = self.req_to_token_pool.req_to_token
            req_pool_indices = forward_batch.req_pool_indices
            seq_lens = forward_batch.seq_lens

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        seq_lens = forward_batch.seq_lens
        if seq_lens.dtype != torch.int64:
            seq_lens = seq_lens.to(torch.int64)

        if layer.v_head_dim == self.v_head_dim and layer.tp_q_head_num == self.num_head:
            attn_logits, _ = self.forward_metadata
        else:
            # This layer's shape differs from the model-wide metadata buffer -
            # size from the same seq_lens the kernel derives num_seqs from
            attn_logits = self._get_attn_logits_buffer(
                seq_lens.shape[0], layer.tp_q_head_num, layer.v_head_dim
            )

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k,
            v,
            cache_loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            layer.scaling,
            layer.logit_cap,
            layer.is_cross_attention,
            layer.sliding_window_size + 1,
            forward_batch.encoder_lens,
            sinks,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _get_attn_logits_buffer(
        self, num_seqs: int, num_heads: int, v_head_dim: int
    ) -> torch.Tensor:
        key = (num_heads, v_head_dim)
        buffer = self._attn_logits_buffers.get(key)
        if buffer is None or buffer.shape[0] < num_seqs:
            # decode_attention_cpu writes every element it later reads, so the
            # buffer needs no initialization and can be reused; it only grows.
            buffer = torch.empty(
                (num_seqs, num_heads, self.num_kv_splits, v_head_dim + 1),
                dtype=torch.float32,
                device=self.device,
            )
            self._attn_logits_buffers[key] = buffer
        return buffer[:num_seqs]

    def support_triton(self):
        return False


class IntelAMXMultiStepDraftBackend:
    """
    Wrap multiple intel amx attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sgl_kernel import build_draft_decode_metadata_cpu

        self.build_draft_decode_metadata = build_draft_decode_metadata_cpu
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: list[IntelAMXAttnBackend] = []
        for _ in range(self.speculative_num_steps - 1):
            self.attn_backends.append(IntelAMXAttnBackend(model_runner))
        self.device = model_runner.device
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        num_seqs = forward_batch.batch_size
        topk = self.topk
        bs = num_seqs * topk
        num_steps = self.speculative_num_steps
        req_to_token = self.attn_backends[0].req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        pool_len = self.pool_len
        num_head = self.attn_backends[0].num_head
        v_head_dim = self.attn_backends[0].v_head_dim
        device = self.device

        # Build expanded req_to_token via C++ kernel
        req_to_token_draft = self.build_draft_decode_metadata(
            req_to_token,
            forward_batch.req_pool_indices,
            seq_lens,
            topk,
            num_steps,
            pool_len,
        )

        req_pool_indices_expanded = torch.arange(bs, dtype=torch.int64, device=device)

        num_kv_splits = self.attn_backends[0].num_kv_splits
        for step in range(num_steps - 1):
            # Each candidate sees prefix + (step + 1) draft tokens.
            seq_lens_expanded = seq_lens.repeat_interleave(topk) + step + 1
            attn_logits = torch.zeros(
                (bs, num_head, num_kv_splits, v_head_dim + 1),
                dtype=torch.float32,
                device=device,
            )
            self.attn_backends[step].forward_metadata = (attn_logits, None)
            self.attn_backends[step].draft_decode_metadata = (
                req_to_token_draft,
                seq_lens_expanded,
                req_pool_indices_expanded,
            )
