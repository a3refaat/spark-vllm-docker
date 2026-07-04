#!/bin/bash
# Bound the dense-kvarn FA materialize scratch (KVARN_FA_SCRATCH_CAP_TOKENS)
# and add a KV-windowed chunked materialize+FA path (LSE merge) for B==1
# contexts beyond the cap. Why: the stock sizing floors the scratch at
# max_model_len tokens; for a 32-kv-head EAGLE draft layer that is
# 16.4 KB/token (2x1.06 GiB @131k, 2x2.1 @262k) charged against the KV
# budget at profiling. Verify steps already bypass the scratch (fused
# verify); the only long-context user is chunked-prefill continuation,
# which windows cleanly: full windows are entirely visible (non-causal),
# the diagonal tail keeps FA's bottom-right causal mask, partials merge
# with flash-decoding LSE math. Apply AFTER mods/add-kvarn-kv-quant.
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import py_compile
P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/kvarn_attn.py")
s = P.read_text()
if "KVARN_FA_SCRATCH_CAP_TOKENS" in s:
    print("already applied"); raise SystemExit

# 1) sizing cap
old = """        FA_SCRATCH_CAP = 262144
        fa_rows = max(
            min(self._max_num_seqs * self._max_model_len, FA_SCRATCH_CAP),
            self._max_model_len,
            4096,
        )"""
assert s.count(old) == 1, "fa_rows sizing anchor"
s = s.replace(old, old + """
        # (mod: fix-kvarn-dense-fa-scratch-cap) hard cap; longer contexts
        # take the KV-windowed chunked path below (same math, bounded mem).
        _fa_cap = int(os.environ.get("KVARN_FA_SCRATCH_CAP_TOKENS", "0") or 0)
        if _fa_cap > 0:
            fa_rows = max(min(fa_rows, _fa_cap), 4096)""", 1)

# 2) routing
old = """        total_k = int(cu_k[-1].item())
        if total_k <= 0 or total_k > self._fa_K_buf.shape[0]:
            return self._decode_path_slow(q, kv_cache, md)"""
assert s.count(old) == 1, "routing anchor"
s = s.replace(old, """        total_k = int(cu_k[-1].item())
        if total_k <= 0:
            return self._decode_path_slow(q, kv_cache, md)
        if total_k > self._fa_K_buf.shape[0]:
            if B == 1 and q.shape[0] + self.kvarn_config.group <= self._fa_K_buf.shape[0]:
                return self._multi_query_cached_chunked(q, kv_cache, md)
            return self._decode_path_slow(q, kv_cache, md)""", 1)

# 3) chunked method
anchor = "    def _mixed_batch_path(\n"
assert s.count(anchor) == 1, "method anchor"
method = '''    def _multi_query_cached_chunked(self, q, kv_cache, md):
        """(mod: fix-kvarn-dense-fa-scratch-cap) B==1 materialize+FA in KV
        windows through the bounded scratch. Full windows lie entirely below
        the cached boundary (visible to every query row: causal=False); the
        diagonal tail uses FA's bottom-right causal mask; window partials
        merge with flash-decoding LSE math. Eager-only route."""
        import torch.nn.functional as F
        from vllm.v1.attention.ops.triton_kvarn_decode import (
            _kvarn_build_packed_kv_kernel,
        )
        cfg = self.kvarn_config
        group = cfg.group
        D = self.head_size
        Hk = self.num_kv_heads
        nq = q.shape[0]
        seq_len = int(md.seq_lens[0].item())
        cached = seq_len - nq
        fa_rows = self._fa_K_buf.shape[0]
        W = max(((fa_rows - nq) // group) * group, group)
        n_full = max(cached // W, 0)
        H16 = (
            self._H_fp16
            if self._H_fp16 is not None
            else self._hadamard(q.device).to(torch.float16)
        )
        q_rot = torch.mm(q.reshape(-1, D), H16).view(nq, self.num_heads, D)
        cu_q = torch.tensor([0, nq], dtype=torch.int32, device=q.device)
        acc = None
        lse = None
        w0 = 0
        while w0 < seq_len:
            is_tail = w0 >= n_full * W
            w1 = seq_len if is_tail else min(w0 + W, n_full * W)
            wlen = w1 - w0
            blk0 = w0 // group
            wblocks = (wlen + group - 1) // group
            bt = md.block_table[:1, blk0 : blk0 + wblocks]
            w_seq = torch.tensor([wlen], dtype=torch.int32, device=q.device)
            w_cu_k = torch.tensor([0, wlen], dtype=torch.int32, device=q.device)
            K_p = self._fa_K_buf
            V_p = self._fa_V_buf
            _kvarn_build_packed_kv_kernel[(wblocks, Hk)](
                bt,
                w_seq,
                w_cu_k,
                self._block_to_slot_t,
                kv_cache,
                self._tail_K_pool,
                self._tail_V_pool,
                K_p,
                V_p,
                bt.stride(0),
                kv_cache.stride(0),
                kv_cache.stride(1),
                self._tail_K_pool.stride(0),
                self._tail_K_pool.stride(1),
                self._tail_K_pool.stride(2),
                K_p.stride(0),
                K_p.stride(1),
                MAX_BLOCKS_PER_REQ=wblocks,
                D=D,
                GROUP=group,
                K_BITS=cfg.key_bits,
                V_BITS=cfg.value_bits,
                NUM_BLOCKS_LOOKUP=self._block_lookup_size,
                K_PACKED_OFFSET=cfg.k_packed_offset,
                K_S_COL_OFFSET=cfg.k_s_col_offset,
                K_ZP_OFFSET=cfg.k_zp_offset,
                K_S_ROW_OFFSET=cfg.k_s_row_offset,
                V_PACKED_OFFSET=cfg.v_packed_offset,
                V_S_COL_OFFSET=cfg.v_s_col_offset,
                V_S_ROW_OFFSET=cfg.v_s_row_offset,
                V_ZP_OFFSET=cfg.v_zp_offset,
                num_warps=4,
                num_stages=2,
            )
            kwargs = {} if self.fa_version is None else {"fa_version": self.fa_version}
            out_w, lse_w = flash_attn_varlen_func(
                q=q_rot,
                k=K_p[:wlen],
                v=V_p[:wlen],
                cu_seqlens_q=cu_q,
                cu_seqlens_k=w_cu_k,
                max_seqlen_q=nq,
                max_seqlen_k=wlen,
                softmax_scale=self.scale,
                causal=bool(is_tail),
                return_softmax_lse=True,
                **kwargs,
            )
            o32 = out_w.to(torch.float32)
            l32 = lse_w.to(torch.float32)
            if acc is None:
                acc, lse = o32, l32
            else:
                m = torch.maximum(lse, l32)
                a = torch.exp(lse - m)
                b = torch.exp(l32 - m)
                aw = a.transpose(0, 1).unsqueeze(-1)
                bw = b.transpose(0, 1).unsqueeze(-1)
                acc = (acc * aw + o32 * bw) / (aw + bw)
                lse = m + torch.log(a + b)
            w0 = w1
        out_rot = acc.to(q.dtype).reshape(nq * self.num_heads, D)
        return torch.mm(out_rot, H16).view(nq, self.num_heads, D)

'''
s = s.replace(anchor, method + anchor, 1)
P.write_text(s)
py_compile.compile(str(P), doraise=True)
print("dense FA scratch cap + chunked path applied")
PY
