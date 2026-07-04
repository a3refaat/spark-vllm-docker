"""Parity gate for the VERIFY-grouped fused MSA indexer (q_len>1).

Reference = the SAME fused kernel at q_len=1 on multirow-expanded inputs
(per-row page table + per-row causal seqlens) -- the production spec-verify
fallback, itself gate-tested by test_fused_msa_indexer.py. The q_len-grouped
kernel shares one K stream per request and must be BIT-EXACT: per-slot MMA
math, per-slot causal masking (incl. rows whose causal boundary crosses a
128-token block seam inside the group's spread) and per-slot local force /
topk / sort / -1 padding are all identical computations in a packed tile.

Runs standalone: ``python3 tests/test_fused_msa_verify_indexer.py``.
"""

from __future__ import annotations

import sys

import torch

from b12x.attention.indexer import quantize_msa_q_fp8
from b12x.attention.indexer.msa_reference import MSA_SM_SCALE, MSA_TOPK_BLOCKS
from b12x.attention.indexer.fused_msa_indexer import run_fused_msa_indexer

sys.path.insert(0, "tests")
from test_fused_msa_indexer import (  # noqa: E402
    _PS, _HD, _make_case, _views, _check_contract)


def _verify_seqlens(bases, q_len):
    """Causal verify ladder: query j of a request with base length L attends
    to L - q_len + j + 1 tokens (matches production verify expansion)."""
    out = []
    for L in bases:
        for j in range(q_len):
            out.append(max(L - q_len + j + 1, 0))
    return out


def _run(q_bytes, weights, cache, pt, seqlens, heads, *, q_len, ctas, kv):
    kq, ks = _views(cache)
    return run_fused_msa_indexer(
        q_bytes=q_bytes, weights=weights, k_quant_bytes=kq, k_scales=ks,
        real_page_table=pt, seqlens=seqlens, num_heads=heads,
        topk=MSA_TOPK_BLOCKS, ctas_per_group=ctas,
        kv_quant="nvfp4" if kv == "nvfp4" else "none", q_len=q_len)


def test_verify_parity(kv="fp8", heads=2, width_pages=64, seed=23):
    dev = torch.device("cuda")
    W = width_pages * _PS
    cases = [
        # (bases, q_len): boundary-crossing spreads are the hard cases
        ([W], 4),
        ([130], 4),            # rows 127,128,129,130: local block crosses 0->1
        ([128], 4),            # rows 125..128: boundary-exact
        ([256, 129], 2),       # batch=2, mixed lengths
        ([3], 4),              # tiny: first row seqlen 0 selects nothing
        ([W, 257], 4),         # batch=2 full-width + just past block 2
        ([64], 2),             # single partial page
    ]
    for bases, q_len in cases:
        batch = len(bases)
        rows = batch * q_len
        seq_list = _verify_seqlens(bases, q_len)
        q_fp8, q_scale, cache, pt_rows, seqlens, _ = _make_case(
            rows=rows, heads=heads, width_pages=width_pages,
            seqlens_list=seq_list, seed=seed, kv=kv)
        if kv == "nvfp4":
            q_bytes = (
                (q_fp8.to(torch.float32) * q_scale.unsqueeze(-1))
                .to(torch.bfloat16).contiguous().view(torch.uint8))
            weights = torch.full((rows, heads), MSA_SM_SCALE,
                                 dtype=torch.float32, device=dev)
        else:
            q_bytes = q_fp8.view(torch.uint8)
            weights = (q_scale * MSA_SM_SCALE).to(torch.float32).contiguous()

        pt_group = pt_rows[:batch].contiguous()  # identical rows -> per-request
        for ctas in (1, 4, 48):
            ref = _run(q_bytes, weights, cache, pt_rows, seqlens, heads,
                       q_len=1, ctas=ctas, kv=kv)
            got = _run(q_bytes, weights, cache, pt_group, seqlens, heads,
                       q_len=q_len, ctas=ctas, kv=kv)
            torch.cuda.synchronize()
            _check_contract(got, seqlens, MSA_TOPK_BLOCKS)
            if not torch.equal(got, ref):
                bad = (got != ref).any(dim=-1).nonzero().tolist()
                raise AssertionError(
                    f"kv={kv} bases={bases} q_len={q_len} ctas={ctas} "
                    f"mismatch at (head,row) {bad}\n"
                    f"got {got[:, [r for _, r in bad[:2]]].tolist()}\n"
                    f"ref {ref[:, [r for _, r in bad[:2]]].tolist()}")
        print(f"  [verify {kv} bases={bases} q_len={q_len}] "
              f"bit-exact x ctas(1,4,48) PASS", flush=True)


def test_verify_slot_cap():
    """q_len*heads > 8 must be rejected (select-stage slot cap)."""
    dev = torch.device("cuda")
    try:
        run_fused_msa_indexer(
            q_bytes=torch.zeros((10, 4, _HD), dtype=torch.uint8, device=dev),
            weights=torch.zeros((10, 4), dtype=torch.float32, device=dev),
            k_quant_bytes=torch.zeros((1, 64, 128), dtype=torch.uint8, device=dev),
            k_scales=torch.zeros((1, 64), dtype=torch.float32, device=dev),
            real_page_table=torch.zeros((2, 4), dtype=torch.int32, device=dev),
            seqlens=torch.zeros((10,), dtype=torch.int32, device=dev),
            num_heads=4, topk=16, q_len=5)
    except ValueError as ex:
        print(f"  [slot-cap] rejected as expected: {ex}", flush=True)
        return
    raise AssertionError("q_len*heads > 8 was not rejected")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)
    test_verify_slot_cap()
    for kv in ("fp8", "nvfp4"):
        test_verify_parity(kv=kv)
    print("ALL PASS", flush=True)
