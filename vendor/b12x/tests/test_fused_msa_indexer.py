"""Parity + contract tests for the one-pass fused MSA indexer (Phase 5).

Reference = the production MSA decode path (``msa_q2k_indices_decode``), which
is itself gate-tested against the pure-torch msa_reference. Runs standalone:
``python3 tests/test_fused_msa_indexer.py``.
"""

from __future__ import annotations

import sys

import torch

from b12x.attention.indexer import (
    IndexerPagedDecodeMetadata,
    msa_q2k_indices_decode,
    quantize_msa_q_fp8,
)
from b12x.attention.indexer.msa_reference import MSA_SM_SCALE, MSA_TOPK_BLOCKS
from b12x.attention.indexer.reference import pack_index_k_cache_reference
from b12x.attention.indexer.fused_msa_indexer import (
    msa_fused_scratch_shapes,
    run_fused_msa_indexer,
)

_PS = 64
_HD = 128
_BLK = 16
_E2M1_MAX = 6.0
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _views(index_k_cache: torch.Tensor):
    # production runtime views (fp8: u8 [p,64,128] + strided f32 [p,64];
    # nvfp4: e2m1 [p,64,64] + e4m3 [p,64,8] u8)
    from b12x.attention.indexer.kernel import _split_index_k_cache_runtime_views

    return _split_index_k_cache_runtime_views(index_k_cache)


def _pack_index_k_cache_nvfp4(k: torch.Tensor) -> torch.Tensor:
    """Self-contained port of the gate-C harness nvfp4 page-major packer."""
    km = k.reshape(-1, _HD).float()
    n = km.shape[0]
    assert n % _PS == 0
    npages = n // _PS
    xb = km.reshape(n, _HD // _BLK, _BLK)
    scale_e4 = (xb.abs().amax(dim=-1) / _E2M1_MAX).clamp(min=1e-6).to(torch.float8_e4m3fn)
    xn = (xb / scale_e4.float().unsqueeze(-1)).reshape(n, _HD)
    sign = (xn < 0).to(torch.uint8)
    mag = _E2M1_MAG.to(km.device)
    idx = (xn.abs().unsqueeze(-1) - mag).abs().argmin(dim=-1).to(torch.uint8)
    codes = (sign << 3) | idx
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    sc = scale_e4.view(torch.uint8)
    data_b, scale_b = _PS * (_HD // 2), _PS * (_HD // _BLK)
    cache = torch.zeros((npages, data_b + scale_b), dtype=torch.uint8, device=km.device)
    cache[:, :data_b] = packed.reshape(npages, data_b)
    cache[:, data_b:] = sc.reshape(npages, scale_b)
    return cache.contiguous()


def _make_case(*, rows, heads, width_pages, seqlens_list, seed, kv="fp8"):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    q = torch.randn((rows, heads, _HD), generator=gen, dtype=torch.float32, device=dev) / 3
    q_fp8, q_scale = quantize_msa_q_fp8(q)
    k = torch.randn((width_pages * _PS, _HD), generator=gen, dtype=torch.float32, device=dev) / 3
    index_k_cache = (
        _pack_index_k_cache_nvfp4(k) if kv == "nvfp4"
        else pack_index_k_cache_reference(k)
    )
    pt = (
        torch.arange(width_pages, dtype=torch.int32, device=dev)
        .unsqueeze(0)
        .expand(rows, width_pages)
        .contiguous()
    )
    seqlens = torch.tensor(seqlens_list, dtype=torch.int32, device=dev)
    assert seqlens.shape[0] == rows
    meta = IndexerPagedDecodeMetadata(real_page_table=pt, cache_seqlens_int32=seqlens)
    return q_fp8, q_scale, index_k_cache, pt, seqlens, meta


def _check_contract(got, seqlens, topk):
    heads, rows, k = got.shape
    for h in range(heads):
        for r in range(rows):
            row = got[h, r].tolist()
            vals = [v for v in row if v != -1]
            assert vals == sorted(vals), f"not ascending: {row}"
            assert len(vals) == len(set(vals)), f"duplicates: {row}"
            tail = row[len(vals):]
            assert all(v == -1 for v in tail), f"pad not trailing: {row}"
            L = int(seqlens[r])
            if L > 0:
                local = (L - 1) // 128
                assert local in vals, f"local block {local} missing: {row} (L={L})"
            else:
                assert not vals, f"seqlen 0 must select nothing: {row}"


def _run_fused(q_fp8, q_scale, index_k_cache, pt, seqlens, heads, topk, **kw):
    kq, ks = _views(index_k_cache)
    weights = (q_scale * MSA_SM_SCALE).to(torch.float32).contiguous()
    return run_fused_msa_indexer(
        q_bytes=q_fp8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=kq,
        k_scales=ks,
        real_page_table=pt,
        seqlens=seqlens,
        num_heads=heads,
        topk=topk,
        **kw,
    )


def _overlap(a, b):
    sa = {v for v in a.tolist() if v != -1}
    sb = {v for v in b.tolist() if v != -1}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def test_fused_msa_matches_production(rows=3, heads=2, width_pages=64, seed=7):
    torch.manual_seed(seed)
    W = width_pages * _PS
    q_fp8, q_scale, cache, pt, seqlens, meta = _make_case(
        rows=rows, heads=heads, width_pages=width_pages,
        seqlens_list=[W, max(W - 63, 1), 129][:rows] + [W] * max(0, rows - 3),
        seed=seed,
    )
    ref = msa_q2k_indices_decode(
        q_fp8=q_fp8, q_scale=q_scale, index_k_cache=cache, metadata=meta,
        topk=MSA_TOPK_BLOCKS,
    )
    for ctas in (1, 4, 33):
        got = _run_fused(q_fp8, q_scale, cache, pt, seqlens, heads, MSA_TOPK_BLOCKS,
                         ctas_per_group=ctas)
        torch.cuda.synchronize()
        _check_contract(got, seqlens, MSA_TOPK_BLOCKS)
        for h in range(heads):
            for r in range(rows):
                ov = _overlap(got[h, r], ref[h, r])
                assert ov >= 15 / 16, (
                    f"ctas={ctas} h={h} r={r}: overlap {ov:.3f}\n"
                    f"got {got[h, r].tolist()}\nref {ref[h, r].tolist()}"
                )
        exact = (got == ref).all().item()
        print(f"  [parity rows={rows} heads={heads} W={width_pages}p ctas={ctas}] "
              f"exact={exact} PASS", flush=True)


def test_fused_msa_edge_lengths(heads=2, width_pages=16, seed=11):
    W = width_pages * _PS
    lens = [0, 1, 64, 127, 128, 129, W]
    rows = len(lens)
    q_fp8, q_scale, cache, pt, seqlens, meta = _make_case(
        rows=rows, heads=heads, width_pages=width_pages, seqlens_list=lens, seed=seed)
    ref = msa_q2k_indices_decode(
        q_fp8=q_fp8, q_scale=q_scale, index_k_cache=cache, metadata=meta,
        topk=MSA_TOPK_BLOCKS)
    got = _run_fused(q_fp8, q_scale, cache, pt, seqlens, heads, MSA_TOPK_BLOCKS,
                     ctas_per_group=4)
    torch.cuda.synchronize()
    _check_contract(got, seqlens, MSA_TOPK_BLOCKS)
    assert (got == ref).all().item(), f"edge mismatch:\n{got}\n{ref}"
    print(f"  [edges lens={lens}] exact PASS", flush=True)


def test_fused_msa_heads4(width_pages=32, seed=13):
    W = width_pages * _PS
    q_fp8, q_scale, cache, pt, seqlens, meta = _make_case(
        rows=2, heads=4, width_pages=width_pages, seqlens_list=[W, W // 2], seed=seed)
    ref = msa_q2k_indices_decode(
        q_fp8=q_fp8, q_scale=q_scale, index_k_cache=cache, metadata=meta,
        topk=MSA_TOPK_BLOCKS)
    got = _run_fused(q_fp8, q_scale, cache, pt, seqlens, 4, MSA_TOPK_BLOCKS,
                     ctas_per_group=8)
    torch.cuda.synchronize()
    _check_contract(got, seqlens, MSA_TOPK_BLOCKS)
    assert (got == ref).all().item()
    print("  [heads=4] exact PASS", flush=True)


def test_fused_msa_nvfp4_decode(heads=2, width_pages=32, seed=19):
    """nvfp4 one-pass decode vs the production nvfp4 fallback route.

    Two q flavors: (a) fp8-roundtripped q (matches the fallback's quantized q
    -> near-exact expectation), (b) raw bf16 q (production wiring; strictly
    higher precision so near-ties may flip -> overlap bound + contract).
    """
    dev = torch.device("cuda")
    W = width_pages * _PS
    rows = 3
    q_fp8, q_scale, cache, pt, seqlens, meta = _make_case(
        rows=rows, heads=heads, width_pages=width_pages,
        seqlens_list=[W, W - 63, 130], seed=seed, kv="nvfp4")
    ref = msa_q2k_indices_decode(
        q_fp8=q_fp8, q_scale=q_scale, index_k_cache=cache, metadata=meta,
        topk=MSA_TOPK_BLOCKS)
    kq, ks = _views(cache)
    sm_w = torch.full((rows, heads), MSA_SM_SCALE, dtype=torch.float32, device=dev)

    # (a) fp8-roundtripped q: same quantized q the fallback scores
    qdq = (q_fp8.to(torch.float32) * q_scale.unsqueeze(-1)).to(torch.bfloat16).contiguous()
    got = run_fused_msa_indexer(
        q_bytes=qdq.view(torch.uint8), weights=sm_w, k_quant_bytes=kq, k_scales=ks,
        real_page_table=pt, seqlens=seqlens, num_heads=heads, topk=MSA_TOPK_BLOCKS,
        ctas_per_group=8, kv_quant="nvfp4")
    torch.cuda.synchronize()
    _check_contract(got, seqlens, MSA_TOPK_BLOCKS)
    n_bad = 0
    for h in range(heads):
        for r in range(rows):
            ov = _overlap(got[h, r], ref[h, r])
            assert ov >= 15 / 16, (f"qdq h={h} r={r} overlap {ov:.3f}\n"
                                   f"{got[h, r].tolist()}\n{ref[h, r].tolist()}")
            n_bad += int((got[h, r] != ref[h, r]).any())
    print(f"  [nvfp4 qdq] rows-with-diff={n_bad}/{heads*rows} (ties only) PASS", flush=True)

    # (b) raw bf16 q (production wiring)
    qraw = torch.randn((rows, heads, _HD), dtype=torch.float32, device=dev).to(torch.bfloat16) / 3
    qraw = qraw.contiguous()
    qf8, qs = quantize_msa_q_fp8(qraw.float())
    ref2 = msa_q2k_indices_decode(
        q_fp8=qf8, q_scale=qs, index_k_cache=cache, metadata=meta, topk=MSA_TOPK_BLOCKS)
    got2 = run_fused_msa_indexer(
        q_bytes=qraw.view(torch.uint8), weights=sm_w, k_quant_bytes=kq, k_scales=ks,
        real_page_table=pt, seqlens=seqlens, num_heads=heads, topk=MSA_TOPK_BLOCKS,
        ctas_per_group=4, kv_quant="nvfp4")
    torch.cuda.synchronize()
    _check_contract(got2, seqlens, MSA_TOPK_BLOCKS)
    for h in range(heads):
        for r in range(rows):
            ov = _overlap(got2[h, r], ref2[h, r])
            assert ov >= 14 / 16, (f"raw h={h} r={r} overlap {ov:.3f}\n"
                                   f"{got2[h, r].tolist()}\n{ref2[h, r].tolist()}")
    print("  [nvfp4 raw-bf16-q] overlap>=14/16 + contract PASS", flush=True)


def test_fused_msa_nvfp4_graph_replay(heads=2, width_pages=32, seed=23):
    dev = torch.device("cuda")
    rows = 2
    W = width_pages * _PS
    q_fp8, q_scale, cache, pt, seqlens, meta = _make_case(
        rows=rows, heads=heads, width_pages=width_pages, seqlens_list=[W, W],
        seed=seed, kv="nvfp4")
    kq, ks = _views(cache)
    qdq = (q_fp8.to(torch.float32) * q_scale.unsqueeze(-1)).to(torch.bfloat16).contiguous()
    sm_w = torch.full((rows, heads), MSA_SM_SCALE, dtype=torch.float32, device=dev)
    out = torch.full((heads, rows, MSA_TOPK_BLOCKS), -2, dtype=torch.int32, device=dev)
    slab_shape, state_shape = msa_fused_scratch_shapes(rows, heads, width_pages)
    slab = torch.empty(slab_shape, dtype=torch.float32, device=dev)
    state = torch.zeros(state_shape, dtype=torch.int32, device=dev)

    def step():
        run_fused_msa_indexer(
            q_bytes=qdq.view(torch.uint8), weights=sm_w, k_quant_bytes=kq, k_scales=ks,
            real_page_table=pt, seqlens=seqlens, num_heads=heads, topk=MSA_TOPK_BLOCKS,
            out_indices=out, ctas_per_group=8, slab=slab, state=state,
            state_preinitialized=True, kv_quant="nvfp4")

    step(); step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for lens in ([200, 90], [W, 1500], [1, W]):
        seqlens.copy_(torch.tensor(lens, dtype=torch.int32, device=dev))
        g.replay()
        torch.cuda.synchronize()
        got = out.clone()
        meta2 = IndexerPagedDecodeMetadata(real_page_table=pt, cache_seqlens_int32=seqlens)
        ref = msa_q2k_indices_decode(
            q_fp8=q_fp8, q_scale=q_scale, index_k_cache=cache, metadata=meta2,
            topk=MSA_TOPK_BLOCKS)
        _check_contract(got, seqlens, MSA_TOPK_BLOCKS)
        for h in range(heads):
            for r in range(rows):
                ov = _overlap(got[h, r], ref[h, r])
                assert ov >= 15 / 16, f"graph nvfp4 lens={lens} h={h} r={r}: {ov:.3f}"
        print(f"  [nvfp4 graph lens={lens}] PASS", flush=True)


def test_fused_msa_graph_replay(heads=2, width_pages=64, seed=17):
    """Capture once with caller-owned scratch; replay across growing seqlens."""
    dev = torch.device("cuda")
    rows = 2
    W = width_pages * _PS
    q_fp8, q_scale, cache, pt, seqlens, meta = _make_case(
        rows=rows, heads=heads, width_pages=width_pages, seqlens_list=[W, W], seed=seed)
    kq, ks = _views(cache)
    weights = (q_scale * MSA_SM_SCALE).to(torch.float32).contiguous()
    out = torch.full((heads, rows, MSA_TOPK_BLOCKS), -2, dtype=torch.int32, device=dev)
    slab_shape, state_shape = msa_fused_scratch_shapes(rows, heads, width_pages)
    slab = torch.empty(slab_shape, dtype=torch.float32, device=dev)
    state = torch.zeros(state_shape, dtype=torch.int32, device=dev)
    ctas = 8

    def step():
        run_fused_msa_indexer(
            q_bytes=q_fp8.view(torch.uint8), weights=weights,
            k_quant_bytes=kq, k_scales=ks, real_page_table=pt, seqlens=seqlens,
            num_heads=heads, topk=MSA_TOPK_BLOCKS, out_indices=out,
            ctas_per_group=ctas, slab=slab, state=state, state_preinitialized=True)

    # warm (compiles + validates state self-reset before capture)
    step(); step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for lens in ([130, 77], [1000, 4096], [W, W - 1], [512, W]):
        seqlens.copy_(torch.tensor(lens, dtype=torch.int32, device=dev))
        g.replay()
        torch.cuda.synchronize()
        got = out.clone()
        meta2 = IndexerPagedDecodeMetadata(real_page_table=pt, cache_seqlens_int32=seqlens)
        ref = msa_q2k_indices_decode(
            q_fp8=q_fp8, q_scale=q_scale, index_k_cache=cache, metadata=meta2,
            topk=MSA_TOPK_BLOCKS)
        _check_contract(got, seqlens, MSA_TOPK_BLOCKS)
        assert (got == ref).all().item(), f"graph lens={lens}:\n{got}\n{ref}"
        print(f"  [graph lens={lens}] exact PASS", flush=True)


def main():
    torch.set_printoptions(linewidth=200)
    for fn in (
        test_fused_msa_matches_production,
        test_fused_msa_edge_lengths,
        test_fused_msa_heads4,
        test_fused_msa_graph_replay,
        test_fused_msa_nvfp4_decode,
        test_fused_msa_nvfp4_graph_replay,
    ):
        print(f"== {fn.__name__}", flush=True)
        fn()
    print("== ALL PASS ==", flush=True)


if __name__ == "__main__":
    sys.exit(main())
