"""Dense (non-block-sparse) decode split-KV: planner, eager, and graph replay.

Phase 2 of the decode-throughput plan: dense decode rides the same
partial+merge rails as MSA decode / dense verify. Unsplit stays the default;
split is opt-in via force_split_kv at plan/bind time.

Standalone: `python3 tests/test_paged_dense_decode_split_kv.py` (or pytest).
Requires a GPU; the optional nvfp4 case additionally requires the vLLM glue.
"""

from __future__ import annotations

import math
import sys

import torch

from b12x.attention.paged.planner import create_paged_plan
from b12x.attention.paged.reference import paged_attention_reference
from b12x.integration.attention import (
    B12XPagedAttentionScratchCaps,
    paged_attention_forward,
    plan_paged_attention_scratch,
)

sys.path.insert(0, "/b12x/tests")
try:
    from paged_attention_helpers import make_paged_inputs
except ImportError:  # pytest package-relative
    from tests.paged_attention_helpers import make_paged_inputs

PAGE = 128
HD = 128


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


# ---------------------------------------------------------------- planner ----


def _tiny_plan_inputs(batch=1, pages=40, kv_heads=2, gqa=16):
    q, k, v, pt, cs, cu = make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=[pages * PAGE - 7] * batch,
        page_size=PAGE,
        q_heads=kv_heads * gqa,
        kv_heads=kv_heads,
        head_dim=HD,
        dtype=torch.bfloat16,
        seed=11,
    )
    return q, k, v, pt, cs, cu


def test_planner_dense_decode_split_plan():
    q, k, v, pt, cs, cu = _tiny_plan_inputs(batch=2, pages=40)
    plan = create_paged_plan(
        q, k, v, pt, cs, cu, mode="decode", force_split_kv=True, fixed_split_size=8
    )
    assert plan.split_kv, "forced dense decode plan must split"
    # 40 pages / 8 pages-per-chunk = 5 chunks per request
    assert tuple(plan.merge_indptr) == (0, 5, 10)
    assert tuple(plan.o_indptr) == (0, 5, 10)
    assert plan.total_num_partial_rows == 10
    assert len(plan.request_indices) == 10
    assert tuple(plan.kv_tile_indices) == tuple(range(5)) * 2
    assert all(t == 0 for t in plan.qo_tile_indices)


def test_planner_dense_decode_default_unsplit():
    q, k, v, pt, cs, cu = _tiny_plan_inputs(batch=1, pages=40)
    plan = create_paged_plan(q, k, v, pt, cs, cu, mode="decode")
    assert not plan.split_kv
    assert plan.total_num_partial_rows == 0
    assert tuple(plan.merge_indptr) == (0, 1)


def test_planner_dense_decode_split_rejects_unmergeable():
    q, k, v, pt, cs, cu = _tiny_plan_inputs(batch=1, pages=8)
    bad_q = torch.randn(
        q.shape[0], q.shape[1], 64, device=q.device, dtype=torch.bfloat16
    )
    bad_k = k[..., :64].contiguous()
    bad_v = v[..., :64].contiguous()
    try:
        create_paged_plan(
            bad_q, bad_k, bad_v, pt, cs, cu, mode="decode", force_split_kv=True
        )
    except ValueError as e:
        assert "merge-supported" in str(e)
    else:
        raise AssertionError("head_dim_vo=64 dense decode split must be rejected")


# ------------------------------------------------------------------ eager ----


def _eager_decode(q, k, v, pt, cs, cu, *, kv_quant="none", kd=None, vd=None,
                  force_split_kv=None, fixed_split_size=None):
    plan_probe = create_paged_plan(
        q, k, v, pt, cs, cu, mode="decode", kv_quant=kv_quant,
        force_split_kv=force_split_kv,
        fixed_split_size=-1 if fixed_split_size is None else int(fixed_split_size),
    )
    sp = plan_paged_attention_scratch(B12XPagedAttentionScratchCaps(
        device=q.device, mode="decode", dtype=q.dtype, kv_dtype=k.dtype,
        num_q_heads=q.shape[1], num_kv_heads=k.shape[2],
        head_dim_qk=q.shape[2], head_dim_vo=HD, page_size=PAGE,
        max_total_q=plan_probe.total_q, max_batch=pt.shape[0],
        max_page_table_width=pt.shape[1],
        max_work_items=max(plan_probe.padded_batch_size, 1),
        max_partial_rows=max(plan_probe.total_num_partial_rows, 0),
        num_cache_pages=k.shape[0], msa_block_sparse=False, kv_quant=kv_quant))
    scratch = tuple(torch.empty(s, dtype=d, device=q.device)
                    for s, d in sp.shapes_and_dtypes())
    out = torch.empty((q.shape[0], q.shape[1], HD), dtype=q.dtype, device=q.device)
    binding = sp.bind(
        scratch=scratch, q=q, k_cache=k, v_cache=v, output=out,
        page_table=pt, cache_seqlens=cs, cu_seqlens_q=cu, q2k_indices=None,
        k_descale=kd, v_descale=vd,
        force_split_kv=force_split_kv, fixed_split_size=fixed_split_size)
    o, lse2 = paged_attention_forward(binding=binding)
    return o.clone(), (lse2 * math.log(2.0)).clone()


def _run_eager_case(*, batch, cache_lens, kv_heads, gqa, kv, split_pages, seed):
    q, k, v, pt, cs, cu = make_paged_inputs(
        q_seqlens=[1] * batch, cache_seqlens=cache_lens, page_size=PAGE,
        q_heads=kv_heads * gqa, kv_heads=kv_heads, head_dim=HD,
        dtype=torch.bfloat16, seed=seed)
    kd = vd = None
    kv_quant = "none"
    if kv == "fp8":
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        kd = torch.ones((batch, kv_heads), dtype=torch.float32, device=q.device)
        vd = torch.ones((batch, kv_heads), dtype=torch.float32, device=q.device)
    elif kv == "nvfp4":
        from b12x.vllm.minimax_m3.backend import nvfp4_block_quant_write

        packed = HD // 2 + HD // 16
        blocks = int(k.shape[0])
        kq = torch.zeros(blocks, PAGE, kv_heads, packed, dtype=torch.uint8, device=q.device)
        vq = torch.zeros(blocks, PAGE, kv_heads, packed, dtype=torch.uint8, device=q.device)
        slots = torch.arange(blocks * PAGE, dtype=torch.int32, device=q.device)
        nvfp4_block_quant_write(k.reshape(blocks * PAGE, kv_heads * HD), kq, slots, kv_heads, HD)
        nvfp4_block_quant_write(v.reshape(blocks * PAGE, kv_heads * HD), vq, slots, kv_heads, HD)
        # identity page table required for the flat slot fill above
        pt = torch.arange(blocks, dtype=torch.int32, device=q.device).view(1, -1)
        pt = pt.expand(batch, -1).contiguous()
        k, v = kq, vq
        kv_quant = "nvfp4"

    base_o, base_lse = _eager_decode(q, k, v, pt, cs, cu, kv_quant=kv_quant, kd=kd, vd=vd)
    split_o, split_lse = _eager_decode(
        q, k, v, pt, cs, cu, kv_quant=kv_quant, kd=kd, vd=vd,
        force_split_kv=True, fixed_split_size=split_pages)
    c = _cos(split_o, base_o)
    dl = (split_lse - base_lse).abs().max().item()
    tag = f"batch={batch} kv={kv} lens={cache_lens} split={split_pages}p kvh={kv_heads}"
    assert c >= 0.99999 and dl < 2e-3, f"{tag}: cos={c:.7f} |dlse|={dl:.2e}"
    if kv == "bf16":
        ref_o, _ = paged_attention_reference(q, k, v, pt, cs, cu, causal=True)
        cr = _cos(split_o, ref_o)
        assert cr >= 0.9999, f"{tag}: split vs reference cos={cr:.7f}"
    print(f"  [eager {tag}] cos={c:.7f} |dlse|={dl:.2e} PASS", flush=True)


def test_eager_split_matches_unsplit_bf16():
    _run_eager_case(batch=1, cache_lens=[4321], kv_heads=2, gqa=16, kv="bf16",
                    split_pages=8, seed=5)
    _run_eager_case(batch=2, cache_lens=[3325, 5070], kv_heads=2, gqa=16,
                    kv="bf16", split_pages=8, seed=7)
    _run_eager_case(batch=1, cache_lens=[32768], kv_heads=2, gqa=16, kv="bf16",
                    split_pages=32, seed=9)
    _run_eager_case(batch=2, cache_lens=[2048, 630], kv_heads=4, gqa=16,
                    kv="bf16", split_pages=4, seed=13)


def test_eager_split_matches_unsplit_fp8():
    _run_eager_case(batch=1, cache_lens=[4321], kv_heads=2, gqa=16, kv="fp8",
                    split_pages=8, seed=5)
    _run_eager_case(batch=2, cache_lens=[3325, 5070], kv_heads=2, gqa=16,
                    kv="fp8", split_pages=8, seed=7)


def test_eager_split_matches_unsplit_nvfp4():
    try:
        import vllm  # noqa: F401
    except Exception:
        print("  [eager nvfp4] SKIP (vllm glue unavailable)", flush=True)
        return
    _run_eager_case(batch=1, cache_lens=[4096], kv_heads=2, gqa=16, kv="nvfp4",
                    split_pages=8, seed=5)


# ----------------------------------------------------------- graph replay ----


def _run_graph_case(*, batch, kv, width_pages, replay_lens, kv_heads=2, gqa=16, seed=21):
    dev = torch.device("cuda")
    max_len = width_pages * PAGE
    q, k, v, pt, cs, cu = make_paged_inputs(
        q_seqlens=[1] * batch, cache_seqlens=[max_len - 3] * batch,
        page_size=PAGE, q_heads=kv_heads * gqa, kv_heads=kv_heads, head_dim=HD,
        dtype=torch.bfloat16, seed=seed, page_table_width=width_pages)
    kd = vd = None
    kb, vb = k, v
    kv_quant = "none"
    if kv == "fp8":
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        kd = torch.ones((batch, kv_heads), dtype=torch.float32, device=dev)
        vd = torch.ones((batch, kv_heads), dtype=torch.float32, device=dev)
    elif kv == "nvfp4":
        from b12x.vllm.minimax_m3.backend import nvfp4_block_quant_write

        packed = HD // 2 + HD // 16
        blocks = int(k.shape[0])
        kq = torch.zeros(blocks, PAGE, kv_heads, packed, dtype=torch.uint8, device=dev)
        vq = torch.zeros(blocks, PAGE, kv_heads, packed, dtype=torch.uint8, device=dev)
        slots = torch.arange(blocks * PAGE, dtype=torch.int32, device=dev)
        nvfp4_block_quant_write(k.reshape(blocks * PAGE, kv_heads * HD), kq, slots, kv_heads, HD)
        nvfp4_block_quant_write(v.reshape(blocks * PAGE, kv_heads * HD), vq, slots, kv_heads, HD)
        pt = torch.arange(width_pages, dtype=torch.int32, device=dev).view(1, -1)
        pt = pt.expand(batch, -1).contiguous()
        # bf16 shadow caches must follow the identity page table for the ref
        kb, vb = k[:width_pages].contiguous(), v[:width_pages].contiguous()
        k, v = kq[:width_pages].contiguous(), vq[:width_pages].contiguous()
        kv_quant = "nvfp4"

    sp = plan_paged_attention_scratch(B12XPagedAttentionScratchCaps(
        device=dev, mode="decode", dtype=torch.bfloat16, kv_dtype=k.dtype, kv_quant=kv_quant,
        num_q_heads=kv_heads * gqa, num_kv_heads=kv_heads,
        head_dim_qk=HD, head_dim_vo=HD, page_size=PAGE,
        max_total_q=batch, max_batch=batch, max_page_table_width=width_pages,
        max_work_items=batch * 512, max_partial_rows=batch * 512,
        num_cache_pages=k.shape[0], use_cuda_graph=True,
        msa_block_sparse=False))
    sp.prepare_decode_graph_replay_state(
        batch=batch, max_page_table_width=width_pages,
        max_cache_page_count=width_pages, force_split_kv=True)
    assert sp._plan is not None and sp._plan.split_kv, "capacity plan must split"
    scratch = tuple(torch.zeros(s, dtype=d, device=dev)
                    for s, d in sp.shapes_and_dtypes())
    out = torch.empty((batch, kv_heads * gqa, HD), dtype=torch.bfloat16, device=dev)

    def step():
        # Production contract: bind EVERY step (pure views); during capture the
        # LUT metadata update is captured in-graph and re-chunks from the
        # runtime cache_seqlens on every replay.
        binding = sp.bind(
            scratch=scratch, q=q, k_cache=k, v_cache=v, output=out,
            page_table=pt, cache_seqlens=cs, cu_seqlens_q=cu, q2k_indices=None,
            k_descale=kd, v_descale=vd)
        paged_attention_forward(binding=binding)

    for _ in range(2):  # warmup: JIT everything outside capture
        step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()

    for L in replay_lens:
        cs.fill_(int(L))
        g.replay()
        torch.cuda.synchronize()
        got = out.clone()
        # Reference = UNSPLIT eager decode on the SAME (quantized) caches:
        # isolates split/graph error from quantization error.
        ref_o, _ = _eager_decode(
            q, k, v, pt, torch.full_like(cs, int(L)), cu,
            kv_quant=kv_quant, kd=kd, vd=vd)
        c = _cos(got, ref_o)
        assert c >= 0.99999, f"graph batch={batch} kv={kv} L={L}: cos={c:.7f}"
        # bf16 additionally vs the true reference kernel
        if kv == "bf16":
            tref, _ = paged_attention_reference(
                q, kb, vb, pt, torch.full_like(cs, int(L)), cu, causal=True)
            cr = _cos(got, tref)
            assert cr >= 0.9999, f"graph bf16 L={L} vs reference: cos={cr:.7f}"
        print(f"  [graph batch={batch} kv={kv} L={L}] cos={c:.7f} PASS", flush=True)


def test_graph_replay_split_growing_cache_bf16():
    _run_graph_case(batch=1, kv="bf16", width_pages=256,
                    replay_lens=[1000, 8192, 20480, 32765])
    _run_graph_case(batch=3, kv="bf16", width_pages=64,
                    replay_lens=[700, 4096, 8189])


def test_graph_replay_split_growing_cache_fp8():
    _run_graph_case(batch=1, kv="fp8", width_pages=256,
                    replay_lens=[1000, 8192, 32765])


def test_graph_replay_split_growing_cache_nvfp4():
    try:
        import vllm  # noqa: F401
    except Exception:
        print("  [graph nvfp4] SKIP (vllm glue unavailable)", flush=True)
        return
    _run_graph_case(batch=1, kv="nvfp4", width_pages=256,
                    replay_lens=[1000, 8192, 32765])


if __name__ == "__main__":
    torch.manual_seed(0)
    for fn in (
        test_planner_dense_decode_split_plan,
        test_planner_dense_decode_default_unsplit,
        test_planner_dense_decode_split_rejects_unmergeable,
        test_eager_split_matches_unsplit_bf16,
        test_eager_split_matches_unsplit_fp8,
        test_eager_split_matches_unsplit_nvfp4,
        test_graph_replay_split_growing_cache_bf16,
        test_graph_replay_split_growing_cache_fp8,
        test_graph_replay_split_growing_cache_nvfp4,
    ):
        print(f"== {fn.__name__}", flush=True)
        fn()
    print("== ALL PASS ==", flush=True)
