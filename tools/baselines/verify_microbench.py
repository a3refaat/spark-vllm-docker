"""Spec-decode VERIFY-shape microbench (q_len=4 = spec_tokens 3 + bonus).

Times the b12x per-layer attention/indexer paths at the verify shape against
their single-row decode baselines, per KV mode, graph-replayed with L2 flush:

  dense-decode-q1      mode=decode  batch=1        (prod non-spec, split)
  dense-vfy-multirow   mode=decode  batch=4 rows   (PROD verify today:
                                                    backend._verify -> 4 rows
                                                    re-scan full KV each)
  dense-vfy-native     mode=verify  b=1 q_len=4    (Phase-8 candidate: shared
                                                    KV stream)
  msa-decode-q1        block-sparse decode         (prod non-spec)
  msa-vfy-native       mode=verify block-sparse    (PROD verify today)
  msa-vfy-multirow     4-row decode fallback       (B12X_MSA_NATIVE_VERIFY=0)
  idx-decode-q1        fused indexer, 1 row        (prod non-spec)
  idx-vfy-multirow     fused indexer, 4 rows       (PROD fp8 verify; nvfp4
                                                    fallback when
                                                    B12X_INDEXER_NATIVE_NVF4=0)

Run inside the b12x container (server DOWN), fork mounted at /b12x:
  PYTHONPATH=/b12x:/b12x/tests python3 verify_microbench.py --contexts 32768,60000
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, "/b12x")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microbench_matrix import (  # noqa: E402
    DEV, Q_HEADS, KV_HEADS, HD, PAGE, IDX_HEADS, TOPK, WIDTH128, WIDTH64,
    _capture, _time_graph, _main_inputs,
    _PACK_ROW_BYTES, _PACK_ROW_BYTES_NV,
)
from b12x.integration.attention import (  # noqa: E402
    B12XPagedAttentionScratchCaps, paged_attention_forward,
    plan_paged_attention_scratch)
from b12x.attention.indexer import MSA_SM_SCALE  # noqa: E402
from b12x.attention.indexer.fused_msa_indexer import (  # noqa: E402
    msa_fused_scratch_shapes, run_fused_msa_indexer)
from b12x.attention.indexer.kernel import (  # noqa: E402
    _split_index_k_cache_runtime_views)
from b12x.vllm.minimax_m3.indexer import (  # noqa: E402
    _triton_msa_qquant, write_packed_index_cache, write_packed_index_cache_nvfp4)

Q_LEN = 4  # spec_tokens=3 + bonus token
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


def _q2k_rows(ctx, q_rows):
    nb = (ctx + 127) // 128
    idx = torch.full((KV_HEADS, q_rows, TOPK), -1, dtype=torch.int32, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(11)
    for h in range(KV_HEADS):
        for r in range(q_rows):
            cand = torch.randperm(nb - 1, generator=g)[: TOPK - 1].tolist()
            sel = sorted(set(cand + [nb - 1]))[:TOPK]
            idx[h, r, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device=DEV)
    return idx


def bench_attn(ctx, kv_mode, sparse, shape, return_output=False, fixed_split=-1):
    """shape: 'decode-q1' | 'vfy-multirow' | 'vfy-native'."""
    q1, kc, vc, pt1, _, _, kd1, vd1 = _main_inputs(ctx, kv_mode)
    kv_quant = "nvfp4" if kv_mode == "nvfp4" else "none"

    if shape == "decode-q1":
        nd, batch, q_len = 1, 1, 1
        q, pt = q1, pt1
        cs = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        cu = torch.arange(2, dtype=torch.int32, device=DEV)
    elif shape == "vfy-multirow":
        nd, batch, q_len = Q_LEN, Q_LEN, 1
        q = q1.repeat(Q_LEN, 1, 1).contiguous()
        pt = pt1.repeat(Q_LEN, 1).contiguous()
        cs = torch.tensor([ctx - Q_LEN + j + 1 for j in range(Q_LEN)],
                          dtype=torch.int32, device=DEV)
        cu = torch.arange(Q_LEN + 1, dtype=torch.int32, device=DEV)
    else:  # vfy-native
        nd, batch, q_len = Q_LEN, 1, Q_LEN
        q = q1.repeat(Q_LEN, 1, 1).contiguous()
        pt = pt1
        cs = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        cu = torch.tensor([0, Q_LEN], dtype=torch.int32, device=DEV)

    kd = vd = None
    if kv_mode == "fp8":
        kd = torch.ones((nd if shape != "vfy-native" else batch, KV_HEADS),
                        dtype=torch.float32, device=DEV)
        vd = kd.clone()

    mode = "verify" if shape == "vfy-native" else "decode"
    sp = plan_paged_attention_scratch(B12XPagedAttentionScratchCaps(
        device=DEV, mode=mode, dtype=torch.bfloat16, kv_dtype=kc.dtype,
        num_q_heads=Q_HEADS, num_kv_heads=KV_HEADS, head_dim_qk=HD,
        head_dim_vo=HD, page_size=PAGE, max_total_q=nd, max_batch=batch,
        max_page_table_width=WIDTH128, max_work_items=nd * 64,
        max_partial_rows=nd * 64, num_cache_pages=kc.shape[0],
        use_cuda_graph=True, msa_block_sparse=sparse, kv_quant=kv_quant))
    if mode == "verify":
        if sparse:
            sp.prepare_verify_graph_replay_state(
                batch=batch, q_len=q_len, max_page_table_width=WIDTH128,
                max_cache_page_count=min(int(kc.shape[0]), WIDTH128),
                fixed_split_size=fixed_split)
        else:
            # full-attention (dense) native verify: grouped qo-tiles sharing
            # the request KV split, per-replay adaptive chunk metadata
            sp.prepare_prefill_graph_replay_state(
                batch=batch, total_q_capacity=nd,
                max_page_table_width=WIDTH128, cu_seqlens_q=cu,
                max_cache_page_count=min(int(kc.shape[0]), WIDTH128))
    else:
        sp.prepare_decode_graph_replay_state(
            batch=nd, max_page_table_width=WIDTH128,
            max_cache_page_count=min(int(kc.shape[0]), WIDTH128),
            force_split_kv=(not sparse))  # dense prod default: split ON
    scratch = tuple(torch.zeros(s, dtype=d, device=DEV)
                    for s, d in sp.shapes_and_dtypes())
    out = torch.empty(nd, Q_HEADS, HD, dtype=torch.bfloat16, device=DEV)
    q2k = _q2k_rows(ctx, nd) if sparse else None

    def step():  # rebind per step, matching production paths
        binding = sp.bind(scratch=scratch, q=q, k_cache=kc, v_cache=vc,
                          output=out, page_table=pt, cache_seqlens=cs,
                          cu_seqlens_q=cu, q2k_indices=q2k,
                          k_descale=kd, v_descale=vd)
        paged_attention_forward(binding=binding)

    if return_output:
        step()
        torch.cuda.synchronize()
        return out.clone()
    return _time_graph(_capture(step))


def bench_indexer_fused(ctx, idx_mode, q_rows, q_len=1):
    torch.manual_seed(13)
    ik = torch.randn(ctx, HD, dtype=torch.bfloat16, device=DEV) / 3
    slots = torch.arange(ctx, dtype=torch.int32, device=DEV)
    if idx_mode == "fp8":
        cache = torch.zeros(WIDTH64, _PACK_ROW_BYTES, dtype=torch.uint8, device=DEV)
        write_packed_index_cache(cache, ik, slots)
    else:
        cache = torch.zeros(WIDTH64, _PACK_ROW_BYTES_NV, dtype=torch.uint8, device=DEV)
        write_packed_index_cache_nvfp4(cache, ik, slots)
    kq, ks = _split_index_k_cache_runtime_views(cache)
    groups = q_rows // q_len
    page64 = torch.arange(WIDTH64, dtype=torch.int32, device=DEV) \
        .view(1, -1).repeat(groups, 1).contiguous()
    if q_rows == 1:
        seqlens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    else:
        seqlens = torch.tensor([ctx - q_rows + j + 1 for j in range(q_rows)],
                               dtype=torch.int32, device=DEV)
    iq = torch.randn(q_rows, IDX_HEADS, HD, dtype=torch.bfloat16, device=DEV) / 3
    slab_shape, state_shape = msa_fused_scratch_shapes(q_rows, IDX_HEADS, WIDTH64)
    slab = torch.empty(slab_shape, dtype=torch.float32, device=DEV)
    state = torch.zeros(state_shape, dtype=torch.int32, device=DEV)
    q2k_out = torch.full((IDX_HEADS, q_rows, TOPK), -1, dtype=torch.int32, device=DEV)
    weights = torch.full((q_rows, IDX_HEADS), MSA_SM_SCALE,
                         dtype=torch.float32, device=DEV)
    ctas = max(1, min((WIDTH64 + 1) // 2, (2 * NUM_SMS) // max(1, groups)))
    is_nv = idx_mode == "nvfp4"

    def step():
        if is_nv:
            q_in = iq.view(torch.uint8)
        else:
            q_fp8, q_scale = _triton_msa_qquant(iq)
            torch.mul(q_scale, MSA_SM_SCALE, out=weights)
            q_in = q_fp8.view(torch.uint8)
        run_fused_msa_indexer(
            q_bytes=q_in, weights=weights, k_quant_bytes=kq, k_scales=ks,
            real_page_table=page64, seqlens=seqlens, num_heads=IDX_HEADS,
            topk=TOPK, out_indices=q2k_out, ctas_per_group=ctas,
            slab=slab, state=state, state_preinitialized=True,
            kv_quant="nvfp4" if is_nv else "none", q_len=q_len)

    return _time_graph(_capture(step))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="32768,60000")
    ap.add_argument("--modes", default="fp8,nvfp4")
    ap.add_argument("--out", default="/tmp/verify_microbench.tsv")
    ap.add_argument("--parity", action="store_true",
                    help="compare dense vfy-native output vs vfy-multirow")
    args = ap.parse_args()

    if args.parity:
        for ctx in [int(x) for x in args.contexts.split(",")]:
            for mode in args.modes.split(","):
                a = bench_attn(ctx, mode, False, "vfy-multirow", return_output=True)
                b = bench_attn(ctx, mode, False, "vfy-native", return_output=True)
                cos = torch.nn.functional.cosine_similarity(
                    a.float().flatten(1), b.float().flatten(1), dim=1)
                dmax = (a.float() - b.float()).abs().max().item()
                ok = "PASS" if (cos.min().item() > 0.99999 and dmax < 0.02) else "FAIL"
                print(f"dense-verify parity {mode} ctx={ctx}: {ok} "
                      f"mincos={cos.min().item():.6f} maxdiff={dmax:.5f}", flush=True)
        return
    ctxs = [int(x) for x in args.contexts.split(",")]
    modes = args.modes.split(",")

    rows = []

    def rec(comp, mode, ctx, res, layers):
        if res is None:
            print(f"{comp:20s} {mode:6s} ctx={ctx:>6d}  FAILED", flush=True)
            rows.append((comp, mode, ctx, float("nan"), float("nan"), layers))
            return
        m, md = res
        rows.append((comp, mode, ctx, m, md, layers))
        print(f"{comp:20s} {mode:6s} ctx={ctx:>6d}  {m:8.1f} us/layer "
              f"(med {md:.1f})  -> {m*layers/1000:6.2f} ms/step x{layers}",
              flush=True)

    def safe(fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as ex:
            print(f"    [err] {type(ex).__name__}: {str(ex)[:140]}", flush=True)
            return None

    for ctx in ctxs:
        for mode in modes:
            rec("dense-decode-q1", mode, ctx, safe(bench_attn, ctx, mode, False, "decode-q1"), 3)
            rec("dense-vfy-multirow", mode, ctx, safe(bench_attn, ctx, mode, False, "vfy-multirow"), 3)
            rec("dense-vfy-native", mode, ctx, safe(bench_attn, ctx, mode, False, "vfy-native"), 3)
            rec("msa-decode-q1", mode, ctx, safe(bench_attn, ctx, mode, True, "decode-q1"), 57)
            rec("msa-vfy-native", mode, ctx, safe(bench_attn, ctx, mode, True, "vfy-native"), 57)
            rec("msa-vfy-multirow", mode, ctx, safe(bench_attn, ctx, mode, True, "vfy-multirow"), 57)
            rec("idx-decode-q1", mode, ctx, safe(bench_indexer_fused, ctx, mode, 1), 57)
            rec("idx-vfy-multirow", mode, ctx, safe(bench_indexer_fused, ctx, mode, Q_LEN), 57)
            rec("idx-vfy-grouped", mode, ctx,
                safe(bench_indexer_fused, ctx, mode, Q_LEN, Q_LEN), 57)

    with open(args.out, "w") as f:
        f.write("component\tmode\tctx\tmean_us_per_layer\tmedian_us\tlayers\tms_per_step\n")
        for comp, mode, ctx, m, md, L in rows:
            f.write(f"{comp}\t{mode}\t{ctx}\t{m:.1f}\t{md:.1f}\t{L}\t{m*L/1000:.3f}\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
