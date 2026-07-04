"""Per-component decode microbench: dense / MSA / indexer x bf16 / fp8 / nvfp4.

MiniMax-M3 serving geometry per TP2 rank (GB10):
  main attention: 32 q heads / 2 kv heads / head_dim 128 / page 128 (gqa 16)
  indexer: 2 index heads / head_dim 128 / topk 16 blocks of 128 / page 64
  page-table capacity fixed at max_model_len 65536 (width128=512, width64=1024)
  batch=1 single-token decode, CUDA-graph captured (FULL_DECODE_ONLY parity)

Run inside the b12x container with the fork mounted at /b12x:
  PYTHONPATH=/b12x python3 microbench_matrix.py --contexts 8192,32768,60000
"""
import argparse, sys, time
import torch

sys.path.insert(0, "/b12x")

from b12x.integration.attention import (
    B12XPagedAttentionScratchCaps, paged_attention_forward,
    plan_paged_attention_scratch)
from b12x.attention.indexer import (
    MSA_TOPK_BLOCKS, build_paged_mqa_schedule_metadata,
    msa_decode_query_positions, msa_paged_decode_block_scores,
    quantize_msa_q_fp8)
from b12x.attention.indexer.scratch import (
    B12XIndexerPagedScratchCaps, plan_indexer_paged_scratch)
# vLLM glue helpers (Triton cache writers + fused select) -- import inside the
# serving container where vllm is installed.
from b12x.vllm.minimax_m3.indexer import (
    write_packed_index_cache, write_packed_index_cache_nvfp4,
    _triton_msa_select, _triton_msa_qquant, _next_pow2,
    _PACK_ROW_BYTES, _PACK_ROW_BYTES_NV)
from b12x.vllm.minimax_m3.backend import nvfp4_block_quant_write

DEV = torch.device("cuda")
Q_HEADS, KV_HEADS, HD, PAGE = 32, 2, 128, 128
IDX_HEADS, TOPK = 2, 16
WIDTH128, WIDTH64 = 512, 1024          # 65536-token capacity (serving shapes)
FP8 = torch.float8_e4m3fn

_flush_buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)


def _capture(fn, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def _time_graph(g, iters=20):
    us = []
    for _ in range(iters):
        _flush_buf.zero_()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record()
        torch.cuda.synchronize()
        us.append(s.elapsed_time(e) * 1000.0)
    us.sort()
    return sum(us) / len(us), us[len(us)//2]


def _main_inputs(ctx, kv_mode):
    """bf16 base tensors + cache in the requested storage mode."""
    torch.manual_seed(7)
    blocks = WIDTH128
    q = torch.randn(1, Q_HEADS, HD, dtype=torch.bfloat16, device=DEV) / 3
    kb = torch.randn(blocks, PAGE, KV_HEADS, HD, dtype=torch.bfloat16, device=DEV) / 3
    vb = torch.randn(blocks, PAGE, KV_HEADS, HD, dtype=torch.bfloat16, device=DEV) / 3
    pt = torch.arange(WIDTH128, dtype=torch.int32, device=DEV).view(1, -1)
    cs = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    cu = torch.arange(2, dtype=torch.int32, device=DEV)
    kd = vd = None
    if kv_mode == "bf16":
        kc, vc = kb, vb
    elif kv_mode == "fp8":
        kc, vc = kb.to(FP8), vb.to(FP8)
        kd = torch.ones((1, KV_HEADS), dtype=torch.float32, device=DEV)
        vd = torch.ones((1, KV_HEADS), dtype=torch.float32, device=DEV)
    else:  # nvfp4: packed uint8 [blocks,128,kv,72] filled by the fused writer
        packed = HD // 2 + HD // 16
        kc = torch.zeros(blocks, PAGE, KV_HEADS, packed, dtype=torch.uint8, device=DEV)
        vc = torch.zeros(blocks, PAGE, KV_HEADS, packed, dtype=torch.uint8, device=DEV)
        T = blocks * PAGE
        slots = torch.arange(T, dtype=torch.int32, device=DEV)
        nvfp4_block_quant_write(kb.reshape(T, KV_HEADS * HD), kc, slots, KV_HEADS, HD)
        nvfp4_block_quant_write(vb.reshape(T, KV_HEADS * HD), vc, slots, KV_HEADS, HD)
    return q, kc, vc, pt, cs, cu, kd, vd


def _q2k(ctx):
    nb = (ctx + 127) // 128
    idx = torch.full((KV_HEADS, 1, TOPK), -1, dtype=torch.int32, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(11)
    for h in range(KV_HEADS):
        cand = torch.randperm(nb - 1, generator=g)[: TOPK - 1].tolist()
        sel = sorted(set(cand[: TOPK - 1] + [nb - 1]))[:TOPK]
        idx[h, 0, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device=DEV)
    return idx


def bench_main(ctx, kv_mode, sparse, split=False):
    q, kc, vc, pt, cs, cu, kd, vd = _main_inputs(ctx, kv_mode)
    kv_quant = "nvfp4" if kv_mode == "nvfp4" else "none"
    sp = plan_paged_attention_scratch(B12XPagedAttentionScratchCaps(
        device=DEV, mode="decode", dtype=torch.bfloat16, kv_dtype=kc.dtype,
        num_q_heads=Q_HEADS, num_kv_heads=KV_HEADS, head_dim_qk=HD, head_dim_vo=HD,
        page_size=PAGE, max_total_q=1, max_batch=1, max_page_table_width=WIDTH128,
        max_work_items=512, max_partial_rows=512, num_cache_pages=kc.shape[0],
        use_cuda_graph=True, msa_block_sparse=sparse, kv_quant=kv_quant))
    sp.prepare_decode_graph_replay_state(
        batch=1, max_page_table_width=WIDTH128,
        max_cache_page_count=min(int(kc.shape[0]), WIDTH128),
        force_split_kv=bool(split))
    scratch = tuple(torch.zeros(s, dtype=d, device=DEV) for s, d in sp.shapes_and_dtypes())
    out = torch.empty(1, Q_HEADS, HD, dtype=torch.bfloat16, device=DEV)
    q2k = _q2k(ctx) if sparse else None
    if split:
        # Production-faithful: bind EVERY step so the captured graph includes
        # the per-step LUT metadata update (re-chunks from runtime seqlens).
        def step():
            binding = sp.bind(scratch=scratch, q=q, k_cache=kc, v_cache=vc,
                              output=out, page_table=pt, cache_seqlens=cs,
                              cu_seqlens_q=cu, q2k_indices=q2k,
                              k_descale=kd, v_descale=vd)
            paged_attention_forward(binding=binding)
        g = _capture(step)
    else:
        binding = sp.bind(scratch=scratch, q=q, k_cache=kc, v_cache=vc, output=out,
                          page_table=pt, cache_seqlens=cs, cu_seqlens_q=cu,
                          q2k_indices=q2k, k_descale=kd, v_descale=vd)
        g = _capture(lambda: paged_attention_forward(binding=binding))
    return _time_graph(g)


def bench_indexer(ctx, idx_mode):
    torch.manual_seed(13)
    ik = torch.randn(ctx, HD, dtype=torch.bfloat16, device=DEV) / 3
    slots = torch.arange(ctx, dtype=torch.int32, device=DEV)
    if idx_mode == "fp8":
        cache = torch.zeros(WIDTH64, _PACK_ROW_BYTES, dtype=torch.uint8, device=DEV)
        write_packed_index_cache(cache, ik, slots)
        kv_quant = "none"
    else:
        cache = torch.zeros(WIDTH64, _PACK_ROW_BYTES_NV, dtype=torch.uint8, device=DEV)
        write_packed_index_cache_nvfp4(cache, ik, slots)
        kv_quant = "nvfp4"
    # identity page64 table at full serving width
    page64 = torch.arange(WIDTH64, dtype=torch.int32, device=DEV).view(1, -1)
    seqlens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    sched = build_paged_mqa_schedule_metadata(seqlens, 64)
    qpos = msa_decode_query_positions(seqlens)
    iq = torch.randn(1, IDX_HEADS, HD, dtype=torch.bfloat16, device=DEV) / 3
    caps = B12XIndexerPagedScratchCaps(
        device=DEV, num_q_heads=IDX_HEADS, num_idx_heads=IDX_HEADS, max_q_rows=1,
        max_page_table_width=WIDTH64, topk=MSA_TOPK_BLOCKS, page_size=64,
        score_mode="msa", kv_quant=kv_quant)
    plan = plan_indexer_paged_scratch(caps)
    (spec,) = plan.scratch_specs()
    scratch = torch.zeros(spec.shape, dtype=spec.dtype, device=DEV)
    q2k_out = torch.full((IDX_HEADS, 1, TOPK), -1, dtype=torch.int32, device=DEV)
    sel_block = _next_pow2(WIDTH128)

    def step():
        q_fp8, q_scale = _triton_msa_qquant(iq)
        binding = plan.bind_msa(scratch=scratch, real_page_table=page64,
                                cache_seqlens_int32=seqlens,
                                schedule_metadata=sched, topk=TOPK)
        bs = msa_paged_decode_block_scores(
            q_fp8=q_fp8, q_scale=q_scale, index_k_cache=cache, binding=binding)
        _triton_msa_select(bs, qpos, TOPK, out=q2k_out, block=sel_block)

    g = _capture(step)
    return _time_graph(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="8192,32768,60000")
    ap.add_argument("--out", default="/tmp/microbench_matrix.tsv")
    args = ap.parse_args()
    ctxs = [int(x) for x in args.contexts.split(",")]

    rows = []
    def rec(comp, mode, ctx, mean_us, med_us, layers):
        rows.append((comp, mode, ctx, mean_us, med_us, layers))
        print(f"{comp:14s} {mode:6s} ctx={ctx:>6d}  {mean_us:8.1f} us/layer "
              f"(med {med_us:.1f})  -> {mean_us*layers/1000:6.2f} ms/tok x{layers}", flush=True)

    for ctx in ctxs:
        for mode in ("bf16", "fp8", "nvfp4"):
            m, md = bench_main(ctx, mode, sparse=False)
            rec("dense-decode", mode, ctx, m, md, 3)
        for mode in ("bf16", "fp8", "nvfp4"):
            m, md = bench_main(ctx, mode, sparse=False, split=True)
            rec("dense-split", mode, ctx, m, md, 3)
        for mode in ("bf16", "fp8", "nvfp4"):
            m, md = bench_main(ctx, mode, sparse=True)
            rec("msa-decode", mode, ctx, m, md, 57)
        for mode in ("fp8", "nvfp4"):
            m, md = bench_indexer(ctx, mode)
            rec("indexer-q2k", mode, ctx, m, md, 57)

    with open(args.out, "w") as f:
        f.write("component\tmode\tctx\tmean_us_per_layer\tmedian_us\tlayers\tms_per_token\n")
        for comp, mode, ctx, m, md, L in rows:
            f.write(f"{comp}\t{mode}\t{ctx}\t{m:.1f}\t{md:.1f}\t{L}\t{m*L/1000:.3f}\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
