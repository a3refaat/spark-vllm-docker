"""Time the PRODUCTION nvfp4 native verify indexer chain (_verify_nvf4_native)
as a faithful replica: same kernels, same serving capacities (qcap=4,
max_reqs=1, supertile_k=32768, max_chunks=2, width128=512), graph-replayed.

PYTHONPATH=/b12x:/b12x/tests python3 idx_native_verify_bench.py --contexts 32768,60000
"""
import argparse
import sys

import torch

sys.path.insert(0, "/b12x")

from b12x.vllm.minimax_m3.indexer import (  # noqa: E402
    _INDEX_HEAD_DIM, _IDX_SUPERTILE_K, _MSA_BLOCK_TOKENS, _NV_NBLK,
    _NVFP4_BLK, _NVFP4_E2M1_MAX, _PACK_DATA_BYTES_NV, _PACK_ROW_BYTES_NV,
    _msa_qquant_nvfp4_kernel, _next_pow2, _nvf4_topk_finalize_kernel,
    _nvf4_topk_init_kernel, _nvf4_block_topk_merge_kernel,
    _nvf4_verify_base_kernel, _nvf4_verify_gather_chunk_kernel,
    _nvf4_verify_local_bounds_kernel, _nvf4_verify_query_meta_kernel,
    _nvf4_verify_buffers, write_packed_index_cache_nvfp4)
from b12x.attention.indexer.contiguous_kernel import (  # noqa: E402
    run_contiguous_block_scores_kernel_nvf4)

DEV = torch.device("cuda")
HEADS, TOPK, Q_LEN = 2, 16, 4
WIDTH128, WIDTH64 = 512, 1024
MAX_REQS, QCAP = 1, 4
SUPERTILE = ((_IDX_SUPERTILE_K + 127) // 128) * 128
MAX_CHUNKS = max(1, (WIDTH128 * _MSA_BLOCK_TOKENS + SUPERTILE - 1) // SUPERTILE)

_flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)


def bench(ctx):
    torch.manual_seed(13)
    ik = torch.randn(ctx, _INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEV) / 3
    slots = torch.arange(ctx, dtype=torch.int32, device=DEV)
    cache = torch.zeros(WIDTH64, _PACK_ROW_BYTES_NV, dtype=torch.uint8, device=DEV)
    write_packed_index_cache_nvfp4(cache, ik, slots)
    npages = int(cache.shape[0])

    seq_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    block_table = torch.arange(WIDTH128, dtype=torch.int32, device=DEV).view(1, -1)
    iq_full = torch.randn(QCAP, HEADS, _INDEX_HEAD_DIM,
                          dtype=torch.bfloat16, device=DEV) / 3
    b = _nvf4_verify_buffers(QCAP, SUPERTILE, HEADS, TOPK, MAX_REQS, WIDTH128, DEV)
    chunk_blocks = int(b["chunk_blocks"])
    block_pow2 = _next_pow2(chunk_blocks)
    block_req = _next_pow2(MAX_REQS)
    req_log = max(1, (MAX_REQS + 1).bit_length())
    q_block = 128
    q_grid = ((QCAP + q_block - 1) // q_block,)

    def step():
        b["seq"][:1].copy_(seq_lens)
        b["block_table"][:1, :WIDTH128].copy_(block_table)
        _nvf4_verify_base_kernel[(1,)](
            b["seq"], b["base"], b["cum"], 1, block_req, num_warps=1)
        _nvf4_verify_query_meta_kernel[q_grid](
            b["seq"], b["base"], b["k_start_g"], b["k_end_g"], b["block_base"],
            b["qpos"], QCAP, Q_LEN, QCAP, q_block, num_warps=4)
        _msa_qquant_nvfp4_kernel[(QCAP * HEADS,)](
            iq_full, b["q_e2m1"], b["q_sfa"], _INDEX_HEAD_DIM, _NV_NBLK,
            _NVFP4_BLK, _NVFP4_E2M1_MAX, num_warps=4)
        _nvf4_topk_init_kernel[(HEADS * QCAP,)](
            b["carry_v"][0], b["carry_i"][0], QCAP, QCAP, TOPK, num_warps=1)
        for chunk_idx in range(MAX_CHUNKS):
            chunk_start = chunk_idx * SUPERTILE
            _nvf4_verify_gather_chunk_kernel[(SUPERTILE,)](
                cache.reshape(-1), b["block_table"], b["seq"], b["base"], b["cum"],
                b["k_e2m1"], b["k_sfb"], chunk_start, 1, npages, WIDTH128,
                int(b["block_table"].stride(0)), int(b["block_table"].stride(1)),
                req_log, _PACK_ROW_BYTES_NV, _INDEX_HEAD_DIM // 2, _NV_NBLK,
                _PACK_DATA_BYTES_NV, num_warps=2)
            _nvf4_verify_local_bounds_kernel[q_grid](
                b["k_start_g"], b["k_end_g"], b["k_start"], b["k_end"],
                QCAP, chunk_start, SUPERTILE, QCAP, q_block, num_warps=4)
            run_contiguous_block_scores_kernel_nvf4(
                b["q_e2m1"], b["q_sfa"], b["weights"], b["k_e2m1"], b["k_sfb"],
                b["k_start"], b["k_end"], valid_q_rows=QCAP,
                valid_k_rows=SUPERTILE, num_blocks_out=chunk_blocks,
                block_scores=b["bs"])
            in_slot = chunk_idx & 1
            _nvf4_block_topk_merge_kernel[(HEADS * QCAP,)](
                b["bs"], b["carry_v"][in_slot], b["carry_i"][in_slot],
                b["carry_v"][1 - in_slot], b["carry_i"][1 - in_slot], b["qpos"],
                QCAP, QCAP, chunk_blocks, chunk_start // _MSA_BLOCK_TOKENS,
                TOPK, block_pow2, _MSA_BLOCK_TOKENS, num_warps=4)
        _nvf4_topk_finalize_kernel[(HEADS * QCAP,)](
            b["carry_v"][MAX_CHUNKS & 1], b["carry_i"][MAX_CHUNKS & 1],
            b["block_base"], b["q2k"], QCAP, QCAP, TOPK, num_warps=1)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    us = []
    for _ in range(20):
        _flush.zero_()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); g.replay(); e.record()
        torch.cuda.synchronize()
        us.append(s.elapsed_time(e) * 1000.0)
    us.sort()
    return sum(us) / len(us), us[len(us) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="32768,60000")
    args = ap.parse_args()
    print(f"supertile={SUPERTILE} max_chunks={MAX_CHUNKS}")
    for ctx in [int(x) for x in args.contexts.split(",")]:
        m, md = bench(ctx)
        print(f"idx-vfy-NATIVE-nvf4  ctx={ctx:>6d}  {m:8.1f} us/layer "
              f"(med {md:.1f})  -> {m*57/1000:6.2f} ms/step x57", flush=True)


if __name__ == "__main__":
    main()
