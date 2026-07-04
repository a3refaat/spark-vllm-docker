"""Phase 4 meta-once: follower layers skip replay-metadata copy/update.

Two plans with IDENTICAL caps share one scratch tuple (the production layer
family pattern). The owner binds with metadata updates; the follower binds
with skip_replay_metadata_update=True. The follower's forward must match a
fully-updating bind bit-for-bit, eagerly and under graph replay with changing
cache_seqlens. Runs standalone: ``python3 tests/test_attn_meta_once.py``.
"""

from __future__ import annotations

import sys

import torch

from b12x.integration.attention import (
    B12XPagedAttentionScratchCaps,
    paged_attention_forward,
    plan_paged_attention_scratch,
)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from paged_attention_helpers import make_paged_inputs  # noqa: E402

PAGE, HD, KV, GQA = 128, 128, 2, 16


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def _mk_plan(dev, batch, width, ncb, split):
    caps = B12XPagedAttentionScratchCaps(
        device=dev, mode="decode", dtype=torch.bfloat16, kv_dtype=torch.bfloat16,
        num_q_heads=KV * GQA, num_kv_heads=KV, head_dim_qk=HD, head_dim_vo=HD,
        page_size=PAGE, max_total_q=batch, max_batch=batch,
        max_page_table_width=width, max_work_items=batch * 512,
        max_partial_rows=batch * 512, num_cache_pages=ncb, use_cuda_graph=True,
        msa_block_sparse=False)
    plan = plan_paged_attention_scratch(caps)
    plan.prepare_decode_graph_replay_state(
        batch=batch, max_page_table_width=width, max_cache_page_count=width,
        force_split_kv=split)
    return plan


def test_meta_once_dense(split=True, batch=2, width=64):
    dev = torch.device("cuda")
    q, k, v, pt, cs, cu = make_paged_inputs(
        q_seqlens=[1] * batch, cache_seqlens=[width * PAGE - 5] * batch,
        page_size=PAGE, q_heads=KV * GQA, kv_heads=KV, head_dim=HD,
        dtype=torch.bfloat16, seed=31, page_table_width=width)
    # layer family: owner + follower plans, ONE shared scratch tuple
    owner = _mk_plan(dev, batch, width, k.shape[0], split)
    follower = _mk_plan(dev, batch, width, k.shape[0], split)
    scratch = tuple(torch.zeros(s, dtype=d, device=dev)
                    for s, d in owner.shapes_and_dtypes())
    out_own = torch.zeros((batch, KV * GQA, HD), dtype=torch.bfloat16, device=dev)
    out_fol = torch.zeros_like(out_own)
    q2 = (q * 0.7 + 0.1).contiguous()  # follower layer gets different q

    def step(skip_follower):
        b = owner.bind(scratch=scratch, q=q, k_cache=k, v_cache=v,
                       output=out_own, page_table=pt, cache_seqlens=cs,
                       cu_seqlens_q=cu, q2k_indices=None)
        paged_attention_forward(binding=b)
        b2 = follower.bind(scratch=scratch, q=q2, k_cache=k, v_cache=v,
                           output=out_fol, page_table=pt, cache_seqlens=cs,
                           cu_seqlens_q=cu, q2k_indices=None,
                           skip_replay_metadata_update=skip_follower)
        paged_attention_forward(binding=b2)

    for L in (1000, 8192, width * PAGE - 5):
        cs.fill_(L)
        step(False)
        torch.cuda.synchronize()
        ref = out_fol.clone()
        out_fol.zero_()
        step(True)
        torch.cuda.synchronize()
        assert torch.equal(out_fol, ref), f"eager L={L}: meta-once mismatch"
        print(f"  [eager L={L} split={split}] bitwise-equal PASS", flush=True)

    # graph capture: owner captures updates; follower captures none
    g = torch.cuda.CUDAGraph()
    step(True)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        step(True)
    for L in (700, 4096, 5000, width * PAGE - 5):
        cs.fill_(L)
        g.replay()
        torch.cuda.synchronize()
        got = out_fol.clone()
        step(False)
        torch.cuda.synchronize()
        c = _cos(got, out_fol)
        assert torch.equal(got, out_fol), f"graph L={L}: cos={c:.7f}"
        print(f"  [graph L={L} split={split}] bitwise-equal PASS", flush=True)


def main():
    for split in (True, False):
        print(f"== test_meta_once_dense split={split}", flush=True)
        test_meta_once_dense(split=split)
    print("== ALL PASS ==", flush=True)


if __name__ == "__main__":
    sys.exit(main())
