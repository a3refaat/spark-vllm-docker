"""Phase-3 (EAGLE3) offline gate for the KVarN stack.

A) Flush equivalence under speculative rejection: a CLEAN single-token
   stream and a SPEC stream (verify steps write draft junk past the
   committed boundary; rejections rewind and rewrite) must end with
   BYTE-IDENTICAL packed records for every flushed tile.
B) Regression probe: a tile whose SCHEDULED fill crossed 128 but whose
   COMMITTED fill has not must stay pool-resident (the pre-fix code
   packed it -> permanent junk + orphaned rewrites).
C) Fused decode self-consistency at q_len=4: row j of a verify batch ==
   a q_len=1 call at that row's kv_len. Plus ws-path parity at q_len=4.
D) Indexer decode scorer self-consistency at q_len=4.
"""
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/common")
import kvarn_sparse as ks
import kvarn_indexer as ki

dev = torch.device("cuda")
torch.manual_seed(0)

HK, D, G = 2, 128, 128
NPAGES, WIDTH = 32, 16
N_FINAL = 392  # committed tokens at end (tiles 0,1,2 full; tile 3 partial)
PAGES = [1, 2, 3, 4]  # tiles 0..3 of the one request


def mkcfg(spec_tokens):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=2048),
        speculative_config=(SimpleNamespace(num_speculative_tokens=spec_tokens)
                            if spec_tokens else None),
        model_config=SimpleNamespace(max_model_len=WIDTH * G),
        cache_config=SimpleNamespace(num_gpu_blocks=None),
    )


def mkstream(spec_tokens):
    group = ks.KVarNSparseGroup(mkcfg(spec_tokens))
    layer = ks.KVarNSparseLayer("l0", HK, D, "kvarn_k4v2_g128")
    cache = torch.zeros(NPAGES, HK, layer.cfg.tile_bytes_aligned,
                        dtype=torch.uint8, device=dev)
    assert layer.ensure(cache, group)
    return group, layer


# one shared correct token stream + per-position junk (drafts that get rejected)
K_T = torch.randn(400, HK, D, device=dev, dtype=torch.bfloat16) * 1.5
V_T = torch.randn(400, HK, D, device=dev, dtype=torch.bfloat16)
K_J = torch.randn(400, HK, D, device=dev, dtype=torch.bfloat16) * 3.0
V_J = torch.randn(400, HK, D, device=dev, dtype=torch.bfloat16) * 3.0

bt_gpu = torch.full((1, WIDTH), -1, dtype=torch.int32, device=dev)
bt_gpu[0, :4] = torch.tensor(PAGES, dtype=torch.int32, device=dev)


def step(group, layer, c0, s1, correct_upto):
    """One serving step: builder (flush/alloc) then forward store of
    positions [c0, s1). Positions < correct_upto get final values, the
    rest junk (drafts to be rejected)."""
    cm = SimpleNamespace(
        num_reqs=1,
        seq_lens_cpu=torch.tensor([s1]),
        num_computed_tokens_cpu=torch.tensor([c0]),
        block_table_tensor=bt_gpu,
    )
    group.builder_step(cm)
    if s1 > c0:
        pos = torch.arange(c0, s1, device=dev)
        K = torch.where((pos < correct_upto)[:, None, None], K_T[c0:s1], K_J[c0:s1])
        V = torch.where((pos < correct_upto)[:, None, None], V_T[c0:s1], V_J[c0:s1])
        # slot mapping may span tiles: build per position
        smap = torch.tensor(
            [PAGES[p // G] * G + p % G for p in range(c0, s1)],
            dtype=torch.int32, device=dev)
        layer.store(K.reshape(s1 - c0, HK * D), V.reshape(s1 - c0, HK * D),
                    smap, group)


# ── CLEAN stream: prefill 380, then single-token decode to 392 ──────────────
gC, lC = mkstream(0)
step(gC, lC, 0, 380, 400)
for c in range(380, N_FINAL):
    step(gC, lC, c, c + 1, 400)
step(gC, lC, N_FINAL, N_FINAL, 400)  # trailing builder-only step (final walk)

# ── SPEC stream: verify steps with rejection + one all-accept jump ──────────
gS, lS = mkstream(3)
step(gS, lS, 0, 380, 400)
step(gS, lS, 380, 384, 381)   # T[380] + 3 junk drafts (all rejected)
# B) regression probe: scheduled crossed tile-2 boundary (384) but committed
#    is only 381 -> tile 2 MUST still be pool-resident
cm_probe = SimpleNamespace(num_reqs=1, seq_lens_cpu=torch.tensor([385]),
                           num_computed_tokens_cpu=torch.tensor([381]),
                           block_table_tensor=bt_gpu)
gS.builder_step(cm_probe)
assert int(gS.block_to_slot[PAGES[2]]) >= 0, \
    "REGRESSION: tile 2 packed while last token (383) still speculative"
print("B) uncommitted tile stays pool-resident: PASS")
pos = torch.arange(381, 385, device=dev)
K = torch.where((pos < 382)[:, None, None], K_T[381:385], K_J[381:385])
V = torch.where((pos < 382)[:, None, None], V_T[381:385], V_J[381:385])
smap = torch.tensor([PAGES[p // G] * G + p % G for p in range(381, 385)],
                    dtype=torch.int32, device=dev)
lS.store(K.reshape(4, HK * D), V.reshape(4, HK * D), smap, gS)
step(gS, lS, 382, 386, 383)   # keep rejecting drafts one at a time
step(gS, lS, 383, 387, 384)
step(gS, lS, 384, 388, 385)   # builder here must flush tile 2 (c0=384)
assert int(gS.block_to_slot[PAGES[2]]) == -1, "tile 2 should pack at c0=384"
step(gS, lS, 385, 389, 400)   # all 3 drafts correct -> all accepted
step(gS, lS, 389, 393, 390)   # back to rejecting
step(gS, lS, 390, 394, 391)
step(gS, lS, 391, 395, 392)
step(gS, lS, N_FINAL, N_FINAL, 400)  # trailing builder-only step

# ── A) packed-record byte equality (tiles 1 and 2; tile 0 sink later) ───────
for t in (1, 2):
    a, b = lC.kv_cache[PAGES[t]], lS.kv_cache[PAGES[t]]
    assert int(gC.block_to_slot[PAGES[t]]) == -1
    assert int(gS.block_to_slot[PAGES[t]]) == -1
    assert torch.equal(a, b), f"tile {t} records differ (spec rejection leaked)"
print("A) tiles 1,2 byte-identical across CLEAN/SPEC: PASS")

# ── C) fused decode q_len=4 self-consistency (on the SPEC stream state:
#     its ws buffers are sized for q_len_max=4; mixed sink-pool/record/
#     partial-pool tiles) ────────────────────────────────────────────────────
HQ = 8
sm_scale = 0.088
topk4 = torch.zeros(HK, 4, 4, dtype=torch.int32, device=dev)
topk4[:, :, 0] = 0
topk4[:, :, 1] = 1
topk4[:, :, 2] = 2
topk4[:, :, 3] = 3
q4 = torch.randn(4, HQ, D, device=dev, dtype=torch.bfloat16)
sl4 = torch.tensor([N_FINAL], dtype=torch.int32, device=dev)
out4 = torch.empty_like(q4)
ks.kvarn_sparse_attn_decode_fused(q4, lS, gS, topk4, bt_gpu, sl4, HK,
                                  sm_scale, out4, 4)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0).item()


ok = True
for j in range(4):
    q1 = q4[j:j + 1].contiguous()
    o1 = torch.empty_like(q1)
    sl1 = torch.tensor([N_FINAL - 3 + j], dtype=torch.int32, device=dev)
    ks.kvarn_sparse_attn_decode_fused(q1, lS, gS, topk4[:, j:j + 1].contiguous(),
                                      bt_gpu, sl1, HK, sm_scale, o1, 1)
    c = cos(out4[j], o1[0])
    md = (out4[j].float() - o1[0].float()).abs().max().item()
    print(f"  fused row {j} (kv_len={N_FINAL - 3 + j}): cos={c:.6f} maxdiff={md:.5f}")
    ok &= c > 0.9999
assert ok
# ws path at q_len=4 vs fused
from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn_decode
ws4, btws4 = ks.build_ws_decode(lS, gS, topk4, bt_gpu, 4, 4)
q4r = ks.rotate_heads(q4, lS._H16)
ow = torch.empty_like(q4)
minimax_m3_sparse_attn_decode(q4r, ws4, topk4, btws4, sl4, HK, sm_scale, ow, 4)
ow = ks.rotate_heads(ow, lS._H16)
cw = cos(out4, ow)
print(f"C) fused vs ws @q_len=4: cos={cw:.6f}")
assert cw > 0.999

# ── D) indexer decode scorer q_len=4 self-consistency ───────────────────────
from vllm.utils.math_utils import round_up
from vllm.platforms import current_platform
H_IDX = 2
gI = ks.KVarNSparseGroup(mkcfg(3))
ki._STATE = gI
lI = ki.KVarNIndexerLayer("idx0", D)
lo = lI.layout
icache = torch.zeros(NPAGES, lo["REC"], dtype=torch.uint8, device=dev)
assert lI.ensure(icache, gI)
SEQI = 5 * G + 77
K_i = torch.randn(SEQI, D, device=dev, dtype=torch.bfloat16)
bt_i = torch.arange(NPAGES, dtype=torch.int32, device=dev).unsqueeze(0)
for p in range(6):
    s = gI._free.pop(); gI._slot_of[p] = s; gI.block_to_slot[p] = s
lI.store(K_i, torch.arange(SEQI, dtype=torch.int32, device=dev), gI)
gI._flush([0, 1, 2, 3])  # page 4 pool-resident + page 5 partial (mixed)
iq4 = torch.randn(4, H_IDX, D, device=dev, dtype=torch.bfloat16)
iq4r = torch.matmul(iq4.to(torch.float16), lI._H16).to(iq4.dtype)
use_pdl = current_platform.is_arch_support_pdl()
pdl = {"launch_pdl": True} if use_pdl else {}
max_block = (SEQI + G - 1) // G
sc4 = torch.empty((H_IDX, 4, round_up(max_block, 16)), dtype=torch.float32, device=dev)
sli = torch.tensor([SEQI], dtype=torch.int32, device=dev)
ki.kvarn_index_decode_score(iq4r, lI, bt_i, sli, SEQI, 0, 1, D ** -0.5, 4,
                            sc4, 2, use_pdl, pdl)
ok = True
for j in range(4):
    sc1 = torch.empty((H_IDX, 1, round_up(max_block, 16)), dtype=torch.float32,
                      device=dev)
    sl1 = torch.tensor([SEQI - 3 + j], dtype=torch.int32, device=dev)
    ki.kvarn_index_decode_score(iq4r[j:j + 1].contiguous(), lI, bt_i, sl1,
                                SEQI, 0, 1, D ** -0.5, 1, sc1, 2, use_pdl, pdl)
    nb = (SEQI - 3 + j + G - 1) // G
    c = cos(sc4[:, j, :nb], sc1[:, 0, :nb])
    md = (sc4[:, j, :nb] - sc1[:, 0, :nb]).abs().max().item()
    print(f"  idx row {j}: cos={c:.6f} maxdiff={md:.4f}")
    ok &= md < 1e-3
assert ok
print("D) indexer scorer q_len=4 self-consistent: PASS")

# ── F) fused prefill (2D) vs ws path parity ────────────────────────────────
# production head shape: gqa MUST be 16 (prefill heuristic has no min-16
# clamp; 64q/4kv gives gqa 16 at any TP). Mixed sink-pool/record/partial.
HQ_P = 32  # 32 q heads / 2 kv heads == production TP2 (gqa 16)
NQ, PREFIX = 100, 292  # chunk: query rows 292..391 of the 392-token seq
qp = torch.randn(NQ, HQ_P, D, device=dev, dtype=torch.bfloat16)
topkp = torch.zeros(HK, NQ, 4, dtype=torch.int32, device=dev)
for b in range(4):
    topkp[:, :, b] = b
cu_q = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
sl_p = torch.tensor([392], dtype=torch.int32, device=dev)
pl_p = torch.tensor([PREFIX], dtype=torch.int32, device=dev)
out_f = torch.empty_like(qp)
ks.kvarn_sparse_attn_prefill_fused(qp, lS, gS, topkp, bt_gpu, cu_q, sl_p,
                                   pl_p, NQ, HK, sm_scale, out_f)
from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn
gS._seqlens_cpu = torch.tensor([392])
ws_p, btws_p = ks.build_ws_prefill(lS, gS, bt_gpu, 1)
qp_rot = ks.rotate_heads(qp, lS._H16)
out_w = torch.empty_like(qp)
minimax_m3_sparse_attn(qp_rot, ws_p, topkp, btws_p, cu_q, sl_p, pl_p,
                       NQ, HK, sm_scale, out_w)
out_w = ks.rotate_heads(out_w, lS._H16)
torch.cuda.synchronize()
cf = cos(out_f, out_w)
mdf = (out_f.float() - out_w.float()).abs().max().item()
print(f"F) fused prefill vs ws: cos={cf:.6f} maxdiff={mdf:.4f}")
assert cf > 0.999
# causality probe: earliest chunk row (global pos 292) must ignore
# positions > 292 — compare against a ws run with seq truncated there
out_f1 = torch.empty_like(qp[:1])
ks.kvarn_sparse_attn_prefill_fused(qp[:1].contiguous(), lS, gS,
                                   topkp[:, :1].contiguous(), bt_gpu, 
                                   torch.tensor([0, 1], dtype=torch.int32, device=dev),
                                   torch.tensor([293], dtype=torch.int32, device=dev),
                                   pl_p, 1, HK, sm_scale, out_f1)
c1 = cos(out_f[0], out_f1[0])
print(f"   causal row-0 self-consistency: cos={c1:.6f}")
assert c1 > 0.9999

# ── E) reclaim: committed sink flushed, uncommitted tail discarded ──────────
bt_new = torch.full((1, WIDTH), -1, dtype=torch.int32, device=dev)
bt_new[0, 0] = 30
for g_, l_ in ((gC, lC), (gS, lS)):
    cm = SimpleNamespace(num_reqs=1, seq_lens_cpu=torch.tensor([5]),
                         num_computed_tokens_cpu=torch.tensor([0]),
                         block_table_tensor=bt_new)
    g_.builder_step(cm)
    assert int(g_.block_to_slot[PAGES[0]]) == -1, "sink not reclaimed"
    assert int(g_.block_to_slot[PAGES[3]]) == -1, "tail not reclaimed"
assert torch.equal(lC.kv_cache[PAGES[0]], lS.kv_cache[PAGES[0]]), \
    "sink records differ"
assert int(lC.kv_cache[PAGES[0]].to(torch.int64).sum()) > 0, \
    "sink was discarded, not flushed (prefix-cache hit would read zeros)"
assert int(lC.kv_cache[PAGES[3]].to(torch.int64).sum()) == 0, \
    "uncommitted partial tile must NOT be packed"
assert int(lS.kv_cache[PAGES[3]].to(torch.int64).sum()) == 0, \
    "uncommitted partial tile must NOT be packed (spec stream)"
print("E) reclaim: sink flushed, uncommitted tail discarded: PASS")
print("ALL PASS")
