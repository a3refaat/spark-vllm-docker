"""Unit tests for the Phase-1 unified decode policy (descriptive only).

CPU-only: validates the policy dataclass vocabulary, cache-key stability and
kv-quant normalization without touching CUDA or the planner.
"""

import pytest

from b12x.attention.paged.tuning.policy import (
    PagedDecodePolicy,
    normalize_kv_quant,
)


def _policy(**overrides):
    base = dict(
        family="dense",
        mode="decode",
        kv_quant="none",
        kv_dtype="torch.bfloat16",
        page_size=128,
        head_dim_qk=128,
        head_dim_vo=128,
        gqa_group_size=16,
        max_batch=1,
        max_total_q=1,
        max_page_table_width=512,
        cache_page_capacity=512,
        split_kv=False,
        chunk_pages=None,
        max_chunks_per_request=8,
        graph_ctas_per_sm=4,
        metadata_strategy="captured-update",
        backend_variant="base",
        use_cuda_graph=True,
    )
    base.update(overrides)
    return PagedDecodePolicy(**base)


def test_policy_cache_key_is_stable_and_hashable():
    a = _policy()
    b = _policy()
    assert a == b
    assert a.cache_key() == b.cache_key()
    assert hash(a.cache_key()) == hash(b.cache_key())


def test_policy_cache_key_distinguishes_plan_pinning_fields():
    base = _policy()
    for field, value in (
        ("max_batch", 2),
        ("max_page_table_width", 1024),
        ("cache_page_capacity", 256),
        ("kv_dtype", "torch.float8_e4m3fn"),
        ("kv_quant", "nvfp4"),
        ("page_size", 64),
        ("split_kv", True),
        ("chunk_pages", 16),
        ("mode", "verify"),
    ):
        assert _policy(**{field: value}).cache_key() != base.cache_key(), field


def test_policy_describe_covers_every_field():
    text = _policy().describe()
    for name in ("family=", "mode=", "kv_quant=", "split_kv=", "chunk_pages=",
                 "graph_ctas_per_sm=", "metadata_strategy=", "backend_variant="):
        assert name in text


@pytest.mark.parametrize(
    "field,value",
    [
        ("family", "moe"),
        ("mode", "train"),
        ("kv_quant", "int8"),
        ("metadata_strategy", "adhoc"),
        ("backend_variant", "custom"),
    ],
)
def test_policy_rejects_unknown_vocabulary(field, value):
    with pytest.raises(ValueError):
        _policy(**{field: value})


@pytest.mark.parametrize(
    "kv_quant,kv_dtype,expected",
    [
        ("none", "torch.bfloat16", "none"),
        (None, "torch.bfloat16", "none"),
        ("none", "torch.float8_e4m3fn", "fp8"),
        ("none", "fp8_e4m3fn", "fp8"),
        ("nvfp4", "torch.uint8", "nvfp4"),
    ],
)
def test_normalize_kv_quant(kv_quant, kv_dtype, expected):
    assert normalize_kv_quant(kv_quant, kv_dtype) == expected
