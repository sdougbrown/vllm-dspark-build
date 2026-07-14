#!/usr/bin/env python3
"""Verify the vllm-dspark-env stack is healthy for DSpark + B12X + nvfp4_ds_mla."""
import sys

fail = []

def check(label: str, fn):
    try:
        fn()
        print(f"[OK] {label}")
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        fail.append(label)

# 1. vllm loads from _forks/vllm
def _vllm_path():
    import vllm, pathlib
    p = pathlib.Path(vllm.__file__)
    assert "_forks/vllm" in str(p), f"unexpected vllm path: {p}"
check("vllm loads from ~/Code/_forks/vllm", _vllm_path)

# 2. nvfp4_ds_mla is a valid CacheDType
def _nvfp4_dtype():
    from vllm.config.cache import CacheDType
    assert "nvfp4_ds_mla" in CacheDType.__args__, \
        f"nvfp4_ds_mla not in CacheDType: {CacheDType.__args__}"
check("nvfp4_ds_mla in CacheDType", _nvfp4_dtype)

# 3. DSpark speculator exists (new V2 worker path)
def _dspark_spec():
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    assert DSparkSpeculator is not None
check("DSparkSpeculator importable", _dspark_spec)

# 4. FlashInferB12xExperts available
def _b12x_experts():
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
        FlashInferB12xExperts,
    )
    assert FlashInferB12xExperts is not None
check("FlashInferB12xExperts importable", _b12x_experts)

# 5. flashinfer.fused_moe.B12xMoEWrapper available (runtime kernel)
def _b12x_wrapper():
    from flashinfer.fused_moe import B12xMoEWrapper
    assert B12xMoEWrapper is not None
check("flashinfer.B12xMoEWrapper importable", _b12x_wrapper)

# 6. DSparkDraftModel in registry
def _registry():
    from vllm.model_executor.models.registry import _VLLM_MODELS
    assert "DSparkDraftModel" in _VLLM_MODELS, \
        f"DSparkDraftModel missing from registry"
check("DSparkDraftModel in model registry", _registry)

# 7. nvfp4_ds_mla in sparse_mla supported dtypes
def _sparse_mla():
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
    assert "nvfp4_ds_mla" in DeepseekV4FlashMLABackend.supported_kv_cache_dtypes, \
        f"nvfp4_ds_mla missing from sparse_mla supported dtypes"
check("nvfp4_ds_mla in DeepseekV4FlashMLABackend.supported_kv_cache_dtypes", _sparse_mla)

# 8. has_flashinfer_b12x_moe() returns True
def _has_b12x():
    from vllm.utils.flashinfer import has_flashinfer_b12x_moe
    assert has_flashinfer_b12x_moe(), "has_flashinfer_b12x_moe() returned False"
check("has_flashinfer_b12x_moe() == True", _has_b12x)

print()
if fail:
    print(f"FAILED: {len(fail)} check(s): {', '.join(fail)}")
    sys.exit(1)
else:
    print("All checks passed — vllm-dspark-env is ready.")
