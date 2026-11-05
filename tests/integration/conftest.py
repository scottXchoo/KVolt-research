"""Opt-in GPU fixture: lock before torch import, no device configuration writes."""
import importlib
import os
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope='module')
def gpu_context():
    if os.environ.get('KVOLT_RUN_GPU') != '1':
        pytest.skip('set KVOLT_RUN_GPU=1 on a coordinated pinned AGX to run GPU correctness checks')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'tools'))
    from qualify_backend import hardware_lock
    from kvolt_runtime.backend.provenance import validate_sources,import_pinned_source,checked_file
    baseline=Path(os.environ.get('KVOLT_BASELINE','/home/agx3/kvolt'))
    kivi=Path(os.environ.get('KVOLT_KIVI_ROOT',str(baseline/'third_party/KIVI')))
    native_profile=os.environ.get('KVOLT_NATIVE_PROFILE','agx1')
    snapshot=Path(os.environ.get('KVOLT_SNAPSHOT',str(baseline/'.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659')))
    with hardware_lock(Path(os.environ.get('KVOLT_HARDWARE_LOCK','/run/lock/kvolt-p3-hardware.lock'))):
        provenance=validate_sources(kivi,snapshot,native_profile=native_profile)
        checked_file(baseline/'scripts/k2_decode_lut/backend.py','2702735c01b64bd84f146a8b042b96960aa7540ada52b98208caf7421295c859')
        checked_file(baseline/'scripts/p6/cache.py','f0de50eae45cf204f63dd29837ba5b7dd33f2b21b0dc9544473e4c8c5258f24f')
        sys.path.insert(0,str(kivi)); sys.path.insert(0,str(baseline))
        import torch
        import_pinned_source(kivi,provenance)
        helper=importlib.import_module('scripts.k2_decode_lut.backend')
        quant=importlib.import_module('quant.new_pack').triton_quantize_and_pack_along_last_dim
        matmul=importlib.import_module('quant.matmul').cuda_bmm_fA_qB_outer
        yield torch,helper,quant,matmul
        torch.cuda.synchronize()


def fp16_parameters_only(attention,torch):
    """Match FP16 weights while preserving original FP32 RoPE buffers."""
    with torch.no_grad():
        for parameter in attention.parameters():
            if parameter.is_floating_point():
                parameter.data=parameter.data.to(dtype=torch.float16)


@pytest.fixture(scope='module')
def gpu_attention(gpu_context):
    """One unchanged native attention module, never full model weights."""
    torch,_,_,_=gpu_context
    from kvolt_runtime.backend.vendor.llama_kivi import LlamaFlashAttention_KIVI,LlamaConfig
    baseline=Path(os.environ.get('KVOLT_BASELINE','/home/agx3/kvolt'))
    snapshot=Path(os.environ.get('KVOLT_SNAPSHOT',str(baseline/'.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659')))
    config=LlamaConfig.from_pretrained(str(snapshot),local_files_only=True)
    config.k_bits=config.v_bits=2
    config.residual_length=128
    config.group_size=8
    config.use_flash=True
    torch.manual_seed(20261104)
    with torch.device('cuda:0'):
        attention=LlamaFlashAttention_KIVI(config).eval()
    fp16_parameters_only(attention,torch)
    yield attention
    del attention
    torch.cuda.empty_cache()
