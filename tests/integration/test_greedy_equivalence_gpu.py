"""The executable harness owns its lock and sequential model lifetimes."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark=pytest.mark.gpu


def test_frozen_greedy_suite(tmp_path):
    if os.environ.get('KVOLT_RUN_GPU')!='1':
        pytest.skip('opt-in pinned AGX correctness qualification')
    root=Path(__file__).resolve().parents[2]
    baseline=Path(os.environ.get('KVOLT_BASELINE','/home/agx3/kvolt'))
    snapshot=os.environ.get('KVOLT_SNAPSHOT',str(baseline/'.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'))
    output=Path(os.environ.get('KVOLT_QUALIFICATION_OUTPUT',str(tmp_path/'qualification')))
    result=subprocess.run([sys.executable,str(root/'tools/qualify_backend.py'),'--suite',os.environ.get('KVOLT_GPU_SUITE','smoke'),
                           '--snapshot',snapshot,'--output',str(output),
                           '--kivi-root',os.environ.get('KVOLT_KIVI_ROOT',str(baseline/'third_party/KIVI')),
                           '--native-profile',os.environ.get('KVOLT_NATIVE_PROFILE','agx1'),
                           '--lock',os.environ.get('KVOLT_HARDWARE_LOCK','/run/lock/kvolt-p3-hardware.lock')],cwd=root)
    assert result.returncode==0, f'first-mismatch/provenance evidence: {output}'
