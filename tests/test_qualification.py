"""CPU orchestration tests; no GPU packages required."""
from types import SimpleNamespace
from pathlib import Path
import json
import subprocess
import sys

import pytest

from kvolt_runtime.contracts import RequestSpec, StopReason
from kvolt_runtime.qualification.reference import GenerationRecord, generate_candidate, generate_reference
from kvolt_runtime.qualification.compare import compare_generation, load_cohort, build_cases


@pytest.mark.parametrize('first,nmax,expected,reason', [(128009,512,0,StopReason.EOS),(17,1,0,StopReason.LENGTH),(17,2,1,StopReason.LENGTH)])
def test_candidate_skip_first_eos_and_cap_one(first,nmax,expected,reason):
    events=[]
    cache=SimpleNamespace(metadata=SimpleNamespace(length=1024))
    def decode(handle, previous):
        events.append(('decode',previous)); handle.metadata.length+=1
        return SimpleNamespace(token_id=19,cache=handle)
    backend=SimpleNamespace(prefill_raw=lambda request:SimpleNamespace(token_id=first,cache=cache),
        pack_initial=lambda handle,group:events.append(('pack',group)),decode_one=decode,
        release_request=lambda handle:events.append(('release',handle)))
    request=RequestSpec('fixture',tuple([1]*1024),'a'*64,nmax)
    record=generate_candidate(backend,request,8)
    assert record.decode_forwards==expected and record.stop_reason==reason.value
    assert record.final_cache_length==1024+expected
    assert sum(e[0]=='pack' for e in events)==int(expected>0)
    assert events[-1][0]=='release'


def test_candidate_releases_on_pack_failure():
    events=[]
    cache=SimpleNamespace(metadata=SimpleNamespace(length=1024))
    def fail(*args): raise RuntimeError('injected')
    backend=SimpleNamespace(prefill_raw=lambda r:SimpleNamespace(token_id=1,cache=cache),pack_initial=fail,
        release_request=lambda h:events.append(h))
    with pytest.raises(RuntimeError): generate_candidate(backend,RequestSpec('x',(1,)*1024,'b'*64,2),16)
    assert events==[cache]


def test_exact_token_stop_and_length_differences_are_reported():
    a=GenerationRecord((1,2,128009),'eos',2,1026)
    assert compare_generation(a,a) is None
    assert compare_generation(a,GenerationRecord((1,4,128009),'eos',2,1026))=={'kind':'token','index':1,'expected':2,'actual':4}
    assert compare_generation(a,GenerationRecord((1,2),'length',1,1025))['index']==2
    assert compare_generation(a,GenerationRecord(a.token_ids,'length',2,1026))['kind']=='stop_reason'


def test_frozen_cohort_cases_and_tamper_rejection(tmp_path):
    root=Path(__file__).resolve().parents[1]
    rows,proof=load_cohort(root/'artifacts/m1/cohort/inputs.json',root/'artifacts/m1/cohort/manifest.json')
    assert len(rows)==30 and len(build_cases(rows,'full'))==150
    assert len(build_cases(rows,'smoke'))==15
    assert {c.request.nmax for c in build_cases(rows,'boundaries')}=={1,2,512}
    damaged=tmp_path/'inputs.json'; damaged.write_text('{}')
    with pytest.raises(Exception,match='hash'): load_cohort(damaged,root/'artifacts/m1/cohort/manifest.json')


def test_tool_help_has_no_torch_import_or_gpu_access():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'tools/qualify_backend.py'),'--help'],cwd=root,capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    assert '--suite' in result.stdout and '--lock' in result.stdout


def test_disk_chunks_are_bounded_and_cover_exact_order():
    from kvolt_runtime.qualification.compare import bounded_chunks,initial_cache_bytes
    root=Path(__file__).resolve().parents[1]
    rows,_=load_cohort(root/'artifacts/m1/cohort/inputs.json',root/'artifacts/m1/cohort/manifest.json')
    cases=build_cases(rows,'full')
    budget=3072*1024**2
    chunks=bounded_chunks(cases,budget)
    assert len(chunks)>1 and [case.key for chunk in chunks for case in chunk]==[case.key for case in cases]
    assert all(sum(initial_cache_bytes(case.request.length,case.group) for case in chunk)<=budget for chunk in chunks)
    assert initial_cache_bytes(128,8)==32*((256+512)*128+2048*128)
    with pytest.raises(Exception,match='exceeds'): bounded_chunks(cases,1)


def test_first_token_check_precedes_pack_and_preserves_release():
    cache=SimpleNamespace(metadata=SimpleNamespace(length=1024))
    events=[]
    backend=SimpleNamespace(prefill_raw=lambda r:SimpleNamespace(token_id=17,cache=cache),
        pack_initial=lambda *args:events.append('pack'),release_request=lambda h:events.append('release'))
    def check(token):
        assert token==17
        events.append('check')
        raise RuntimeError('first token mismatch')
    with pytest.raises(RuntimeError,match='first token mismatch'):
        generate_candidate(backend,RequestSpec('eos-mismatch',(1,)*1024,'a'*64,2),8,first_token=check)
    assert events==['check','release']


def test_nonfinite_mismatch_values_remain_serializable():
    from kvolt_runtime.qualification.compare import _json_scalar
    assert json.dumps([_json_scalar(float('nan')),_json_scalar(float('inf')),_json_scalar(float('-inf'))],allow_nan=False)=='["nan", "inf", "-inf"]'
    assert _json_scalar(1.5)==1.5
