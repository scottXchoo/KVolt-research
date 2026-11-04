"""Strict paired comparison, frozen cases, and first-difference evidence."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from ..config import GROUPS
from ..contracts import RequestSpec, token_ids_sha256
from ..errors import QualificationError

INPUTS_SHA256 = 'e9f0c05e0d21e13356f6a64d1176b03bcff613c708b0b0fd72228f711aadc20d'
MANIFEST_SHA256 = '5fddb4749e0f7353fb26864e3cc3e3f319488e38a5f20b3e4f91bcb7369023bf'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cohort(inputs, manifest):
    proof = {}
    for name, path, expected in [('inputs',inputs,INPUTS_SHA256),('manifest',manifest,MANIFEST_SHA256)]:
        actual = sha256(path)
        if actual != expected:
            raise QualificationError(f'frozen {name} hash mismatch')
        proof[name] = {'path':str(Path(path).resolve()), 'sha256':actual}
    data = json.loads(Path(inputs).read_text())
    rows = data['items']
    if len(rows) != 30 or len({row['id'] for row in rows}) != 30:
        raise QualificationError('frozen cohort must contain 30 distinct inputs')
    for row in rows:
        ids = tuple(row['input_ids'])
        if row['length'] not in (1024,4096,8192) or len(ids) != row['length']:
            raise QualificationError('frozen input length mismatch')
        if token_ids_sha256(ids) != row['input_hash']:
            raise QualificationError('frozen token hash mismatch')
        if hashlib.sha256(row['prompt'].encode()).hexdigest() != row['prompt_sha256']:
            raise QualificationError('frozen prompt hash mismatch')
    return rows, proof


@dataclass(frozen=True)
class Case:
    key: str
    request: RequestSpec
    group: int


def build_cases(rows, suite):
    if suite not in ('full','smoke','boundaries'):
        raise QualificationError('unknown qualification suite')
    selected = rows if suite == 'full' else [next(row for row in rows if row['length']==length) for length in (1024,4096,8192)]
    caps = (1,2,512) if suite == 'boundaries' else ((128,) if suite == 'full' else (4,))
    return [Case(f'{row["id"]}-g{group}-n{cap}',RequestSpec(row['id'],tuple(row['input_ids']),row['prompt_sha256'],cap),group)
            for row in selected for group in GROUPS for cap in caps]


def compare_generation(expected, actual):
    for index in range(max(len(expected.token_ids),len(actual.token_ids))):
        left = expected.token_ids[index] if index < len(expected.token_ids) else None
        right = actual.token_ids[index] if index < len(actual.token_ids) else None
        if left != right:
            return {'kind':'token','index':index,'expected':left,'actual':right}
    for name in ('stop_reason','decode_forwards','final_cache_length'):
        if getattr(expected,name) != getattr(actual,name):
            return {'kind':name,'expected':getattr(expected,name),'actual':getattr(actual,name)}
    return None


def _json_scalar(value):
    if isinstance(value,float) and not math.isfinite(value):
        return repr(value)
    return value


def compare_cache(expected, actual, torch):
    """Tensor equality is bit-exact, including signed zeros; report first byte."""
    from ..backend.cache import FIELDS
    if len(expected) != len(actual):
        return {'kind':'layers','expected':len(expected),'actual':len(actual)}
    for layer, (left,right) in enumerate(zip(expected,actual)):
        if len(left) != len(right):
            return {'kind':'fields','layer':layer,'expected':len(left),'actual':len(right)}
        for field,(a,b) in enumerate(zip(left,right)):
            where = {'layer':layer,'field':FIELDS[field]}
            if a is None or b is None or type(a) is int or type(b) is int:
                if a != b:
                    return dict(where,kind='presence_or_length',expected=a,actual=b)
                continue
            a,b = a.detach().cpu().contiguous(),b.detach().cpu().contiguous()
            if a.shape != b.shape or a.dtype != b.dtype:
                return dict(where,kind='shape_or_dtype',expected=[list(a.shape),str(a.dtype)],actual=[list(b.shape),str(b.dtype)])
            ab,bb = a.view(torch.uint8).reshape(-1),b.view(torch.uint8).reshape(-1)
            mismatch = (ab != bb).nonzero()
            if mismatch.numel():
                offset = int(mismatch[0].item())
                element = offset // a.element_size()
                return dict(where,kind='value',byte_offset=offset,flat_index=element,
                            expected=_json_scalar(a.reshape(-1)[element].item()),actual=_json_scalar(b.reshape(-1)[element].item()),
                            expected_byte=int(ab[offset].item()),actual_byte=int(bb[offset].item()))
    return None


def initial_cache_bytes(length, group):
    """Exact logical bytes for native32-layer K2V2/R128 fields."""
    kp=length//128*128
    kr=length%128
    vp=max(0,length-128)
    vr=min(length,128)
    return 32*((256+4096//group)*(kp+vp)+2048*(kr+vr))


def bounded_chunks(cases, byte_budget):
    chunks=[]; current=[]; used=0
    for case in cases:
        size=initial_cache_bytes(case.request.length,case.group) if case.request.nmax>1 else 0
        if size>byte_budget:
            raise QualificationError('one initial cache exceeds disk proof budget')
        if current and used+size>byte_budget:
            chunks.append(current); current=[]; used=0
        current.append(case); used+=size
    if current: chunks.append(current)
    return chunks
