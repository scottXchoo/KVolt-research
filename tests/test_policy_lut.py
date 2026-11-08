"""CPU LUT contracts and independent checks against the sealed published data."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from kvolt_runtime.config import GROUPS
from kvolt_runtime.errors import ContractError
from kvolt_runtime.policy.lut import MeasuredLUTRepository, MANIFEST_SHA256, SOURCE_COMMIT
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.types import FREQUENCIES_MHZ as F, PERFORMANCE_LENGTHS as P, QUALITY_LENGTHS as Q, LUTError


@pytest.fixture(scope='module')
def bundle():
    configured = os.environ.get('KVOLT_LUT_BUNDLE')
    path = Path(configured) if configured else Path(__file__).resolve().parents[2] / 'KVolt-lut-source/data/luts/agx2-repeat5-20261103'
    if not path.is_dir():
        pytest.skip('sealed measured integration unavailable: set KVOLT_LUT_BUNDLE')
    return path


@pytest.fixture(scope='module')
def measured(bundle):
    return MeasuredLUTRepository(bundle)


def read(bundle, name):
    return json.loads((bundle / name).read_text())


@pytest.mark.parametrize('scenario', ['baseline', 'flat_quality', 'nonmonotonic', 'exact_ties'])
def test_mock_complete_deterministic_synthetic(scenario):
    a, b = MockLUTRepository(scenario), MockLUTRepository(scenario)
    assert a.provenance.kind == 'synthetic'
    assert json.loads(a.provenance.metadata_json)['synthetic'] is True
    for n in [1024, 1025, 1280, 1536, 2048, 4097, 8192]:
        assert a.preflight(n) is None
        assert len(a.prefill(n)) == 8
        assert len(a.quality(n)) == 5
        assert len(a.decode(n)) == 40
        assert a.prefill(n) == b.prefill(n)
        assert a.decode(n) == b.decode(n)
        assert a.quality(n) == b.quality(n)
    if scenario == 'flat_quality':
        assert len({p.score for p in a.quality(1537)}) == 1
    if scenario == 'nonmonotonic':
        scores = [p.score for p in a.quality(1024)]
        assert scores[1] < scores[2] and scores[2] > scores[3]
    if scenario == 'exact_ties':
        assert len({(p.ttft_s, p.energy_j) for p in a.prefill(1537)}) == 1
        assert len({(p.tpot_s, p.energy_j_per_token) for p in a.decode(1537)}) == 1


@pytest.mark.parametrize('bad', [True, False, None, '1024', 1024.0, 1023, 8193])
@pytest.mark.parametrize('method', ['preflight', 'prefill', 'decode', 'quality', 'prefill_target', 'decode_target'])
def test_reject_invalid_length(bad, method):
    with pytest.raises(ContractError):
        getattr(MockLUTRepository(), method)(bad)


def test_preflight_does_not_query_online(monkeypatch):
    repo = MockLUTRepository()
    def forbidden(*args):
        pytest.fail('preflight invoked online lookup')
    for name in ['prefill', 'quality', 'decode', 'prefill_target', 'decode_target']:
        monkeypatch.setattr(repo, name, forbidden)
    repo.preflight(1777)


def test_prefill_latency_scan_skips_energy_and_details_only_query_requested_rows(monkeypatch):
    from kvolt_runtime.policy import lut
    from kvolt_runtime.policy.types import PrefillLatencyPrediction

    table = {(n, f): (n / 1024 + f / 1000, 1000.0 + f) for n in P for f in F}
    repo = MockLUTRepository(prefill=table)
    original = lut._at
    energy_queries = []

    def observed(lo, hi, bracket):
        if lo >= 1000:
            energy_queries.append(lo)
        return original(lo, hi, bracket)

    monkeypatch.setattr(lut, '_at', observed)
    latencies = repo.prefill_latencies(1536)
    assert tuple(row.frequency_mhz for row in latencies) == F
    assert all(isinstance(row, PrefillLatencyPrediction) and row.energy_j is None for row in latencies)
    assert latencies[0].ttft_s == 0.5 * table[(1024, 408)][0] + 0.5 * table[(2048, 408)][0]
    assert energy_queries == []

    details = repo.prefill_candidates(1536, (918, 510))
    assert tuple(row.frequency_mhz for row in details) == (510, 918)
    assert tuple(row.energy_j for row in details) == (1510.0, 1918.0)
    assert energy_queries == [1510.0, 1918.0]
    assert repo.prefill_candidates(1536, ()) == ()
    assert len(repo.prefill(1536)) == 8


@pytest.mark.parametrize('frequencies', [None, [408], (408, 408), (409,), (408.0,), (True,)])
def test_prefill_partial_lookup_rejects_invalid_frequency_requests(frequencies):
    with pytest.raises(ContractError):
        MockLUTRepository().prefill_candidates(1536, frequencies)


@pytest.mark.parametrize('method', ['prefill_latencies', 'prefill_candidates'])
def test_partial_prefill_lookup_preserves_length_support(method):
    args = () if method == 'prefill_latencies' else ((408,),)
    with pytest.raises(ContractError):
        getattr(MockLUTRepository(), method)(8193, *args)


def test_mock_overrides_copied_validated_and_exact():
    quality = {(n, g): 0.1 + g / 1000 for n in Q for g in GROUPS}
    repo = MockLUTRepository(quality=quality)
    quality[(1024, 8)] = 0.9
    assert repo.quality(1024)[0].score == 0.10800000000000001
    assert repo.quality(1280)[0].score == (1 - 0.5) * 0.10800000000000001 + 0.5 * 0.10800000000000001
    with pytest.raises(LUTError):
        MockLUTRepository(quality={(1024, 8): 0.5})


@pytest.mark.parametrize('value', [None, True, '0.1', float('nan'), float('inf'), -0.1, 1.1])
def test_reject_invalid_mock_quality(value):
    quality = {(n, g): 0.8 for n in Q for g in GROUPS}
    quality[(1024, 8)] = value
    with pytest.raises(LUTError):
        MockLUTRepository(quality=quality)


def test_measured_provenance_hashes_flags(bundle, measured):
    p = measured.provenance
    assert p.kind == 'measured' and p.source_device == 'AGX2'
    assert p.source_commit == SOURCE_COMMIT
    assert p.manifest_sha256 == MANIFEST_SHA256
    assert hashlib.sha256((bundle / 'bundle-manifest.json').read_bytes()).hexdigest() == MANIFEST_SHA256
    assert len(p.files_sha256) == 10
    for name, expected in p.files_sha256:
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == expected
    metadata = json.loads(p.metadata_json)
    assert metadata['quality']['runtime_eligible'] is False
    assert metadata['quality']['runtime_validation_required'] is True
    assert metadata['bundle']['limitations']


def test_all_measured_exact_knots_and_midpoints(bundle, measured):
    # PWL knots must survive exact lookups byte-for-byte as Python binary64.
    prefill = read(bundle, 'prefill-pwl.json')
    decode = read(bundle, 'decode-pwl.json')
    quality = {(r['input_length_target'], r['group']): r['mean_score'] for r in read(bundle, 'quality-lut.json')['entries']}
    for grid, query, metrics, records in [
        (P, measured.prefill, ('ttft_s', 'prefill_energy_j'), prefill),
        (P, measured.decode, ('tpot_actual_s', 'epsilon_actual_j_per_token'), decode),
    ]:
        for n in grid:
            for row in query(n):
                key = str(row.frequency_mhz) if len(metrics[0]) == 6 else f'{row.group}:{row.frequency_mhz}'
                values = tuple(next(k['value'] for k in records[key][m]['knots'] if k['length'] == n) for m in metrics)
                actual = (row.ttft_s, row.energy_j) if hasattr(row, 'ttft_s') else (row.tpot_s, row.energy_j_per_token)
                assert actual == values
                assert (row.bracket.lower, row.bracket.upper, row.bracket.weight) == (n, n, 0.0)
        for lo, hi in zip(grid, grid[1:]):
            for a, b, mid in zip(query(lo), query(hi), query((lo + hi) // 2)):
                attrs = ('ttft_s', 'energy_j') if hasattr(a, 'ttft_s') else ('tpot_s', 'energy_j_per_token')
                for attr in attrs:
                    assert getattr(mid, attr) == 0.5 * getattr(a, attr) + 0.5 * getattr(b, attr)
                assert (mid.bracket.lower, mid.bracket.upper, mid.bracket.weight) == (lo, hi, 0.5)
    for n in Q:
        assert [p.score for p in measured.quality(n)] == [quality[n, g] for g in GROUPS]
    for lo, hi in zip(Q, Q[1:]):
        assert [p.score for p in measured.quality((lo + hi) // 2)] == [0.5 * quality[lo, g] + 0.5 * quality[hi, g] for g in GROUPS]
    for row in read(bundle, 'quality-lut.json')['entries']:
        task_counts = row['n_items_by_task']
        independent_mean = sum(row['task_mean_scores'][task] * count for task, count in task_counts.items()) / sum(task_counts.values())
        assert quality[row['input_length_target'], row['group']] == pytest.approx(independent_mean, rel=1e-12, abs=1e-12)
    assert measured.quality(1024)[0].score != read(bundle, 'quality-lut.json')['entries'][0]['q']


def test_measured_independent_raw_repeat_aggregates(bundle, measured):
    # Recompute means directly from request ratios, not from PWL lookup internals.
    prefill = read(bundle, 'prefill-lut.json')['prefill']
    decode = read(bundle, 'decode-lut.json')['decode']
    assert len(prefill) == 64 and len(decode) == 320
    for rows, query, fields in [
        (prefill, measured.prefill, [('ttft_s', 'ttft_s'), ('prefill_energy_j', 'energy_j')]),
        (decode, measured.decode, [('tpot_actual_s', 'tpot_s'), ('epsilon_actual_j_per_token', 'energy_j_per_token')]),
    ]:
        for row in rows:
            prediction = next(p for p in query(row['length']) if p.frequency_mhz == row['gpu_mhz'] and (not hasattr(p, 'group') or p.group == row['group']))
            for source, dest in fields:
                raw = row['metrics'][source]['raw']
                assert len(raw) == 5
                assert getattr(prediction, dest) == pytest.approx(sum(raw) / 5, rel=1e-12, abs=1e-12)
            if 'group' in row:
                metrics = row['metrics']
                for t, e, n, rt, re in zip(metrics['T_tail_s']['raw'], metrics['E_tail_j']['raw'], metrics['output_tokens']['raw'], metrics['tpot_actual_s']['raw'], metrics['epsilon_actual_j_per_token']['raw']):
                    assert rt == pytest.approx(t / (n - 1), rel=1e-12, abs=1e-12)
                    assert re == pytest.approx(e / (n - 1), rel=1e-12, abs=1e-12)


def test_reference_independent_raw_maxima_and_midpoints(bundle, measured):
    ref = read(bundle, 'fp16-reference-curves.json')
    assert len(ref['knots']) == 8 and len(ref['requests']) == 24
    for n in P:
        rows = [r for r in ref['requests'] if r['length'] == n]
        assert len(rows) == 3
        assert measured.prefill_target(n) == max(r['TTFT_s'] for r in rows)
        assert measured.decode_target(n) == max(r['TPOT_actual_s'] for r in rows)
    for lo, hi in zip(P, P[1:]):
        for query in [measured.prefill_target, measured.decode_target]:
            assert query((lo + hi) // 2) == 0.5 * query(lo) + 0.5 * query(hi)


@pytest.mark.parametrize('name', ['bundle-manifest.json', 'decode-lut.csv', 'decode-lut.json', 'decode-pwl.json', 'fp16-reference-curves.json', 'prefill-lut.csv', 'prefill-lut.json', 'prefill-pwl.json', 'quality-lut.json', 'source-export-manifest.json', 'source-validation.json'])
def test_every_sealed_file_tamper_and_missing_rejected(bundle, tmp_path, name):
    target = tmp_path / 'bundle'
    shutil.copytree(bundle, target)
    (target / name).write_bytes((target / name).read_bytes() + b' ')
    with pytest.raises(LUTError):
        MeasuredLUTRepository(target)
    (target / name).unlink()
    with pytest.raises(LUTError):
        MeasuredLUTRepository(target)


def test_manifest_traversal_and_unknown_pin_rejected(bundle, tmp_path):
    shutil.copytree(bundle, tmp_path / 'bundle')
    manifest = read(bundle, 'bundle-manifest.json')
    manifest['files_sha256']['../outside.json'] = '0' * 64
    (tmp_path / 'bundle/bundle-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(LUTError, match='manifest'):
        MeasuredLUTRepository(tmp_path / 'bundle')


def test_no_gpu_or_scientific_imports():
    code = "import sys; from kvolt_runtime.policy.lut import MeasuredLUTRepository; from kvolt_runtime.policy.mock import MockLUTRepository; MockLUTRepository().preflight(1024); assert not any(m in sys.modules for m in ('torch','transformers','numpy','scipy','kneed'))"
    result = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_mock_performance_exact_knots_nonmidpoint_and_no_result_cache():
    table = {(n, f): (0.10000000000000002 + n / 12345, n / 31 + f / 17) for n in P for f in F}
    repo = MockLUTRepository(prefill=table)
    for n in P:
        for p in repo.prefill(n):
            assert (p.ttft_s, p.energy_j) == table[n, p.frequency_mhz]
    n, lo, hi = 1337, 1024, 2048
    w = (n - lo) / (hi - lo)
    first = repo.prefill(n)
    for p in first:
        a, b = table[lo, p.frequency_mhz], table[hi, p.frequency_mhz]
        assert p.ttft_s == (1 - w) * a[0] + w * b[0]
        assert p.energy_j == (1 - w) * a[1] + w * b[1]
        assert p.bracket.weight == w
    second = repo.prefill(n)
    assert first == second and first is not second and first[0] is not second[0]


@pytest.mark.parametrize('family,table', [
    ('prefill', {(1024, 408): (1.0, 1.0)}),
    ('decode', {(1024, 8, 408): (1.0, 1.0)}),
    ('prefill_target', {1024: 1.0}),
    ('decode_target', {1024: 1.0}),
])
def test_missing_full_candidates_rejected(family, table):
    with pytest.raises(LUTError):
        MockLUTRepository(**{family: table})


@pytest.mark.parametrize('value', [None, True, '1', float('nan'), float('inf'), -1.0, 0.0])
@pytest.mark.parametrize('family', ['prefill', 'decode', 'prefill_target', 'decode_target'])
def test_invalid_time_or_target_rejected_upfront(family, value):
    if family == 'prefill':
        table = {(n, f): (1.0, 1.0) for n in P for f in F}
    elif family == 'decode':
        table = {(n, g, f): (1.0, 1.0) for n in P for g in GROUPS for f in F}
    else:
        table = {n: 1.0 for n in P}
    key = next(iter(table))
    table[key] = (value, 1.0) if family in ('prefill', 'decode') else value
    with pytest.raises(LUTError):
        MockLUTRepository(**{family: table})


@pytest.mark.parametrize('value', [None, True, '1', float('nan'), float('inf'), -1.0])
def test_invalid_energy_rejected_upfront(value):
    table = {(n, f): (1.0, 1.0) for n in P for f in F}
    table[1024, 408] = (1.0, value)
    with pytest.raises(LUTError):
        MockLUTRepository(prefill=table)


@pytest.mark.parametrize('key', [(1024.0, 408), (1024, 408.0), (1024, True), (1024, 409), (1024, 408, 8)])
def test_no_candidate_interpolation_or_identifier_coercion(key):
    table = {(n, f): (1.0, 1.0) for n in P for f in F}
    del table[1024, 408]
    table[key] = (1.0, 1.0)
    with pytest.raises(LUTError):
        MockLUTRepository(prefill=table)


def test_unknown_mock_scenario():
    with pytest.raises(LUTError):
        MockLUTRepository('unknown')


def test_symlink_escape_rejected_even_when_hash_matches(bundle, tmp_path):
    target = tmp_path / 'bundle'
    shutil.copytree(bundle, target)
    name = 'prefill-pwl.json'
    (target / name).unlink()
    (target / name).symlink_to(bundle / name)
    with pytest.raises(LUTError, match='escapes'):
        MeasuredLUTRepository(target)


def test_json_rejects_duplicate_keys_and_nonfinite():
    from kvolt_runtime.policy.lut import _json
    for data in [b'{"length":1024,"length":2048}', b'{"value":NaN}', b'{"value":Infinity}', b'{"value":1e999}', b'{']:
        with pytest.raises(LUTError):
            _json(data, 'fixture')
