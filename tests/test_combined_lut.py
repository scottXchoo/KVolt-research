"""Phase lookups return predictions and the frozen FP16 goal together."""
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path

import pytest

from kvolt_runtime.errors import ContractError
from kvolt_runtime.policy import lut
from kvolt_runtime.policy.compact import load_lut_repository
from kvolt_runtime.policy.lut import MeasuredLUTRepository
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.types import PERFORMANCE_LENGTHS, PrefillLatencyPrediction


ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (*PERFORMANCE_LENGTHS, *range(1536, 8192, 1024), 1025, 4609, 8191)


@pytest.fixture(params=['mock', 'sealed', 'agx1', 'agx2-repeat3', 'agx2-repeat5'])
def repository(request):
    if request.param == 'mock':
        return MockLUTRepository()
    if request.param == 'sealed':
        path = Path(os.environ.get('KVOLT_LUT_BUNDLE',
                    ROOT.parent / 'KVolt-lut-source/data/luts/agx2-repeat5-20261103'))
        if not path.is_dir():
            pytest.skip('sealed measured integration unavailable: set KVOLT_LUT_BUNDLE')
        return MeasuredLUTRepository(path)
    runtime = 'latest' if request.param == 'agx1' else 'agx2-runtime-repeat5'
    reference = 'agx2-reference-repeat3' if request.param == 'agx2-repeat3' else 'agx2-reference-repeat5'
    return load_lut_repository(ROOT / 'data/luts' / runtime,
                               reference_lut_dir=ROOT / 'data/luts' / reference)


def test_combined_values_preserve_existing_exact_and_interpolated_results(repository):
    for length in LENGTHS:
        for latency_only in (False, True):
            result = repository.prefill_lookup(length, latency_only=latency_only)
            expected = (repository.prefill_latencies(length) if latency_only
                        else repository.prefill(length))
            assert result.length == length
            assert result.predictions == expected
            assert result.target_s == repository.prefill_target(length)
            assert all(row.bracket is result.bracket for row in result.predictions)
        result = repository.decode_lookup(length)
        assert result.length == length
        assert result.predictions == repository.decode(length)
        assert result.target_s == repository.decode_target(length)
        assert all(row.bracket is result.bracket for row in result.predictions)


@pytest.mark.parametrize('method,options', [('prefill_lookup', {}),
                                          ('prefill_lookup', {'latency_only': True}),
                                          ('decode_lookup', {})])
def test_single_bracket_and_no_separate_prediction_or_target_calls(monkeypatch, method, options):
    repo = MockLUTRepository()
    brackets = []
    original = lut._bracket

    def observed(length, grid):
        brackets.append((length, grid))
        return original(length, grid)

    def forbidden(*args, **kwargs):
        pytest.fail('combined lookup called a separate online lookup')

    monkeypatch.setattr(lut, '_bracket', observed)
    for name in ('prefill', 'prefill_latencies', 'prefill_candidates', 'prefill_target',
                 'decode', 'decode_target'):
        monkeypatch.setattr(repo, name, forbidden)
    result = getattr(repo, method)(1537, **options)
    assert brackets == [(1537, PERFORMANCE_LENGTHS)]
    assert result.bracket.lower == 1024 and result.bracket.upper == 2048
    assert result.bracket.weight == (1537 - 1024) / 1024


def test_reference_targets_come_from_original_reference_not_candidate_latencies():
    repo = load_lut_repository(ROOT / 'data/luts/agx2-runtime-repeat5',
                               reference_lut_dir=ROOT / 'data/luts/agx2-reference-repeat5')
    rows = json.loads((ROOT / 'data/luts/agx2-reference-repeat5/reference.json').read_bytes())
    for index, row in enumerate(rows):
        length, prefill_target, decode_target = row
        assert repo.prefill_lookup(length).target_s == prefill_target
        assert repo.decode_lookup(length).target_s == decode_target
        if index + 1 < len(rows):
            upper = rows[index + 1]
            middle = (length + upper[0]) // 2
            assert repo.prefill_lookup(middle).target_s == .5 * prefill_target + .5 * upper[1]
            assert repo.decode_lookup(middle).target_s == .5 * decode_target + .5 * upper[2]


def test_screening_lookup_never_interpolates_energy(monkeypatch):
    frequencies = lut.FREQUENCIES_MHZ
    table = {(n, f): (n / 1024 + f / 1000, 1000.0 + f)
             for n in PERFORMANCE_LENGTHS for f in frequencies}
    repo = MockLUTRepository(prefill=table)
    original = lut._at

    def no_energy(lo, hi, bracket):
        assert lo < 1000 and hi < 1000, 'screening interpolated an energy cell'
        return original(lo, hi, bracket)

    monkeypatch.setattr(lut, '_at', no_energy)
    result = repo.prefill_lookup(1536, latency_only=True)
    assert len(result.predictions) == 8
    assert all(isinstance(row, PrefillLatencyPrediction) and row.energy_j is None
               for row in result.predictions)


@pytest.mark.parametrize('method', ['prefill_lookup', 'decode_lookup'])
@pytest.mark.parametrize('length', [True, False, None, '1024', 1024.0, 1023, 8193])
def test_combined_lookup_rejects_invalid_or_unsupported_length(method, length):
    with pytest.raises(ContractError):
        getattr(MockLUTRepository(), method)(length)


@pytest.mark.parametrize('latency_only', [None, 0, 1, 'true', ()])
def test_screening_mode_requires_boolean(latency_only):
    with pytest.raises(ContractError):
        MockLUTRepository().prefill_lookup(1024, latency_only=latency_only)


@pytest.mark.parametrize('method', ['prefill_lookup', 'decode_lookup'])
def test_combined_result_is_immutable(method):
    result = getattr(MockLUTRepository(), method)(1536)
    assert type(result.predictions) is tuple
    with pytest.raises(FrozenInstanceError):
        result.target_s = 1.0
