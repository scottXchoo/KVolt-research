"""Target-filtered Prefill queries preserve raw values and avoid wasted details."""
from collections import Counter
import math
from pathlib import Path
import random

import pytest

from kvolt_runtime.errors import ContractError
from kvolt_runtime.policy import lut
from kvolt_runtime.policy.compact import load_lut_repository
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.types import FREQUENCIES_MHZ, PERFORMANCE_LENGTHS


ROOT = Path(__file__).resolve().parents[1]


def repository(times):
    return MockLUTRepository(prefill={
        (n, f): (time, 1000.0 + f)
        for n in PERFORMANCE_LENGTHS
        for f, time in zip(FREQUENCIES_MHZ, times)
    })


def expected_rows(repo, length, target):
    rows = repo.prefill(length)
    eligible = tuple(row for row in rows if row.ttft_s <= target)
    fastest = min(row.ttft_s for row in rows)
    return eligible or tuple(row for row in rows if row.ttft_s == fastest)


@pytest.mark.parametrize('times', [
    (8, 7, 6, 5, 4, 3, 2, 1),
    (1, 5, 3, 2, 7, 4, 6, 1),
    (1,) * 8,
])
@pytest.mark.parametrize('target', [0.1, 1.0, 1.5, 3.0, 8.0, 9.0])
def test_filtered_rows_match_independent_full_grid(times, target):
    repo = repository(times)
    expected = expected_rows(repo, 1537, target)
    result = repo.prefill_within_target(1537, target)
    assert result.predictions == expected
    assert result.length == 1537 and result.target_s == target
    assert all(row.bracket is result.bracket for row in result.predictions)


def test_monotone_search_checks_boundary_only_and_reuses_ttft(monkeypatch):
    repo = repository((8, 7, 6, 5, 4, 3, 2, 1))
    original = lut._at
    calls = []

    def observed(lo, hi, bracket):
        calls.append((lo, hi))
        return original(lo, hi, bracket)

    def forbidden(*args, **kwargs):
        pytest.fail('filtered query used a full-grid or separate lookup')

    monkeypatch.setattr(lut, '_at', observed)
    for name in ('prefill', 'prefill_lookup', 'prefill_latencies',
                 'prefill_candidates', 'prefill_target'):
        monkeypatch.setattr(repo, name, forbidden)
    result = repo.prefill_within_target(1537, 1.5)
    assert [row.frequency_mhz for row in result.predictions] == [1122]
    time_calls = [pair for pair in calls if pair[0] < 1000]
    energy_calls = [pair for pair in calls if pair[0] >= 1000]
    assert 1 <= len(time_calls) < 8
    assert all(count == 1 for count in Counter(time_calls).values())
    assert energy_calls == [(2122.0, 2122.0)]


@pytest.mark.parametrize('target', [0.1, 1.0, 3.0, 9.0])
def test_energy_is_interpolated_only_for_returned_rows(monkeypatch, target):
    repo = repository((8, 7, 6, 5, 4, 3, 1, 1))
    expected = expected_rows(repo, 1537, target)
    original = lut._at
    energies = []

    def observed(lo, hi, bracket):
        if lo >= 1000:
            energies.append(lo)
        return original(lo, hi, bracket)

    monkeypatch.setattr(lut, '_at', observed)
    result = repo.prefill_within_target(1537, target)
    assert result.predictions == expected
    assert sorted(energies) == sorted(row.energy_j for row in expected)


def test_crossing_length_curves_are_not_assumed_monotone():
    repo = MockLUTRepository(prefill={
        (n, f): ((index + 1 if n == 1024 else 8 - index), 1000.0 + f)
        for n in PERFORMANCE_LENGTHS
        for index, f in enumerate(FREQUENCIES_MHZ)
    })
    for length in (1024, 1377, 1536, 1777, 2048):
        for target in (1.0, 3.0, 4.5, 6.0, 9.0):
            assert repo.prefill_within_target(length, target).predictions == expected_rows(repo, length, target)


def test_binary64_threshold_neighbors_preserve_exact_membership():
    randomizer = random.Random(20261108)
    for monotone in (True, False):
        table = {}
        for n in PERFORMANCE_LENGTHS:
            times = [randomizer.uniform(0.1, 9.0) for _ in FREQUENCIES_MHZ]
            if monotone:
                times.sort(reverse=True)
            for f, time in zip(FREQUENCIES_MHZ, times):
                table[(n, f)] = (time, randomizer.uniform(0.0, 10.0))
        repo = MockLUTRepository(prefill=table)
        for length in (1024, 1377, 4609, 8192):
            for row in repo.prefill(length):
                for target in (math.nextafter(row.ttft_s, 0.0), row.ttft_s,
                               math.nextafter(row.ttft_s, math.inf)):
                    assert repo.prefill_within_target(length, target).predictions == expected_rows(repo, length, target)


@pytest.mark.parametrize('bad', [True, False, None, '1', 0, -1, float('nan'), float('inf')])
def test_invalid_target_is_rejected(bad):
    with pytest.raises(ContractError):
        MockLUTRepository().prefill_within_target(1024, bad)


@pytest.mark.parametrize('bad', [True, False, None, '1024', 1024.0, 1023, 8193])
def test_invalid_or_unsupported_length_is_rejected(bad):
    with pytest.raises(ContractError):
        MockLUTRepository().prefill_within_target(bad, 1.0)


def test_latest_runtime_and_reference_match_full_query_at_knots_and_between():
    repo = load_lut_repository(ROOT / 'data/luts/agx2-runtime-repeat5',
                               reference_lut_dir=ROOT / 'data/luts/agx2-reference-repeat10')
    for length in (*PERFORMANCE_LENGTHS, *range(1025, 8192, 257), 8191):
        target = repo.prefill_target(length)
        assert repo.prefill_within_target(length, target).predictions == expected_rows(repo, length, target)
