"""Signed Prefill budgets filter exact predictions without querying other energy."""
from collections import Counter
import math

import pytest

from kvolt_runtime.errors import ContractError
from kvolt_runtime.policy import lut
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.types import FREQUENCIES_MHZ, PERFORMANCE_LENGTHS, finite


def repository(times):
    return MockLUTRepository(prefill={
        (n, f): (time, 1000.0 + f)
        for n in PERFORMANCE_LENGTHS
        for f, time in zip(FREQUENCIES_MHZ, times)
    })


@pytest.mark.parametrize('times', [(8, 7, 6, 5, 4, 3, 1, 1),
                                  (1, 5, 3, 2, 7, 4, 6, 1), (1,) * 8])
@pytest.mark.parametrize('budget', [-2.0, 0.0, 0.1, 1.0, 1.5, 3.0, 8.0, 9.0])
def test_signed_budget_matches_full_grid_or_fastest_ties(times, budget):
    repo = repository(times)
    full = repo.prefill(1537)
    passing = tuple(row for row in full if row.ttft_s <= budget)
    fastest = min(row.ttft_s for row in full)
    expected = passing or tuple(row for row in full if row.ttft_s == fastest)
    result = repo.prefill_within_budget(1537, budget)
    assert result.predictions == expected
    assert result.budget_s == budget and result.length == 1537
    assert all(row.bracket is result.bracket for row in result.predictions)


@pytest.mark.parametrize('budget', [-2.0, 0.0, 1.5, 3.0, 9.0])
def test_budget_query_only_interpolates_returned_energy_and_reuses_ttft(monkeypatch, budget):
    repo = repository((8, 7, 6, 5, 4, 3, 2, 1))
    calls = []
    original = lut._at

    def observed(lo, hi, bracket):
        calls.append((lo, hi))
        return original(lo, hi, bracket)

    def forbidden(*args, **kwargs):
        pytest.fail('budget query used a separate target/full-grid API')

    for name in ('prefill', 'prefill_lookup', 'prefill_target', 'prefill_within_target',
                 'prefill_latencies', 'prefill_candidates'):
        monkeypatch.setattr(repo, name, forbidden)
    monkeypatch.setattr(lut, '_at', observed)
    result = repo.prefill_within_budget(1537, budget)
    times = [pair for pair in calls if pair[0] < 1000]
    energies = [pair[0] for pair in calls if pair[0] >= 1000]
    assert all(count == 1 for count in Counter(times).values())
    assert sorted(energies) == sorted(row.energy_j for row in result.predictions)
    if budget <= 1.5:
        assert len(times) < 8


def test_budget_binary64_neighbors_include_equality_only():
    repo = repository((8, 7, 6, 5, 4, 3, 2, 1))
    for budget, expected in ((math.nextafter(2.0, 0), [1122]),
                             (2.0, [1020, 1122]),
                             (math.nextafter(2.0, math.inf), [1020, 1122])):
        result = repo.prefill_within_budget(1024, budget)
        assert [row.frequency_mhz for row in result.predictions] == expected


@pytest.mark.parametrize('bad', [True, False, None, '1', float('nan'), float('inf'), -float('inf')])
def test_nonfinite_or_nonnumeric_budget_rejected(bad):
    with pytest.raises(ContractError):
        MockLUTRepository().prefill_within_budget(1024, bad)


@pytest.mark.parametrize('bad', [True, False, None, '1024', 1024.0, 1023, 8193])
def test_unsupported_length_rejected(bad):
    with pytest.raises(ContractError):
        MockLUTRepository().prefill_within_budget(bad, -1.0)


def test_signed_budget_support_does_not_allow_negative_metric_values():
    with pytest.raises(ContractError):
        finite(-1.0, 'energy')
    with pytest.raises(ContractError):
        finite(-1.0, 'target', positive=True)
    with pytest.raises(ContractError):
        finite(-1.0, 'target', positive=True, allow_negative=True)
    with pytest.raises(ContractError):
        finite(-0.1, 'score', unit_interval=True, allow_negative=True)
    with pytest.raises(ContractError):
        MockLUTRepository().prefill_within_target(1024, -1.0)
