"""Decode budget queries match a full-grid oracle without reading unused energy."""
import math
from pathlib import Path

import pytest

from kvolt_runtime.config import GROUPS
from kvolt_runtime.errors import ContractError
from kvolt_runtime.policy import lut
from kvolt_runtime.policy.compact import load_lut_repository
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.types import FREQUENCIES_MHZ, PERFORMANCE_LENGTHS


ROOT = Path(__file__).resolve().parents[1]


def repository(times):
    return MockLUTRepository(decode={
        (length, group, frequency): (time + index / 10, 1000 + group + frequency)
        for length in PERFORMANCE_LENGTHS
        for index, group in enumerate(GROUPS)
        for frequency, time in zip(FREQUENCIES_MHZ, times)
    })


def expected_rows(repo, length, groups, budget):
    rows = repo.decode(length)
    eligible = tuple(row for row in rows if row.group in groups and row.tpot_s <= budget)
    fastest = min(row.tpot_s for row in rows)
    return eligible or tuple(row for row in rows if row.tpot_s == fastest)


@pytest.mark.parametrize('times', [(8, 7, 6, 5, 4, 3, 2, 1),
                                   (1, 5, 3, 2, 7, 4, 6, 1), (1,) * 8])
@pytest.mark.parametrize('groups', [(32,), (8, 16), GROUPS])
@pytest.mark.parametrize('budget', [-1.5, 0.0, 1.0, 1.2, 4.5, 9.0])
def test_budget_query_matches_full_grid_oracle(times, groups, budget):
    repo = repository(times)
    expected = expected_rows(repo, 1537, groups, budget)
    result = repo.decode_within_budget(1537, groups, budget)
    assert type(result) is tuple and result == expected
    assert all(row.bracket is result[0].bracket for row in result)


@pytest.mark.parametrize('fallback', [False, True])
def test_only_returned_energy_is_read_and_each_time_is_interpolated_once(monkeypatch, fallback):
    repo = MockLUTRepository(decode={
        (length, group, frequency): (1 + index * 8 + offset, 1000 + index * 8 + offset)
        for length in PERFORMANCE_LENGTHS
        for index, group in enumerate(GROUPS)
        for offset, frequency in enumerate(FREQUENCIES_MHZ)
    })
    budget = -1 if fallback else 20
    expected = expected_rows(repo, 1537, (32,), budget)
    original = lut._at
    times, energies = [], []

    def observed(lower, upper, bracket):
        (energies if lower >= 1000 else times).append(lower)
        return original(lower, upper, bracket)

    def forbidden(*args, **kwargs):
        pytest.fail('conditional query used a full lookup, Quality or Reference query')

    monkeypatch.setattr(lut, '_at', observed)
    for name in ('decode', 'decode_lookup', 'decode_target', 'quality'):
        monkeypatch.setattr(repo, name, forbidden, raising=False)
    result = repo.decode_within_budget(1537, (32,), budget)
    assert result == expected
    assert sorted(energies) == sorted(row.energy_j_per_token for row in expected)
    assert len(times) == len(set(times)) == (40 if fallback else 8)
    if not fallback:
        assert sorted(times) == list(range(17, 25))


def test_fallback_preserves_all_global_fastest_ties_outside_active_group():
    table = {(length, group, frequency): (5.0, 10.0)
             for length in PERFORMANCE_LENGTHS for group in GROUPS for frequency in FREQUENCIES_MHZ}
    for length in PERFORMANCE_LENGTHS:
        table[(length, 8, 408)] = (0.25, 100)
        table[(length, 128, 1122)] = (0.25, 1)
    repo = MockLUTRepository(decode=table)
    result = repo.decode_within_budget(1537, (32,), 1.0)
    assert [(row.group, row.frequency_mhz) for row in result] == [(8, 408), (128, 1122)]


def test_budget_boundary_uses_exact_binary64_comparisons():
    repo = repository((8, 7, 6, 5, 4, 3, 2, 1))
    for row in repo.decode(1537):
        for budget in (math.nextafter(row.tpot_s, 0.0), row.tpot_s,
                       math.nextafter(row.tpot_s, math.inf)):
            assert repo.decode_within_budget(1537, (row.group,), budget) == expected_rows(
                repo, 1537, (row.group,), budget)


def test_latest_runtime_reference_and_intermediate_lengths_match_full_query():
    repo = load_lut_repository(ROOT / 'data/luts/agx2-runtime-repeat5',
                               reference_lut_dir=ROOT / 'data/luts/agx2-reference-repeat10')
    for length in (*PERFORMANCE_LENGTHS, 1025, 1377, 4609, 8191):
        for groups in ((32,), (8, 64), GROUPS):
            for budget in (-0.01, repo.decode_target(length)):
                assert repo.decode_within_budget(length, groups, budget) == expected_rows(
                    repo, length, groups, budget)


@pytest.mark.parametrize('bad', [True, False, None, '1', float('nan'), float('inf'), -float('inf'), 10**400])
def test_invalid_budget_is_rejected(bad):
    with pytest.raises(ContractError):
        MockLUTRepository().decode_within_budget(1024, (32,), bad)


@pytest.mark.parametrize('bad', [(), (8, 8), (12,), (True,), [32], None, (8.0,)])
def test_invalid_group_candidates_are_rejected(bad):
    with pytest.raises(ContractError):
        MockLUTRepository().decode_within_budget(1024, bad, 1.0)


@pytest.mark.parametrize('bad', [True, False, None, '1024', 1024.0, 1023, 8193])
def test_invalid_or_unsupported_length_is_rejected(bad):
    with pytest.raises(ContractError):
        MockLUTRepository().decode_within_budget(bad, (32,), 1.0)
