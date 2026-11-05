"""Frozen binary64 policy inputs; reject coercion before selection."""
from dataclasses import FrozenInstanceError
import math

import pytest

from kvolt_runtime.errors import ContractError
from kvolt_runtime.policy.types import (
    Bracket, DecodePrediction, KneeResult, PrefillPrediction, QualityPrediction,
    finite, frequency, group, length,
)


@pytest.mark.parametrize('value', [None, True, False, '1', complex(1), float('nan'), float('inf'), -float('inf'), -1, 10**1000])
def test_finite_rejects_invalid_numbers(value):
    with pytest.raises(ContractError):
        finite(value, 'input')


def test_binary64_conversion_and_ranges():
    assert type(finite(1, 'input')) is float
    exact = math.nextafter(0.5, 1.0)
    assert finite(exact, 'input') == exact
    assert finite(0, 'energy') == 0.0
    assert finite(1, 'raw quality', unit_interval=True) == 1.0
    with pytest.raises(ContractError):
        finite(0, 'time', positive=True)
    with pytest.raises(ContractError):
        finite(1.00001, 'quality', unit_interval=True)


@pytest.mark.parametrize('validator,value', [(length, True), (length, 1024.0), (length, 1023), (length, 8193), (frequency, 409), (frequency, 408.0), (frequency, True), (group, 12), (group, 8.0), (group, True)])
def test_candidate_identifiers_are_strict(validator, value):
    with pytest.raises(ContractError):
        validator(value)


def test_predictions_frozen_and_full_precision():
    b = Bracket(1024, 2048, 0.5)
    x = math.nextafter(0.1, 1)
    p = PrefillPrediction(1536, 408, x, 0.0, b)
    d = DecodePrediction(1536, 8, 408, x, 0.0, b)
    q = QualityPrediction(1536, 8, x, b)
    assert p.ttft_s == d.tpot_s == q.score == x
    for record in [p, d, q, b]:
        with pytest.raises(FrozenInstanceError):
            record.length = 2048


@pytest.mark.parametrize('bracket', [Bracket(1024, 2048, 0.25), Bracket(2048, 3072, 0.5)])
def test_prediction_bracket_must_contain_correct_weight(bracket):
    with pytest.raises(ContractError):
        QualityPrediction(1536, 8, 0.5, bracket)


@pytest.mark.parametrize('args', [(2048, 1024, 0.0), (1024, 1024, 0.5), (1024, 2048, True), (1024, 2048, -0.1), (1024, 2048, 1.1)])
def test_invalid_brackets(args):
    with pytest.raises(ContractError):
        Bracket(*args)


@pytest.mark.parametrize('status,groups', [('unknown', ()), ('found', ()), ('not_found', (8,)), ('error_recovered', (8,)), ('found', (16, 8)), ('found', (8, 8)), ('found', (12,)), ('found', [8])])
def test_invalid_knee_results(status, groups):
    with pytest.raises(ContractError):
        KneeResult(status, groups)


@pytest.mark.parametrize('factory,kwargs', [
    (PrefillPrediction, {'frequency_mhz': True}),
    (PrefillPrediction, {'frequency_mhz': '408'}),
    (PrefillPrediction, {'ttft_s': 0}),
    (PrefillPrediction, {'ttft_s': float('inf')}),
    (PrefillPrediction, {'ttft_s': '1'}),
    (PrefillPrediction, {'energy_j': -1}),
    (PrefillPrediction, {'energy_j': True}),
    (DecodePrediction, {'group': True}),
    (DecodePrediction, {'frequency_mhz': 409}),
    (DecodePrediction, {'tpot_s': 0}),
    (DecodePrediction, {'tpot_s': float('nan')}),
    (DecodePrediction, {'energy_j_per_token': -1}),
    (DecodePrediction, {'energy_j_per_token': '1'}),
    (QualityPrediction, {'group': 12}),
    (QualityPrediction, {'score': True}),
    (QualityPrediction, {'score': 1.1}),
    (QualityPrediction, {'score': float('nan')}),
])
def test_each_prediction_validates_its_fields(factory, kwargs):
    inputs = {
        PrefillPrediction: {'length': 1024, 'frequency_mhz': 408, 'ttft_s': 1.0, 'energy_j': 1.0},
        DecodePrediction: {'length': 1024, 'group': 8, 'frequency_mhz': 408, 'tpot_s': 1.0, 'energy_j_per_token': 1.0},
        QualityPrediction: {'length': 1024, 'group': 8, 'score': 0.9},
    }[factory]
    with pytest.raises(ContractError):
        factory(**{**inputs, 'bracket': Bracket(1024, 1024, 0.0), **kwargs})


@pytest.mark.parametrize('factory', [PrefillPrediction, DecodePrediction, QualityPrediction])
@pytest.mark.parametrize('bad', [True, '1024', 1024.0, 1023, 8193])
def test_each_prediction_validates_length(factory, bad):
    inputs = {
        PrefillPrediction: {'frequency_mhz': 408, 'ttft_s': 1.0, 'energy_j': 1.0},
        DecodePrediction: {'group': 8, 'frequency_mhz': 408, 'tpot_s': 1.0, 'energy_j_per_token': 1.0},
        QualityPrediction: {'group': 8, 'score': 0.9},
    }[factory]
    with pytest.raises(ContractError):
        factory(length=bad, bracket=Bracket(1024, 1024, 0.0), **inputs)


@pytest.mark.parametrize('start,now', [(True, 10), (0, True), ('0', 10), (0, '10'), (0.0, 10), (0, 10.0), (-1, 10), (10, 9), (0, float('inf'))])
def test_elapsed_clock_is_strict_and_monotonic(start, now):
    from kvolt_runtime.policy.types import elapsed_since
    with pytest.raises(ContractError):
        elapsed_since(start, now)


def test_elapsed_subtracts_large_integer_nanoseconds_exactly():
    from kvolt_runtime.policy.types import elapsed_since
    start = 10**100
    assert elapsed_since(start, start + 17) == 17
    assert type(elapsed_since(start, start)) is int
