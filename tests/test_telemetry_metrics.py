import math
import pytest
from kvolt_runtime.telemetry import RAILS, energy_metrics, timing_metrics


def samples():
    # P(t)=2+t seconds on each rail. Integral 1..3 is 8 J per rail.
    return [dict(rail=rail, timestamp_ns=t*10**9, power_w=2+t)
            for rail in RAILS for t in (0, 2, 4)]


def test_linear_boundary_integration_and_normalization():
    result = energy_metrics(samples(), 10**9, 2*10**9, 3*10**9, 3,
                            completed=True, max_gap_ns=2*10**9)
    assert result['valid']
    assert result['total_energy_j'] == pytest.approx(24)
    assert result['prefill_energy_j'] == pytest.approx(10.5)
    assert result['decode_energy_j'] == pytest.approx(13.5)
    assert result['decode_j_per_token'] == pytest.approx(6.75)
    assert result['rails'][RAILS[0]]['total_energy_j'] == pytest.approx(8)


def test_timing_integer_ns_n1_and_incomplete_prefix():
    result = timing_metrics(10**18, [10**18+10**9, 10**18+3*10**9], completed=True)
    assert result['ttft_s'] == 1
    assert result['mean_tpot_s'] == 2
    single = timing_metrics(0, [10], completed=True)
    assert single['mean_tpot_s'] is None
    assert single['decode_reason'] == 'no_decode_tokens'
    prefix = timing_metrics(0, [10, 20], completed=False)
    assert not prefix['valid'] and prefix['ttft_s'] is None
    assert prefix['prefix']['ttft_s'] == 1e-8
    assert not timing_metrics(0, [], completed=False)['valid']


@pytest.mark.parametrize('mutation,reason', [
    (lambda rows: rows.pop(0), 'missing_left_bracket'),
    (lambda rows: rows.__setitem__(1, dict(rows[1], power_w=math.nan)), 'invalid_sample'),
    (lambda rows: rows.__setitem__(1, dict(rows[1], error='EIO')), 'invalid_sample'),
    (lambda rows: rows.__setitem__(1, dict(rows[1], timestamp_ns=0)), 'nonincreasing_sample_time'),
])
def test_invalid_trace_preserved(mutation, reason):
    rows=samples(); mutation(rows)
    result=energy_metrics(rows, 10**9, 2*10**9, 3*10**9, 3,
                          completed=True, max_gap_ns=2*10**9)
    assert not result['valid'] and result['total_energy_j'] is None
    assert reason in result['reason']


def test_gap_no_collection_n1_incomplete():
    assert energy_metrics(None, 0, 1, 2, 2, completed=True, max_gap_ns=1)['reason']=='not_collected'
    bad=energy_metrics(samples(), 10**9, 2*10**9, 3*10**9, 3,
                       completed=True,max_gap_ns=10**9)
    assert 'sample_gap_exceeded' in bad['reason']
    single=energy_metrics(samples(),10**9,2*10**9,2*10**9,1,
                          completed=True,max_gap_ns=2*10**9)
    assert single['valid'] and single['decode_energy_j'] is None
    prefix=energy_metrics(samples(),10**9,2*10**9,3*10**9,3,
                          completed=False,max_gap_ns=2*10**9)
    assert prefix['total_energy_j'] is None
    assert prefix['prefix']['total_energy_j']==pytest.approx(24)
