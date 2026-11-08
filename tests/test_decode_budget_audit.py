"""Sparse Decode evidence is checked against an independent full-grid source."""
from dataclasses import asdict

import pytest

from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.qualification.audit import audit_policy


def result_with_decode(*, fallback=False, ties=False):
    repo = MockLUTRepository('exact_ties' if ties else 'baseline')
    length, nmax, t1 = 1024, 2, 2_000_000_000
    elapsed = 500_000_000 if fallback else 5_000_000
    target = repo.decode_target(length)
    budget = target - elapsed / 1e9
    quality = [asdict(row) for row in repo.quality(length)]
    scores = {row['group']: row['score'] for row in quality}
    full = [asdict(row) for row in repo.decode(length)]
    active = [row for row in full if row['group'] == 32 and row['tpot_s'] <= budget]
    fastest = min(row['tpot_s'] for row in full)
    candidates = active or [row for row in full if row['tpot_s'] == fastest]
    key = ((lambda row: (row['energy_j_per_token'], -scores[row['group']], row['tpot_s'],
                         row['group'], row['frequency_mhz'])) if active else
           (lambda row: (row['tpot_s'], -scores[row['group']], row['energy_j_per_token'],
                         row['group'], row['frequency_mhz'])))
    selected = min(candidates, key=key)
    records = []
    for row in candidates:
        is_active, meets = row['group'] == 32, row['tpot_s'] <= budget
        chosen = row is selected
        reasons = ([] if is_active else ['knee_excluded']) + ([] if meets else ['time_budget_exceeded'])
        reasons += ([] if chosen else ['not_selected'])
        if chosen and not active:
            reasons.append('latency_fallback')
        records.append(dict(prediction=row, quality=scores[row['group']], active_knee=is_active,
                            meets_budget=meets, eligible=is_active and meets, selected=chosen,
                            reasons=reasons))
    dc = dict(status='selected', selection_rule='budget_filtered_decode', candidates=records,
              selected=selected, selected_quality=scores[selected['group']], t1_ns=t1,
              td_ns=t1+elapsed, elapsed_ns=elapsed, target_s=target, budget_s=budget,
              selection_path='energy_min' if active else 'latency_fallback',
              predicted_feasible=selected['tpot_s'] <= budget, quality=quality,
              knee=dict(status='not_found', groups=[], reason='flat_quality' if ties else 'no_knee'))
    prefill = [asdict(row) for row in repo.prefill(length)]
    pc_budget = repo.prefill_target(length) - .001
    pc_selected = min((row for row in prefill if row['ttft_s'] <= pc_budget),
                      key=lambda row: (row['energy_j'], row['ttft_s'], row['frequency_mhz']))
    pc = dict(candidates=[dict(prediction=row) for row in prefill], selected=pc_selected,
              target_s=repo.prefill_target(length), budget_s=pc_budget, tp_ns=1_000_000,
              elapsed_ns=1_000_000, selection_path='energy_min')

    def receipt(row, start):
        return dict(verified=True, start_ns=start, end_ns=start+1,
                    observed=dict.fromkeys(('min_hz', 'max_hz', 'cur_hz'), row['frequency_mhz']*1_000_000))

    result = dict(request=dict(length=length, nmax=nmax), timing=dict(t0_ns=0, t1_ns=t1),
                  generation=dict(token_ids=[17, 23], token_ready_ns=[t1, 3_000_000_000]),
                  decisions=dict(prefill=pc, decode=dc),
                  device=dict(prefill=receipt(pc_selected, 1_000_001),
                              decode=receipt(selected, t1+elapsed+1)))
    return repo, result


@pytest.mark.parametrize('fallback,ties', [(False, False), (True, False), (True, True)])
def test_filtered_decode_audit_accepts_complete_source_evidence(fallback, ties):
    repo, result = result_with_decode(fallback=fallback, ties=ties)
    audit_policy(result, repo)


@pytest.mark.parametrize('fallback', [False, True])
def test_filtered_decode_audit_rejects_omitted_eligible_or_fastest_ties(fallback):
    repo, result = result_with_decode(fallback=fallback, ties=True)
    result['decisions']['decode']['candidates'].pop()
    with pytest.raises(ValueError):
        audit_policy(result, repo)


@pytest.mark.parametrize('mutation', ['duplicate', 'energy', 'target', 'bracket', 'quality',
                                     'active_flag', 'selected_flag', 'reasons', 'selected', 'rule'])
def test_filtered_decode_audit_rejects_changed_source_or_selection_evidence(mutation):
    repo, result = result_with_decode()
    dc = result['decisions']['decode']
    if mutation == 'duplicate':
        dc['candidates'].append(dc['candidates'][-1])
    elif mutation == 'energy':
        dc['candidates'][-1]['prediction']['energy_j_per_token'] += 1
    elif mutation == 'target':
        dc['target_s'] += 1
        dc['budget_s'] += 1
    elif mutation == 'bracket':
        dc['candidates'][-1]['prediction']['bracket']['weight'] = .5
    elif mutation == 'quality':
        dc['quality'][0]['score'] = .1
    elif mutation == 'active_flag':
        dc['candidates'][0]['active_knee'] = False
    elif mutation == 'selected_flag':
        dc['candidates'][-1]['selected'] = True
    elif mutation == 'reasons':
        dc['candidates'][-1]['reasons'] = []
    elif mutation == 'selected':
        dc['selected'] = dc['candidates'][-1]['prediction']
    else:
        dc['selection_rule'] = 'unknown'
    with pytest.raises(ValueError):
        audit_policy(result, repo)
