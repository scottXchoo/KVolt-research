"""Request observations adapt exact cells without changing sealed policy data."""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
import stat

import pytest

from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.online import OnlineLUTRepository
from kvolt_runtime.policy.types import LUTError


def observation(length=1536, n=3):
    return {
        'request': {'length': length},
        'generation': {'status': 'completed', 'n_output_tokens': n,
                       'token_ids': list(range(n))},
        'timing': {'valid': True, 'ttft_s': 2.0,
                   'mean_tpot_s': 0.5 if n > 1 else None,
                   'n_output_tokens': n, 'n_decode_tokens': n-1,
                   'decode_duration_s': 0.5*(n-1) if n > 1 else None},
        'energy': {'valid': True, 'prefill_energy_j': 4.0,
                   'decode_energy_j': 0.8*(n-1) if n > 1 else None,
                   'decode_j_per_token': 0.8 if n > 1 else None},
        'decisions': {'prefill': {'selected': {'length': length, 'frequency_mhz': 408}},
                      'decode': {'status': 'selected' if n > 1 else 'skipped',
                                 'selected': {'length': length, 'group': 32, 'frequency_mhz': 510}
                                 if n > 1 else None}},
        'device': {'prefill': {'verified': True}, 'decode': {'verified': True}},
        'slo': {'ttft_met': False, 'mean_tpot_met': False},
        'cleanup': {'status': 'failed'}, 'errors': [{'stage': 'restore'}],
    }


def selected_decode(repo, length=1536):
    return next(p for p in repo.decode(length) if p.group == 32 and p.frequency_mhz == 510)


def checksum(state):
    payload = {k: v for k, v in state.items() if k != 'state_sha256'}
    state['state_sha256'] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return state


def test_ema_exact_length_only_and_frozen_sources(tmp_path):
    base = MockLUTRepository()
    original = (base.prefill(1536), base.decode(1536), base.quality(1536), base.prefill_target(1536))
    repo = OnlineLUTRepository(base, state_path=tmp_path/'state.json')
    assert not (tmp_path/'state.json').exists()
    before = repo.snapshot()
    receipt = repo.observe(observation(), observation_id='request-1')
    assert receipt['status'] == 'updated'
    assert receipt['before_revision'] == 0 and receipt['after_revision'] == 1
    assert len(receipt['updated']) == 4
    assert repo.prefill(1536)[0].ttft_s == pytest.approx(0.9*original[0][0].ttft_s + 0.1*2)
    assert selected_decode(repo).energy_j_per_token == pytest.approx(
        0.9*selected_decode(base).energy_j_per_token+0.1*0.8)
    assert repo.prefill(1537) == base.prefill(1537)
    assert repo.prefill(1024) == base.prefill(1024)
    assert repo.prefill(1536)[1:] == base.prefill(1536)[1:]
    assert (base.prefill(1536), base.decode(1536), repo.quality(1536), repo.prefill_target(1536)) == original
    assert repo.provenance.kind == 'adaptive'
    assert base.provenance.kind == 'synthetic'
    assert before['revision'] == 0
    assert repo.snapshot()['revision'] == 1
    assert repo.export_state()['cells']['decode']['1536:32:510']['tpot_s']['count'] == 1


def test_restart_idempotence_and_actual_n_range(tmp_path):
    path = tmp_path/'state.json'
    repo = OnlineLUTRepository(MockLUTRepository(), state_path=path, context={'model': 'a'})
    repo.observe(observation(), observation_id='a')
    frozen = repo.export_state()
    restored = OnlineLUTRepository(MockLUTRepository(), state_path=path, context={'model': 'a'})
    assert restored.export_state() == frozen
    assert restored.observe(observation(), observation_id='a')['status'] == 'duplicate'
    assert restored.export_state() == frozen
    restored.observe(observation(n=7), observation_id='b')
    stats = restored.export_state()['cells']['decode']['1536:32:510']['tpot_s']
    assert (stats['count'], stats['last_n'], stats['min_n'], stats['max_n']) == (2, 7, 3, 7)


@pytest.mark.parametrize('mutation', [
    lambda r: r['generation'].update(status='interrupted'),
    lambda r: r['timing'].update(valid=False),
    lambda r: r['timing'].update(ttft_s=float('nan')),
    lambda r: r['timing'].update(ttft_s=True),
    lambda r: r['device']['prefill'].update(verified=False),
    lambda r: r['decisions']['prefill']['selected'].update(length=2048),
    lambda r: r['generation'].update(n_output_tokens=4),
    lambda r: r['generation'].update(n_output_tokens=513),
])
def test_invalid_complete_observation_does_not_write(tmp_path, mutation):
    path = tmp_path/'state.json'
    repo = OnlineLUTRepository(MockLUTRepository(), state_path=path)
    result = observation(); mutation(result)
    assert repo.observe(result, observation_id='bad')['status'] == 'skipped'
    assert repo.snapshot()['revision'] == 0
    assert not path.exists()


@pytest.mark.parametrize('energy', [
    {'valid': False, 'prefill_energy_j': 100, 'decode_j_per_token': 100},
    {'valid': True, 'prefill_energy_j': float('nan'), 'decode_j_per_token': -1},
    {'valid': True, 'prefill_energy_j': True, 'decode_j_per_token': None},
])
def test_invalid_energy_preserves_energy_but_updates_time(tmp_path, energy):
    base = MockLUTRepository(); repo = OnlineLUTRepository(base, state_path=tmp_path/'state.json')
    result = observation(); result['energy'] = energy
    receipt = repo.observe(result, observation_id='a')
    assert len(receipt['updated']) == 2
    assert repo.prefill(1536)[0].energy_j == base.prefill(1536)[0].energy_j
    assert selected_decode(repo).energy_j_per_token == selected_decode(base).energy_j_per_token
    assert repo.prefill(1536)[0].ttft_s != base.prefill(1536)[0].ttft_s


def test_n1_updates_prefill_only_and_normalization_mismatch_excludes_decode_time(tmp_path):
    repo = OnlineLUTRepository(MockLUTRepository(), state_path=tmp_path/'state.json')
    assert len(repo.observe(observation(n=1), observation_id='n1')['updated']) == 2
    assert repo.export_state()['cells']['decode'] == {}
    bad = observation(); bad['timing']['decode_duration_s'] = 17
    receipt = repo.observe(bad, observation_id='bad-decode')
    assert len(receipt['updated']) == 3
    assert set(repo.export_state()['cells']['decode']['1536:32:510']) == {'energy_j_per_token'}


@pytest.mark.parametrize('alpha', [0, -1, 1.01, True, float('nan'), float('inf')])
def test_invalid_alpha_rejected(tmp_path, alpha):
    with pytest.raises(LUTError):
        OnlineLUTRepository(MockLUTRepository(), state_path=tmp_path/'state.json', alpha=alpha)


def test_binding_includes_actual_tables_and_context(tmp_path):
    base = MockLUTRepository(); path = tmp_path/'state.json'
    OnlineLUTRepository(base, state_path=path, context={'device': 'a'}).observe(observation(), observation_id='a')
    for kwargs in [{'context': {'device': 'b'}}, {'alpha': 0.2, 'context': {'device': 'a'}}]:
        with pytest.raises(LUTError, match='binding'):
            OnlineLUTRepository(base, state_path=path, **kwargs)
    values = dict(base._prefill); values[(1024,408)] = (9, 9)
    with pytest.raises(LUTError, match='binding'):
        OnlineLUTRepository(MockLUTRepository(prefill=values), state_path=path, context={'device': 'a'})


def test_corruption_schema_keys_counts_and_symlinks_rejected(tmp_path):
    path = tmp_path/'state.json'; base = MockLUTRepository()
    repo = OnlineLUTRepository(base, state_path=path)
    repo.observe(observation(), observation_id='a')
    good = repo.export_state()
    for mutate in [lambda s: s.update(version=True),
                   lambda s: s.update(revision=-1),
                   lambda s: s.update(unexpected=1),
                   lambda s: s['cells']['prefill']['1536:408']['ttft_s'].update(count=True),
                   lambda s: s['cells']['prefill'].update({'1536:999': s['cells']['prefill']['1536:408']})]:
        damaged = deepcopy(good); mutate(damaged)
        path.write_text(json.dumps(checksum(damaged)))
        with pytest.raises(LUTError): OnlineLUTRepository(base, state_path=path)
    path.write_text('{bad')
    with pytest.raises(LUTError): OnlineLUTRepository(base, state_path=path)
    path.unlink(); path.symlink_to(tmp_path/'missing')
    with pytest.raises(LUTError, match='symlink'): OnlineLUTRepository(base, state_path=path)


def test_save_failure_rolls_back_and_stale_writer_rejected(tmp_path, monkeypatch):
    path = tmp_path/'state.json'; base = MockLUTRepository()
    first = OnlineLUTRepository(base, state_path=path)
    stale = OnlineLUTRepository(base, state_path=path)
    initial = first.export_state()
    with monkeypatch.context() as patch:
        patch.setattr('kvolt_runtime.policy.online.os.replace', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(LUTError, match='save'): first.observe(observation(), observation_id='a')
    assert first.export_state() == initial and not path.exists()
    first.observe(observation(), observation_id='a')
    with pytest.raises(LUTError, match='stale'): stale.observe(observation(), observation_id='b')
    assert stale.export_state() == initial


def test_filter_scans_updated_latency_and_preserves_energy_screening(tmp_path):
    base = MockLUTRepository(); repo = OnlineLUTRepository(base, state_path=tmp_path/'state.json', alpha=1)
    result = observation(); result['timing']['ttft_s'] = 0.01
    repo.observe(result, observation_id='fast-low-frequency')
    filtered = repo.prefill_within_budget(1536, 0.02).predictions
    assert [p.frequency_mhz for p in filtered] == [408]
    assert filtered[0].ttft_s == 0.01


def test_post_replace_directory_fsync_failure_reports_actual_committed_state(tmp_path, monkeypatch):
    path = tmp_path/'state.json'
    repo = OnlineLUTRepository(MockLUTRepository(), state_path=path)
    original_fsync = os.fsync
    def fsync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError('directory unavailable')
        return original_fsync(descriptor)
    with monkeypatch.context() as patch:
        patch.setattr('kvolt_runtime.policy.online.os.fsync', fsync)
        with pytest.raises(LUTError, match='committed.*durability uncertain') as caught:
            repo.observe(observation(), observation_id='committed')
    assert caught.value.snapshot == repo.snapshot()
    assert repo.snapshot()['revision'] == 1
    assert json.loads(path.read_bytes()) == repo.export_state()
    reopened = OnlineLUTRepository(repo.base, state_path=path)
    assert reopened.export_state() == repo.export_state()
    assert reopened.observe(observation(), observation_id='committed')['status'] == 'duplicate'


def test_locked_state_rejects_update_without_wait_or_mutation(tmp_path):
    path = tmp_path/'state.json'
    repo = OnlineLUTRepository(MockLUTRepository(), state_path=path)
    with path.with_name('state.json.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(LUTError, match='lock'):
            repo.observe(observation(), observation_id='busy')
    assert repo.snapshot()['revision'] == 0 and not path.exists()


def test_persisted_checksum_duplicate_json_keys_and_output_count_bounds(tmp_path):
    path = tmp_path/'state.json'
    repo = OnlineLUTRepository(MockLUTRepository(), state_path=path)
    repo.observe(observation(), observation_id='a')
    good = repo.export_state()
    bad = deepcopy(good); bad['revision'] = 2
    path.write_text(json.dumps(bad))
    with pytest.raises(LUTError, match='checksum'):
        OnlineLUTRepository(repo.base, state_path=path)
    bad = deepcopy(good)
    bad['cells']['decode']['1536:32:510']['tpot_s'].update(last_n=513, max_n=513)
    path.write_text(json.dumps(checksum(bad)))
    with pytest.raises(LUTError, match='output counts'):
        OnlineLUTRepository(repo.base, state_path=path)
    path.write_text('{"version":1,"version":1}')
    with pytest.raises(LUTError, match='duplicate'):
        OnlineLUTRepository(repo.base, state_path=path)


def test_decode_time_and_energy_are_independent_and_zero_energy_is_valid(tmp_path):
    repo = OnlineLUTRepository(MockLUTRepository(), state_path=tmp_path/'state.json')
    result = observation(); result['timing']['mean_tpot_s'] = None
    result['energy'].update(prefill_energy_j=0, decode_energy_j=0, decode_j_per_token=0)
    receipt = repo.observe(result, observation_id='energy-only')
    assert len(receipt['updated']) == 3
    cell = repo.export_state()['cells']['decode']['1536:32:510']
    assert set(cell) == {'energy_j_per_token'} and cell['energy_j_per_token']['count'] == 1


def test_conflicting_duration_is_not_hidden_by_consistent_boundaries(tmp_path):
    repo = OnlineLUTRepository(MockLUTRepository(), state_path=tmp_path/'state.json')
    result = observation()
    result['timing'].update(t0_ns=0, t1_ns=2_000_000_000, tN_ns=3_000_000_000,
                            decode_duration_s=17)
    receipt = repo.observe(result, observation_id='conflicting-duration')
    assert 'decode_normalization_mismatch' in receipt['reasons']
    assert 'tpot_s' not in repo.export_state()['cells']['decode']['1536:32:510']
