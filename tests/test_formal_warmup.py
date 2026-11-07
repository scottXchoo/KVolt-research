"""CPU state-machine expectations; no GPU or compiled-kernel claims."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kvolt_runtime.contracts import CacheFormat, TokenResult
from kvolt_runtime.execution.contracts import ExecutionError
from kvolt_runtime.qualification.contracts import QualificationGrant, canonical_json
from kvolt_runtime.qualification.warmup import formal_warmup
from test_session_runner import setup


def run_warmup(runner, prepared):
    return formal_warmup(backend=runner.backend, device=runner.device,
                         prepared=prepared, snapshot={'verified': True},
                         clock=runner.clock, stop=runner.stop)


def test_full_512_generation_visits_every_reachable_residual_shape(tmp_path):
    runner, prepared, _, backend, events, _, _ = setup(tmp_path, nmax=512)
    report = run_warmup(runner, prepared)
    assert report['formal_coverage_qualified'] is True
    assert report['coverage_scope'] == 'actual_inputs_natural_reachable_shapes'
    assert [case['group'] for case in report['cases']] == [8, 16, 32, 64, 128]
    for case in report['cases']:
        assert case['nmax'] == case['n_output_tokens'] == 512
        assert (case['prefill_forwards'], case['initial_packs'], case['decode_forwards']) == (1, 1, 511)
        assert case['token_ids'] == [17] + [23] * 511
        assert case['final_cache_length'] == 1535
        assert case['stop_reason'] == 'length'
        assert case['cache_released'] is True
        assert [shape['cache_length'] for shape in case['decode_shapes']] == list(range(1024, 1535))
        assert [shape['key_residual_length'] for shape in case['decode_shapes'][:130]] == list(range(128)) + [0, 1]
        assert case['decode_shapes'][127]['result_cache_length'] == 1152
        assert case['decode_shapes'][255]['result_cache_length'] == 1280
        assert case['decode_shapes'][383]['result_cache_length'] == 1408
        assert {shape['value_residual_length'] for shape in case['decode_shapes']} == {128}
    assert events.count('decode') == 2555
    assert events[-1] == 'restore'
    assert len(backend.caches) == 5
    assert all(cache.format is CacheFormat.RELEASED for cache in backend.caches)


@pytest.mark.parametrize('nmax,first,count,reason', [
    (1, 17, 1, 'nmax_one'), (512, 128001, 5, 'first_token_eos')])
def test_first_eos_and_n1_never_pack_or_decode(tmp_path, nmax, first, count, reason):
    runner, prepared, _, _, events, _, _ = setup(tmp_path, nmax=nmax, first=first)
    report = run_warmup(runner, prepared)
    assert len(report['cases']) == count
    assert 'decode' not in events
    assert not any(isinstance(event, tuple) and event[0] == 'pack' for event in events)
    for case in report['cases']:
        assert case['n_output_tokens'] == 1
        assert case['skip_reason'] == reason
        assert case['stop_reason'] == ('eos' if reason == 'first_token_eos' else 'length')
        assert case['initial_packs'] == case['decode_forwards'] == 0
        assert case['decode_shapes'] == [] and case['initial_pack_shape'] is None
        assert case['mean_tpot_s'] is case['decode_j_per_token'] is None


def test_context_dedup_uses_maximum_cap_and_does_not_merge_first_eos_input(tmp_path):
    runner, prepared, _, backend, _, _, _ = setup(tmp_path, nmax=2, requests=2)
    one, two = prepared.requests
    duplicate = replace(two, spec=replace(two.spec, nmax=4))
    eos = replace(two, spec=replace(two.spec, request_id='eos', token_ids=(9,) * 1024, nmax=8))
    metadata = replace(two, spec=replace(two.spec, request_id='other-context', nmax=3), metadata_json='{"condition":"other"}')
    prepared = replace(prepared, requests=(one, duplicate, eos, metadata))
    prefill = backend.prefill_raw
    def select_first(spec):
        token = prefill(spec)
        return TokenResult(128001 if spec.token_ids[0] == 9 else 17, token.ready_ns, token.cache)
    backend.prefill_raw = select_first
    report = run_warmup(runner, prepared)
    assert len(report['cases']) == 15
    active, stopped, other = report['cases'][:5], report['cases'][5:10], report['cases'][10:]
    assert all(case['nmax'] == 4 and case['decode_forwards'] == 3 for case in active)
    assert active[0]['requests'] == [{'request_id': 'request-0', 'nmax': 2}, {'request_id': 'request-1', 'nmax': 4}]
    assert all(case['stop_reason'] == 'eos' and case['skip_reason'] == 'first_token_eos'
               and not case['decode_shapes'] for case in stopped)
    assert active[0]['input_token_sha256'] != stopped[0]['input_token_sha256']
    assert all(case['nmax'] == 3 and case['decode_forwards'] == 2 for case in other)


def test_later_natural_eos_does_not_claim_unreachable_lengths(tmp_path):
    runner, prepared, _, backend, _, _, _ = setup(tmp_path, nmax=512)
    decode = backend.decode_one
    def eos_after_two(cache, previous):
        token = decode(cache, previous)
        return TokenResult(128009 if cache.metadata.length == 1026 else token.token_id,
                           token.ready_ns, token.cache)
    backend.decode_one = eos_after_two
    report = run_warmup(runner, prepared)
    for case in report['cases']:
        assert case['n_output_tokens'] == 3 and case['decode_forwards'] == 2
        assert case['stop_reason'] == 'eos'
        assert [shape['cache_length'] for shape in case['decode_shapes']] == [1024, 1025]


@pytest.mark.parametrize('failure', ['prefill', 'pack', 'decode', 'release'])
def test_forward_and_release_failure_stop_all_further_shapes_and_restore(tmp_path, failure):
    runner, prepared, _, backend, events, _, _ = setup(tmp_path, fail=failure)
    with pytest.raises(ExecutionError) as caught:
        run_warmup(runner, prepared)
    report = caught.value.receipt
    assert report['formal_coverage_qualified'] is False and report['errors']
    assert len(report['cases']) == 1
    assert events[-1] == 'restore'
    assert all(cache.format is CacheFormat.RELEASED for cache in backend.caches)


def test_restore_failure_rejects_coverage(tmp_path):
    runner, prepared, _, _, _, _, _ = setup(tmp_path, device_fail='restore')
    with pytest.raises(ExecutionError) as caught:
        run_warmup(runner, prepared)
    assert caught.value.receipt['formal_coverage_qualified'] is False
    assert caught.value.receipt['errors'][-1]['stage'] == 'warmup_restore'


@pytest.mark.parametrize('at', ['release', 'restore'])
def test_signal_during_final_cleanup_cannot_qualify_ledger(tmp_path, at):
    runner, prepared, _, backend, _, _, _ = setup(tmp_path, nmax=1)
    owner, name = (backend, 'release_request') if at == 'release' else (runner.device, 'restore')
    original = getattr(owner, name)
    def signal_cleanup(value):
        result = original(value)
        runner.stop.request(2)
        return result
    setattr(owner, name, signal_cleanup)
    with pytest.raises(ExecutionError) as caught:
        run_warmup(runner, prepared)
    assert caught.value.receipt['interrupted'] is True
    assert caught.value.receipt['formal_coverage_qualified'] is False


@pytest.mark.parametrize('at', ['before', 'prefill', 'pack', 'decode'])
def test_interruption_releases_restores_and_stops_all_further_work(tmp_path, at):
    runner, prepared, _, backend, events, _, _ = setup(tmp_path, nmax=512, stop_at=at)
    if at == 'before':
        runner.stop.request(2)
    if at == 'pack':
        pack = backend.pack_initial
        def interrupted_pack(cache, group):
            pack(cache, group)
            runner.stop.request(2)
        backend.pack_initial = interrupted_pack
    with pytest.raises(ExecutionError) as caught:
        run_warmup(runner, prepared)
    report = caught.value.receipt
    assert report['interrupted'] is True and report['formal_coverage_qualified'] is False
    assert len(report['cases']) == (0 if at == 'before' else 1)
    assert events.count('decode') == (1 if at == 'decode' else 0)
    assert events[-1] == 'restore'
    assert all(cache.format is CacheFormat.RELEASED for cache in backend.caches)


def grant_for(runner, collector):
    source = runner.repo.provenance
    runner.repo.provenance = replace(source, kind='measured', source_device='agx2')
    source = runner.repo.provenance
    profile = SimpleNamespace(as_dict=lambda: {'profile': 'device-v1'})
    runner.device.profile = profile
    collector.profile = SimpleNamespace(to_dict=lambda: {'profile': 'power-v1'})
    binding = {'identity': {'physical_id_sha256': 'a' * 64, 'boot_id': 'boot'},
               'native_profile': 'agx2', 'device_profile': profile.as_dict(),
               'power_profile': collector.profile.to_dict(),
               'lut': {name: getattr(source, name) for name in
                       ('kind', 'source_device', 'source_commit', 'manifest_sha256')}}
    binding['lut']['files_sha256'] = dict(source.files_sha256)
    return QualificationGrant(canonical_json(binding), '{"1024":512}', 'b' * 64, '/report.json')


@pytest.mark.parametrize('forged', [None, True, {'formal_performance_qualified': True}])
def test_performance_rejects_forged_grants_before_capture_and_load(tmp_path, forged):
    runner, prepared, options, backend, events, _, _ = setup(tmp_path, collect=True)
    runner.warmup, runner.qualification = 'formal', forged
    options = replace(options, mode='performance', native_profile='agx2')
    result = runner.run(prepared, options, provenance={'formal_performance_qualified': True})
    assert result['exit_code'] == 2 and backend.loads == 0
    assert 'capture' not in events and 'prepare' not in events


@pytest.mark.parametrize('bad', ['warmup', 'collector', 'synthetic', 'device', 'power', 'native', 'coverage'])
def test_performance_binding_mismatch_rejects_before_capture(tmp_path, bad):
    runner, prepared, options, backend, events, _, collector = setup(tmp_path, collect=True)
    runner.warmup, runner.qualification = 'formal', grant_for(runner, collector)
    options = replace(options, mode='performance', native_profile='agx2')
    if bad == 'warmup': runner.warmup = True
    if bad == 'collector': runner.collector_factory = None
    if bad == 'synthetic': runner.repo.provenance = replace(runner.repo.provenance, kind='synthetic')
    if bad == 'device': runner.device.profile = SimpleNamespace(as_dict=lambda: {'changed': True})
    if bad == 'power': collector.profile = SimpleNamespace(to_dict=lambda: {'changed': True})
    if bad == 'native': options = replace(options, native_profile='agx1')
    if bad == 'coverage': runner.qualification = replace(runner.qualification, coverage_json='{"1024":1}')
    result = runner.run(prepared, options)
    assert result['exit_code'] == 2 and backend.loads == 0
    assert 'capture' not in events


def test_device_identity_mismatch_blocks_load_after_read_only_capture(tmp_path):
    runner, prepared, options, backend, events, _, collector = setup(tmp_path, collect=True)
    runner.warmup, runner.qualification = 'formal', grant_for(runner, collector)
    result = runner.run(prepared, replace(options, mode='performance', native_profile='agx2'))
    assert result['exit_code'] == 2 and backend.loads == 0
    assert events == ['capture']


def test_development_formal_warmup_runs_before_requests_without_grant(tmp_path):
    runner, prepared, options, _, _, _, _ = setup(tmp_path, nmax=4)
    runner.warmup = 'formal'
    result = runner.run(prepared, options)
    assert result['exit_code'] == 0
    assert result['warmup']['formal_coverage_qualified'] is True
    assert 'qualification' not in result
    assert result['warmup']['ended_ns'] < result['results'][0]['timing']['t0_ns']
    assert [case['decode_forwards'] for case in result['warmup']['cases']] == [3] * 5


def test_bound_performance_session_runs_formal_preparation_before_power_start(tmp_path):
    runner, prepared, options, backend, events, _, collector = setup(tmp_path, nmax=3, collect=True)
    runner.warmup, runner.qualification = 'formal', grant_for(runner, collector)
    capture = runner.device.capture
    def bound_capture():
        return {**capture(), 'identity': runner.qualification.binding['identity']}
    runner.device.capture = bound_capture
    created = []
    def collect(index):
        created.append(index)
        return collector
    runner.collector_factory = collect
    result = runner.run(prepared, replace(options, mode='performance', native_profile='agx2'))
    assert result['exit_code'] == 0 and backend.loads == 1
    assert result['qualification']['report_sha256'] == 'b' * 64
    assert result['warmup']['formal_coverage_qualified'] is True
    assert created == [0]
    assert events[:3] == ['capture', 'verify', 'load']
    assert events.index('power_start') > events.index('restore')
    assert result['results'][0]['generation']['decode_forwards'] == 2
    assert result['results'][0]['energy']['valid'] is True


def test_formal_warmup_failure_blocks_all_timed_requests_and_keeps_ledger(tmp_path):
    runner, prepared, options, _, _, _, _ = setup(tmp_path, fail='pack', requests=2)
    runner.warmup = 'formal'
    result = runner.run(prepared, options)
    assert result['exit_code'] == 1 and result['counts']['not_run'] == 2
    assert result['warmup']['formal_coverage_qualified'] is False
    assert all(item['timing']['t0_ns'] is None for item in result['results'])
