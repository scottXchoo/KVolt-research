"""Independent CPU integration of real PC/DC, frozen inputs and real recording."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from kvolt_runtime.contracts import CacheFormat, CacheHandle, CacheMetadata, TokenResult
from kvolt_runtime.execution.contracts import ExecutionError, PreparationError, SessionOptions, StopController
from kvolt_runtime.execution.inputs import preflight_inputs
from kvolt_runtime.execution.runner import SessionRunner
from kvolt_runtime.policy.decode import DecodeController
from kvolt_runtime.policy.knee import KneeAdapter
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.types import KneeResult
from kvolt_runtime.recording import SessionRecorder
from kvolt_runtime.telemetry import RAILS


class Clock:
    def __init__(self):
        self.now = 10_000_000_000
    def __call__(self):
        self.now += 1_000_000
        return self.now


class Tokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {'input_ids': [7] * len(text)}
    def decode(self, tokens, **kwargs):
        assert kwargs == {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False}
        return ' '.join(map(str, tokens))


class Backend:
    def __init__(self, clock, events, *, first=17, fail=None, stop=None):
        self.clock, self.events, self.first, self.fail, self.stop = clock, events, first, fail, stop
        self.tokenizer = Tokenizer()
        self.provenance = {'test_backend': True}
        self.loads = 0
        self.caches = []
    def load_session(self, path, config):
        self.events.append('load')
        self.loads += 1
        if self.fail == 'load':
            raise RuntimeError('load failed')
        assert config.dtype == 'float16' and not config.do_sample
    def prefill_raw(self, request):
        self.events.append(('prefill', request.request_id))
        assert all(cache.format is CacheFormat.RELEASED for cache in self.caches)
        if self.fail == 'prefill':
            raise RuntimeError('prefill failed')
        cache = CacheHandle(CacheMetadata(request.request_id, request.length, 'cuda:fake', CacheFormat.RAW), object())
        self.caches.append(cache)
        ready = self.clock()
        if self.stop == 'prefill':
            self.controller.request(2)
        return TokenResult(self.first, ready, cache)
    def pack_initial(self, cache, group):
        self.events.append(('pack', group))
        if self.fail == 'pack':
            raise RuntimeError('pack failed')
        cache.commit_packed(replace(cache.metadata, format=CacheFormat.PACKED, group_size=group), object())
    def decode_one(self, cache, previous):
        self.events.append('decode')
        if self.fail == 'decode':
            raise RuntimeError('decode failed')
        cache.commit_decode(replace(cache.metadata, length=cache.metadata.length + 1), object())
        ready = self.clock()
        if self.stop == 'decode':
            self.controller.request(15)
        return TokenResult(23, ready, cache)
    def release_request(self, cache):
        self.events.append('release')
        cache.release()
        if self.fail == 'release':
            raise RuntimeError('release failed')


class Device:
    def __init__(self, clock, events, *, fail=None):
        self.clock, self.events, self.fail = clock, events, fail
    def receipt(self, operation):
        self.events.append(operation)
        if operation == self.fail:
            raise PreparationError('DEVICE_NOT_PREPARED', 'device', 'not prepared') if operation == 'verify' else RuntimeError(operation)
        return {'operation': operation, 'verified': True,
                'start_ns': self.clock(), 'end_ns': self.clock()}
    def capture(self): return self.receipt('capture')
    def verify_prepared(self): return self.receipt('verify')
    def prepare_request(self): return self.receipt('prepare')
    def apply(self, frequency): return self.receipt(('apply', frequency))
    def restore(self, snapshot): return self.receipt('restore')


class Adapter:
    def __init__(self, events, result=None, clock=None, delay_ns=0):
        self.events, self.result, self.clock, self.delay_ns = events, result, clock, delay_ns
    def detect(self, quality):
        self.events.append('knee')
        if self.clock:
            self.clock.now += self.delay_ns
        return self.result or KneeResult('not_found', reason='test_flat')


class Collector:
    def __init__(self, clock, events, *, failure=None):
        self.clock, self.events, self.failure = clock, events, failure
        self.samples = []
        self.receipt = None
    def start(self):
        self.events.append('power_start')
        t = self.clock()
        self.samples = [{'rail': rail, 'timestamp_ns': t, 'power_w': 2.0} for rail in RAILS]
        return {'ready': self.failure != 'start', 'errors': []}
    def finish(self, end):
        self.events.append(('power_finish', end))
        t = self.clock()
        self.samples.extend({'rail': rail, 'timestamp_ns': t, 'power_w': 2.0} for rail in RAILS)
        return self.stop()
    def stop(self):
        self.events.append('power_stop')
        if self.receipt is None:
            raw = ''.join(json.dumps(row) + '\n' for row in self.samples)
            self.receipt = {'samples': self.samples, 'errors': [{'code': 'WORKER_FAILED'}] if self.failure == 'finish' else [],
                            'profile': {'max_gap_ns': 10**12}, 'raw_trace_jsonl': raw,
                            'trace_sha256': hashlib.sha256(raw.encode()).hexdigest()}
        return self.receipt


def setup(tmp_path, *, nmax=3, requests=1, first=17, fail=None, device_fail=None,
          stop_at=None, knee=None, delay_ns=0, collect=False, collector_failure=None, warmup=False):
    events, emitted = [], []
    clock, repo, stop = Clock(), MockLUTRepository(), StopController()
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('a' * 1024)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 'kvolt-request-manifest-v1', 'requests': [
        {'request_id': f'request-{index}', 'prompt_file': 'prompt.txt', 'nmax': nmax} for index in range(requests)]}))
    prepared = preflight_inputs(prompt_file=None, manifest=manifest, tokenizer=Tokenizer(), repo=repo)
    recorder = SessionRecorder(tmp_path / 'out')
    recorder.save_preflight(prepared)
    backend = Backend(clock, events, first=first, fail=fail, stop=stop_at)
    backend.controller = stop
    device = Device(clock, events, fail=device_fail)
    adapter = knee or Adapter(events, clock=clock, delay_ns=delay_ns)
    dc = DecodeController(repo, adapter, clock=clock)
    collector = Collector(clock, events, failure=collector_failure)
    runner = SessionRunner(backend=backend, repo=repo, device=device, recorder=recorder,
                           decode_controller=dc, clock=clock, stop=stop,
                           collector_factory=lambda index: collector, emit=emitted.append, warmup=warmup)
    options = SessionOptions(tmp_path / 'model', tmp_path / 'out', collect_power=collect)
    return runner, prepared, options, backend, events, emitted, collector


def test_sequential_resident_model_fresh_cache_and_phase_order(tmp_path):
    runner, prepared, options, backend, events, emitted, _ = setup(tmp_path, requests=2)
    session = runner.run(prepared, options)
    assert session['exit_code'] == 0 and backend.loads == 1
    assert session['counts']['generation_completed'] == 2
    assert all(cache.format is CacheFormat.RELEASED for cache in backend.caches)
    assert events[:3] == ['capture', 'verify', 'load']
    start = events.index(('prefill', 'request-0'))
    assert events[start + 1] == 'knee'
    assert events[start + 2][0] == 'apply'
    assert events[start + 3][0] == 'pack'
    assert events[start + 4:start + 8] == ['decode', 'decode', 'release', 'restore']
    assert [record['record_type'] for record in emitted] == ['request_result', 'request_result', 'session_result']
    for result in session['results']:
        gen, timing = result['generation'], result['timing']
        assert gen['n_output_tokens'] == 3 and gen['decode_forwards'] == 2 and gen['initial_packs'] == 1
        assert gen['final_cache_length'] == 1026
        assert timing['t0_ns'] < timing['tp_ns'] < timing['t1_ns'] <= timing['td_ns'] < timing['tN_ns']
        assert timing['ttft_s'] == (timing['t1_ns'] - timing['t0_ns']) / 1e9
        assert timing['mean_tpot_s'] == (timing['tN_ns'] - timing['t1_ns']) / 2e9
        pc = result['decisions']['prefill']
        assert pc['selection_rule'] == 'reference_budget'
        assert pc['tp_ns'] == pc['prefilter_ns'] == timing['tp_ns'] == timing['t0_ns'] + 1_000_000
        assert pc['lookup_end_ns'] == pc['tp_ns'] + 1_000_000
        assert pc['lookup_end_ns'] <= result['device']['prefill']['start_ns']
        assert pc['budget_s'] == 0.999 and pc['selected']['frequency_mhz'] == 510
        assert [c['prediction']['frequency_mhz'] for c in pc['candidates']] == [510, 612, 714, 816, 918, 1020, 1122]
        assert timing['tN_ns'] < result['cleanup_started_ns'] < result['cleanup_ended_ns']
        assert result['energy']['status'] == 'not_collected'
    saved = json.loads((tmp_path / 'out/session.json').read_text())
    assert saved['results'][0]['generation']['token_ids'] == [17, 23, 23]


@pytest.mark.parametrize('nmax,first,reason', [(1, 17, 'nmax_one'), (3, 128009, 'first_token_eos'), (1, 128001, 'first_token_eos')])
def test_first_token_stop_skips_real_dc_and_pack(tmp_path, nmax, first, reason):
    runner, prepared, options, _, events, _, _ = setup(tmp_path, nmax=nmax, first=first)
    session = runner.run(prepared, options)
    result = session['results'][0]
    assert session['exit_code'] == 0 and result['generation']['n_output_tokens'] == 1
    assert 'knee' not in events and 'decode' not in events
    assert not any(isinstance(event, tuple) and event[0] == 'pack' for event in events)
    assert result['decisions']['decode']['skip_reason'] == reason
    assert result['timing']['mean_tpot_s'] is None
    assert result['generation']['stage_timing'] == {'initial_pack': None, 'first_decode': None}


def test_decision_serialization_follows_final_token_without_changing_measurements(tmp_path, monkeypatch):
    import kvolt_runtime.execution.runner as runner_module
    from kvolt_runtime import config
    # Preserve this historical 33-row serialization/measurement boundary fixture.
    monkeypatch.setattr(config, 'TPOT_BUFFER_RATIO', 0.0)
    runner, prepared, options, backend, events, _, _ = setup(
        tmp_path, knee=Adapter([], result=KneeResult('error_recovered', reason='test_recovery')))
    original_asdict = runner_module.asdict
    serialized = []
    def serialize(decision):
        # Model expensive evidence conversion with the same shared clock. Any
        # conversion on the generation path would inflate the actual metrics.
        serialized.append((type(decision).__name__, runner.clock.now, events.count('decode')))
        runner.clock.now += 10_000_000_000
        return original_asdict(decision)
    monkeypatch.setattr(runner_module, 'asdict', serialize)
    session = runner.run(prepared, options)
    result = session['results'][0]
    assert session['exit_code'] == 0
    assert [name for name, _, _ in serialized] == ['PrefillDecision', 'DecodeDecision']
    assert all(time >= result['timing']['tN_ns'] and n == 2 for _, time, n in serialized)
    assert result['timing']['ttft_s'] < 1 and result['timing']['mean_tpot_s'] < 1
    assert len(result['decisions']['prefill']['candidates']) == 7
    # Recovery activates all G, but only 33 combinations meet this budget.
    assert len(result['decisions']['decode']['candidates']) == 33
    assert all(row['eligible'] for row in result['decisions']['decode']['candidates'])
    assert result['events'] == [{'kind': 'knee_error_recovered', **result['decisions']['decode']['knee']}]
    assert backend.caches[0].format is CacheFormat.RELEASED


@pytest.mark.parametrize('failure', ['prefill', 'pack', 'decode'])
def test_failed_generation_preserves_selected_decisions_and_stage_bounds(tmp_path, failure):
    runner, prepared, options, _, _, _, _ = setup(tmp_path, fail=failure)
    session = runner.run(prepared, options)
    result = session['results'][0]
    assert session['exit_code'] == 1 and result['generation']['status'] == 'failed'
    assert len(result['decisions']['prefill']['candidates']) == 7
    dc = result['decisions']['decode']
    stages = result['generation']['stage_timing']
    if failure == 'prefill':
        assert dc is None and stages == {'initial_pack': None, 'first_decode': None}
    else:
        assert len(dc['candidates']) == 8 and dc['selected'] is not None
        assert result['timing']['t1_ns'] <= stages['initial_pack']['start_ns']
        assert stages['initial_pack']['start_ns'] < stages['initial_pack']['end_ns']
        if failure == 'pack':
            assert stages['first_decode'] is None
        else:
            assert stages['initial_pack']['end_ns'] < stages['first_decode']['start_ns']
            assert stages['first_decode']['start_ns'] < stages['first_decode']['end_ns']
            assert stages['first_decode']['end_ns'] <= result['failed_ns']
    assert result['cleanup']['status'] == 'completed'


@pytest.mark.parametrize('nmax', [2, 3])
def test_stage_bounds_partition_existing_decode_interval(tmp_path, nmax):
    runner, prepared, options, backend, _, _, _ = setup(tmp_path, nmax=nmax)
    pack, decode = backend.pack_initial, backend.decode_one
    calls = []
    def measured_pack(cache, group):
        calls.append(('pack', runner.clock.now))
        pack(cache, group)
    def measured_decode(cache, previous):
        calls.append(('decode', runner.clock.now))
        return decode(cache, previous)
    backend.pack_initial, backend.decode_one = measured_pack, measured_decode
    result = runner.run(prepared, options)['results'][0]
    timing, stages = result['timing'], result['generation']['stage_timing']
    pack, decode = stages['initial_pack'], stages['first_decode']
    assert timing['t1_ns'] <= timing['td_ns'] < pack['start_ns']
    assert pack['start_ns'] == calls[0][1] < pack['end_ns'] < decode['start_ns']
    assert decode['start_ns'] == calls[1][1] < result['generation']['token_ready_ns'][1] < decode['end_ns']
    if nmax == 2:
        # Backend stamps readiness before returning, so the diagnostic wrapper
        # can end just after tN without extending the token measurement end.
        assert timing['tN_ns'] < decode['end_ns']
    else:
        assert decode['end_ns'] < result['generation']['token_ready_ns'][2] == timing['tN_ns']
    assert timing['mean_tpot_s'] == (timing['tN_ns'] - timing['t1_ns']) / ((nmax - 1) * 1e9)
    saved = json.loads((tmp_path / 'out/requests/000000/result.json').read_text())
    assert saved['generation']['stage_timing'] == stages


def test_knee_exception_recovered_and_cost_forces_latency_fallback(tmp_path):
    def broken(**kwargs):
        raise RuntimeError('detector failed')
    runner, prepared, options, _, _, _, _ = setup(tmp_path, knee=KneeAdapter(broken))
    # Make the DC clock cost exceed every synthetic time budget.
    runner.dc.clock = lambda: runner.clock() + 10**12
    session = runner.run(prepared, options)
    result = session['results'][0]
    assert session['exit_code'] == 0
    assert result['decisions']['decode']['knee']['status'] == 'error_recovered'
    assert result['events'][0]['kind'] == 'knee_error_recovered'
    assert result['decisions']['decode']['selection_path'] == 'latency_fallback'
    assert all(row['active_knee'] for row in result['decisions']['decode']['candidates'])


@pytest.mark.parametrize('stage', ['prefill', 'pack', 'decode', 'release'])
def test_technical_failure_aborts_without_retry_and_later_not_run(tmp_path, stage):
    runner, prepared, options, backend, events, emitted, _ = setup(tmp_path, requests=3, fail=stage)
    session = runner.run(prepared, options)
    assert session['exit_code'] == 1 and backend.loads == 1
    assert [r['generation']['status'] for r in session['results']][1:] == ['not_run', 'not_run']
    assert sum(isinstance(event, tuple) and event[0] == 'prefill' for event in events) == 1
    assert events.count('restore') == 1
    assert all(cache.format is CacheFormat.RELEASED for cache in backend.caches)
    assert len(emitted) == 4
    for result in session['results'][1:]:
        assert result['generation']['token_ids'] == [] and result['timing']['ttft_s'] is None


def test_restore_failure_preserves_normal_measurement(tmp_path):
    runner, prepared, options, _, _, _, _ = setup(tmp_path, requests=2, device_fail='restore')
    session = runner.run(prepared, options)
    first = session['results'][0]
    assert session['exit_code'] == 1 and first['cleanup']['status'] == 'failed'
    assert first['generation']['status'] == 'completed' and first['timing']['valid']
    assert first['timing']['ttft_s'] is not None
    assert session['results'][1]['generation']['status'] == 'not_run'


@pytest.mark.parametrize('stop_at,exit_code,n', [('prefill', 130, 1), ('decode', 143, 2)])
def test_signal_stops_after_current_forward_preserves_prefix(tmp_path, stop_at, exit_code, n):
    runner, prepared, options, _, events, _, _ = setup(tmp_path, requests=2, stop_at=stop_at)
    session = runner.run(prepared, options)
    first = session['results'][0]
    assert session['exit_code'] == exit_code and session['status'] == 'interrupted'
    assert first['generation']['status'] == 'interrupted' and first['generation']['n_output_tokens'] == n
    assert first['timing']['valid'] is False and first['timing']['prefix']['ttft_s'] is not None
    assert first['timing']['ttft_s'] is None and first['cleanup']['status'] == 'completed'
    assert len(first['decisions']['prefill']['candidates']) == 7
    stages = first['generation']['stage_timing']
    if stop_at == 'prefill':
        assert first['decisions']['decode'] is None
        assert stages == {'initial_pack': None, 'first_decode': None}
    else:
        assert len(first['decisions']['decode']['candidates']) == 8
        assert stages['initial_pack']['start_ns'] < stages['initial_pack']['end_ns']
        assert stages['first_decode']['start_ns'] < stages['first_decode']['end_ns']
    assert session['results'][1]['generation']['status'] == 'not_run'
    assert events.count('decode') == n - 1


def test_signal_inside_request_pack_preserves_decisions_and_skips_decode(tmp_path):
    runner, prepared, options, backend, events, _, _ = setup(tmp_path)
    pack = backend.pack_initial
    def interrupted_pack(cache, group):
        pack(cache, group)
        runner.stop.request(2)
    backend.pack_initial = interrupted_pack
    session = runner.run(prepared, options)
    result = session['results'][0]
    assert session['exit_code'] == 130 and result['generation']['status'] == 'interrupted'
    assert len(result['decisions']['decode']['candidates']) == 8
    stages = result['generation']['stage_timing']
    assert stages['initial_pack']['start_ns'] < stages['initial_pack']['end_ns']
    assert stages['first_decode'] is None and 'decode' not in events
    assert result['generation']['initial_packs'] == 1
    assert result['cleanup']['status'] == 'completed'


def test_interruption_before_pc_has_no_decision_or_stage_evidence(tmp_path):
    runner, prepared, options, _, _, _, _ = setup(tmp_path)
    prepare = runner.device.prepare_request
    def interrupted_prepare():
        receipt = prepare()
        runner.stop.request(2)
        return receipt
    runner.device.prepare_request = interrupted_prepare
    session = runner.run(prepared, options)
    result = session['results'][0]
    assert session['exit_code'] == 130 and result['generation']['status'] == 'interrupted'
    assert result['decisions'] == {'prefill': None, 'decode': None}
    assert result['generation']['stage_timing'] == {'initial_pack': None, 'first_decode': None}
    assert result['cleanup']['status'] == 'completed'


def test_device_rejection_precedes_model_and_returns_three(tmp_path):
    runner, prepared, options, backend, _, _, _ = setup(tmp_path, device_fail='verify')
    session = runner.run(prepared, options)
    assert session['exit_code'] == 3 and backend.loads == 0
    assert session['counts']['not_run'] == 1


def test_model_load_failure_restores_and_returns_one(tmp_path):
    runner, prepared, options, _, events, _, _ = setup(tmp_path, fail='load')
    session = runner.run(prepared, options)
    assert session['exit_code'] == 1 and events[-1] == 'restore'
    assert session['counts']['not_run'] == 1


def test_warmup_exercises_all_groups_outside_t0_and_releases_caches(tmp_path):
    runner, prepared, options, backend, events, _, _ = setup(tmp_path, warmup=True)
    session = runner.run(prepared, options)
    assert session['exit_code'] == 0
    assert [case['group'] for case in session['warmup']['cases']] == [8, 16, 32, 64, 128]
    assert len(backend.caches) == 6
    assert session['warmup']['ended_ns'] < session['results'][0]['timing']['t0_ns']
    assert events.count('restore') == 2


def test_power_brackets_raw_jsonl_csv_and_nonoverlapping_energy(tmp_path):
    runner, prepared, options, _, events, _, collector = setup(tmp_path, collect=True)
    session = runner.run(prepared, options)
    result = session['results'][0]
    assert session['exit_code'] == 0 and result['energy']['valid']
    assert events.index('power_start') < events.index(('prefill', 'request-0'))
    assert ('power_finish', result['timing']['tN_ns']) in events
    expected = 6 * (result['timing']['tN_ns'] - result['timing']['t0_ns']) / 1e9
    assert result['energy']['total_energy_j'] == pytest.approx(expected)
    assert result['energy']['total_energy_j'] == pytest.approx(
        result['energy']['prefill_energy_j'] + result['energy']['decode_energy_j'])
    raw = (tmp_path / 'out/telemetry/000000/power.raw.jsonl').read_text()
    assert raw == collector.receipt['raw_trace_jsonl']
    assert (tmp_path / 'out/telemetry/000000/power.csv').is_file()
    assert result['power']['raw_path'] == 'telemetry/000000/power.raw.jsonl'
    assert 'raw_trace_jsonl' not in result['power']


def test_collector_failure_never_retains_normal_energy_numbers(tmp_path):
    runner, prepared, options, _, _, _, _ = setup(tmp_path, collect=True, collector_failure='finish')
    session = runner.run(prepared, options)
    result = session['results'][0]
    assert session['exit_code'] == 1 and result['generation']['status'] == 'completed'
    assert result['timing']['valid'] and not result['energy']['valid']
    assert result['energy']['total_energy_j'] is None
    assert result['energy']['prefix']['total_energy_j'] is not None


def test_metric_exception_cannot_skip_cache_and_gpu_cleanup(tmp_path, monkeypatch):
    import kvolt_runtime.telemetry as telemetry
    def broken(*args, **kwargs):
        raise RuntimeError('metric failed')
    monkeypatch.setattr(telemetry, 'timing_metrics', broken)
    runner, prepared, options, backend, events, _, _ = setup(tmp_path)
    session = runner.run(prepared, options)
    assert session['exit_code'] == 1
    assert all(cache.format is CacheFormat.RELEASED for cache in backend.caches)
    assert events[-2:] == ['release', 'restore']


def test_warmup_unverified_restore_prevents_request(tmp_path):
    runner, prepared, options, _, events, _, _ = setup(tmp_path, warmup=True)
    runner.device.restore = lambda snapshot: {'verified': False}
    session = runner.run(prepared, options)
    assert session['exit_code'] == 1 and session['counts']['not_run'] == 1
    assert ('prefill', 'request-0') not in events


def test_device_snapshot_storage_error_is_technical(tmp_path):
    runner, prepared, options, backend, _, _, _ = setup(tmp_path)
    save = runner.recorder.atomic_json
    def fail(path, value):
        if path == 'device.before.json':
            raise OSError('disk full')
        return save(path, value)
    runner.recorder.atomic_json = fail
    session = runner.run(prepared, options)
    assert session['exit_code'] == 1 and backend.loads == 0


def test_signal_inside_warmup_pack_skips_decode_and_restores(tmp_path):
    runner, prepared, options, backend, events, _, _ = setup(tmp_path, requests=2, warmup=True)
    pack = backend.pack_initial
    def interrupted_pack(cache, group):
        pack(cache, group)
        runner.stop.request(2)
    backend.pack_initial = interrupted_pack
    session = runner.run(prepared, options)
    assert session['exit_code'] == 130 and session['counts']['not_run'] == 2
    assert session['warmup']['interrupted'] is True
    assert len(session['warmup']['cases']) == 1
    case = session['warmup']['cases'][0]
    assert case['initial_packs'] == 1 and case['decode_forwards'] == 0
    assert case['skip_reason'] == 'interrupted'
    assert 'decode' not in events and events[-2:] == ['release', 'restore']
    assert all(cache.format is CacheFormat.RELEASED for cache in backend.caches)
    assert session['warmup']['restore']['verified'] is True


@pytest.mark.parametrize('warmup,phase', [(True, 'prepare'), (True, 'prefill'),
                                        (False, 'prepare'), (False, 'prefill'), (False, 'decode')])
@pytest.mark.parametrize('verified', [False, None, 1])
def test_unverified_prepare_apply_receipt_blocks_following_gpu_work(tmp_path, warmup, phase, verified):
    runner, prepared, options, backend, events, _, _ = setup(tmp_path, requests=2, warmup=warmup)
    if phase == 'prepare':
        runner.device.prepare_request = lambda: {'verified': verified}
    else:
        apply = runner.device.apply
        calls = []
        def unverified(frequency):
            calls.append(frequency)
            if phase == 'prefill' or len(calls) == 2:
                return {'verified': verified}
            return apply(frequency)
        runner.device.apply = unverified
    session = runner.run(prepared, options)
    assert session['exit_code'] == 1
    if warmup:
        assert session['counts']['not_run'] == 2 and backend.caches == []
        errors = session['errors'][0]['receipt']['errors']
        assert errors[0]['code'] == 'GPU_RECEIPT_UNVERIFIED'
    else:
        assert session['results'][0]['errors'][0]['code'] == 'GPU_RECEIPT_UNVERIFIED'
        assert session['results'][1]['generation']['status'] == 'not_run'
        assert len(backend.caches) == int(phase == 'decode')
    assert 'decode' not in events
    assert not any(isinstance(event, tuple) and event[0] == 'pack' for event in events)
    assert events[-1] == 'restore'


def test_decode_query_cost_is_after_budget_clock_but_inside_actual_decode_interval(tmp_path):
    runner, prepared, options, _, _, _, _ = setup(tmp_path)
    repo = runner.repo
    query = repo.decode_within_budget
    observed = []
    def delayed_query(length, groups, budget_s):
        observed.append((runner.clock.now, budget_s))
        runner.clock.now += 2_000_000_000
        return query(length, groups, budget_s)
    repo.decode_within_budget = delayed_query
    session = runner.run(prepared, options)
    assert session['exit_code'] == 0
    result = session['results'][0]
    decision, timing = result['decisions']['decode'], result['timing']
    assert len(observed) == 1 and observed[0][0] == decision['td_ns']
    assert observed[0][1] == decision['budget_s']
    assert decision['elapsed_ns'] == decision['td_ns'] - timing['t1_ns']
    assert decision['budget_s'] == decision['target_s'] - decision['elapsed_ns'] / 1e9 / 2
    assert timing['tN_ns'] - decision['td_ns'] >= 2_000_000_000
    assert timing['mean_tpot_s'] == (timing['tN_ns'] - timing['t1_ns']) / 2e9
    assert timing['mean_tpot_s'] >= 1
