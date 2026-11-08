"""CPU contracts for the explicitly experimental fixed Decode probe."""
from dataclasses import dataclass, replace
import signal
import unittest
from unittest.mock import patch

from kvolt_runtime.contracts import CacheFormat, CacheHandle, CacheMetadata, RequestSpec, TokenResult
from kvolt_runtime.execution.contracts import StopController
from tools.probe_fixed_decode import ARMS, case_schedule, run_case, validate_request


class Clock:
    def __init__(self):
        self.now = 10**18

    def __call__(self):
        self.now += 13
        return self.now


@dataclass
class Decision:
    selected: object
    target_s: float = 1.0


class Controller:
    def __init__(self, clock, *, group=64, frequency=918):
        self.clock, self.calls = clock, []
        self.decision = Decision(Selection(group, frequency))

    def choose(self, *args):
        self.calls.append(args)
        self.clock()
        return self.decision


@dataclass
class Selection:
    group: int
    frequency_mhz: int


class Device:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def capture(self):
        self.calls.append('capture')
        return {'original': True}

    def prepare_request(self):
        self.calls.append('prepare')
        return {'verified': True}

    def apply(self, frequency):
        self.calls.append(frequency)
        return {'verified': True, 'frequency': frequency}

    def restore(self, snapshot):
        self.calls.append('restore')
        if self.fail == 'restore':
            raise RuntimeError('restore failed')
        return {'verified': True}


class Backend:
    def __init__(self, clock, *, output=(1, 2, 128009), fail=None, stop=None):
        self.clock, self.output, self.fail, self.stop = clock, output, fail, stop
        self.calls, self.cache = [], None

    def prefill_raw(self, spec):
        self.calls.append('prefill')
        self.cache = CacheHandle(CacheMetadata(spec.request_id, spec.length, 'cuda:0', CacheFormat.RAW), object())
        return TokenResult(self.output[0], self.clock(), self.cache)

    def pack_initial(self, cache, group):
        self.calls.append(('pack', group))
        if self.stop:
            self.stop.request(int(signal.SIGTERM))
        if self.fail == 'pack':
            raise RuntimeError('pack failed')
        cache.commit_packed(replace(cache.metadata, format=CacheFormat.PACKED, group_size=group), object())

    def decode_one(self, cache, previous):
        self.calls.append('decode')
        if self.fail == 'decode':
            raise RuntimeError('decode failed')
        cache.commit_decode(replace(cache.metadata, length=cache.metadata.length+1), object())
        index = self.calls.count('decode')
        ready = self.clock()
        self.clock()  # host return bracket exceeds the actual token-ready boundary
        return TokenResult(self.output[index], ready, cache)

    def release_request(self, cache):
        self.calls.append('release')
        cache.release()
        if self.fail == 'release':
            raise RuntimeError('release failed')


class Collector:
    def __init__(self, clock, fail=None):
        self.clock, self.fail, self.calls = clock, fail, []

    def start(self):
        self.calls.append('start')
        self.left = self.clock()
        if self.fail == 'start':
            raise RuntimeError('collector start failed')
        return {'ready': True, 'errors': []}

    def finish(self, end):
        self.calls.append(('finish', end))
        if self.fail == 'finish':
            raise RuntimeError('collector finish failed')
        right = self.clock()
        from kvolt_runtime.telemetry import RAILS
        return {'samples': [{'rail': rail, 'timestamp_ns': t, 'power_w': 1.0}
                            for rail in RAILS for t in (self.left, right)],
                'profile': {'max_gap_ns': 10**9}, 'errors': [], 'raw_trace_jsonl': 'raw\n'}

    def stop(self):
        self.calls.append('stop')
        if self.fail == 'stop':
            raise RuntimeError('collector stop failed')
        return {'errors': []}


class FixedProbeTests(unittest.TestCase):
    def test_signal_at_token_return_preserves_terminal_completion_only(self):
        scenarios = (
            ('prefill_raw', 512, (128009,), True, 'eos', 1),
            ('prefill_raw', 1, (1,), True, 'length', 1),
            ('decode_one', 2, (1, 2), True, 'length', 2),
            ('decode_one', 512, (1, 128009), True, 'eos', 2),
            ('prefill_raw', 512, (1, 2, 128009), False, 'interrupted', 1),
            ('decode_one', 512, (1, 2, 128009), False, 'interrupted', 2),
        )
        for method, cap, output, completed, reason, count in scenarios:
            with self.subTest(method=method, cap=cap, output=output):
                clock, stop = Clock(), StopController()
                backend, device, collector = Backend(clock, output=output), Device(), Collector(clock)
                original = getattr(backend, method)

                def signal_at_return(*args):
                    token = original(*args)
                    stop.request(int(signal.SIGTERM))
                    return token

                with patch.object(backend, method, side_effect=signal_at_return):
                    row = run_case(RequestSpec('original', (1,)*8192, 'a'*64, cap), (128, 1122),
                        backend=backend, device=device, pc=Controller(clock), dc=Controller(clock),
                        collector=collector, clock=clock, stop=stop)
                self.assertEqual(row['status'], 'completed' if completed else 'interrupted')
                self.assertEqual(row['generation_completed'], completed)
                self.assertEqual(row['stop_reason'], reason)
                self.assertEqual(len(row['token_ids']), count)
                self.assertEqual(row['timing']['valid'], completed)
                self.assertEqual(row['energy']['valid'], completed)
                if not completed:
                    self.assertIsNone(row['timing']['mean_tpot_s'])
                    self.assertIsNotNone(row['timing']['prefix'])
                self.assertTrue(row['cleanup']['cache_released'])
                self.assertTrue(row['cleanup']['gpu_restored'])
                self.assertEqual(device.calls[-1], 'restore')
                self.assertTrue(stop.requested)

    def test_metric_exceptions_never_bypass_resource_cleanup(self):
        for helper in ('timing_metrics', 'energy_metrics'):
            with self.subTest(helper=helper):
                clock, stop = Clock(), StopController()
                backend, device, collector = Backend(clock), Device(), Collector(clock)
                spec = RequestSpec('original', (1,)*8192, 'a'*64, 512)
                with patch('kvolt_runtime.telemetry.'+helper, side_effect=RuntimeError('metric failed')):
                    row = run_case(spec, (128, 1122), backend=backend, device=device,
                        pc=Controller(clock), dc=Controller(clock), collector=collector,
                        clock=clock, stop=stop)
                self.assertEqual(row['status'], 'failed')
                self.assertIn('stop', collector.calls)
                self.assertIn('release', backend.calls)
                self.assertEqual(device.calls[-1], 'restore')

    def test_cleanup_only_failure_preserves_complete_energy_and_timing(self):
        for fail in ('release', 'restore'):
            with self.subTest(fail=fail):
                row, *_ = self.case(fail=fail)
                self.assertTrue(row['generation_completed'])
                self.assertTrue(row['timing']['valid'])
                self.assertTrue(row['energy']['valid'])
                self.assertIsNotNone(row['energy']['total_energy_j'])

    def test_energy_uses_same_token_boundaries_and_n_minus_one(self):
        row, *_ = self.case()
        timing, energy = row['timing'], row['energy']
        self.assertAlmostEqual(energy['decode_energy_j'], 3*timing['decode_duration_s'])
        self.assertAlmostEqual(energy['decode_j_per_token'], energy['decode_energy_j']/2)
        self.assertAlmostEqual(energy['total_energy_j'],
            energy['prefill_energy_j']+energy['decode_energy_j'])

    def test_signal_before_capture_stops_before_hardware_preparation(self):
        clock, stop = Clock(), StopController()
        stop.request(int(signal.SIGINT))
        backend, device, collector = Backend(clock), Device(), Collector(clock)
        row = run_case(RequestSpec('original', (1,)*8192, 'a'*64, 512), (128, 1122),
            backend=backend, device=device, pc=Controller(clock), dc=Controller(clock),
            collector=collector, clock=clock, stop=stop)
        self.assertEqual(row['status'], 'interrupted')
        self.assertEqual(device.calls, ['capture', 'restore'])
        self.assertFalse(backend.calls)
        self.assertNotIn('start', collector.calls)

    def case(self, *, cap=512, output=(1, 2, 128009), fail=None, signal_stop=False):
        clock, stop = Clock(), StopController()
        backend = Backend(clock, output=output, fail=fail, stop=stop if signal_stop else None)
        device = Device(fail)
        collector = Collector(clock, fail)
        pc, dc = Controller(clock, frequency=1122), Controller(clock)
        spec = RequestSpec('original', (1,)*8192, 'a'*64, cap)
        row = run_case(spec, (128, 1122), backend=backend, device=device, pc=pc, dc=dc,
                       collector=collector, clock=clock, stop=stop)
        return row, backend, device, collector, dc

    def test_normal_decision_is_distinct_from_actual_intervention(self):
        row, backend, device, collector, dc = self.case()
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(row['normal_decisions']['decode']['selected'], {'group': 64, 'frequency_mhz': 918})
        self.assertEqual(row['experimental_setting'], {'group': 128, 'frequency_mhz': 1122})
        self.assertIn(('pack', 128), backend.calls)
        self.assertEqual(device.calls, ['capture', 'prepare', 1122, 1122, 'restore'])
        self.assertEqual(row['stop_reason'], 'eos')
        self.assertEqual(row['final_cache_length'], 8194)
        self.assertTrue(row['energy']['valid'])

    def test_cap_two_stops_naturally_at_upper_bound(self):
        row, backend, *_ = self.case(cap=2)
        self.assertEqual(row['token_ids'], [1, 2])
        self.assertEqual(row['stop_reason'], 'length')
        self.assertEqual(backend.calls.count('decode'), 1)

    def test_first_eos_and_cap_one_skip_decode_and_packing(self):
        for cap, output in ((512, (128009,)), (1, (1,))):
            with self.subTest(cap=cap):
                row, backend, _, _, dc = self.case(cap=cap, output=output)
                self.assertFalse(dc.calls)
                self.assertNotIn('decode', backend.calls)
                self.assertFalse(any(isinstance(call, tuple) for call in backend.calls))
                self.assertIsNone(row['timing']['mean_tpot_s'])

    def test_token_boundaries_use_integer_subtraction_without_stage_addition(self):
        row, *_ = self.case()
        times = row['token_ready_ns']
        self.assertEqual(row['timing']['mean_tpot_s'], (times[-1]-times[0])/2/1e9)
        first_decode = next(stage for stage in row['stages'] if stage['component'] == 'decode_one')
        self.assertGreater(first_decode['end_ns'], times[1])
        self.assertEqual(row['timing']['tN_ns'], times[-1])

    def test_failures_preserve_prefix_and_independent_cleanup(self):
        for fail in ('pack', 'decode', 'start', 'finish', 'stop', 'release', 'restore'):
            with self.subTest(fail=fail):
                row, backend, device, collector, _ = self.case(fail=fail)
                self.assertEqual(row['status'], 'failed')
                self.assertTrue(row['errors'])
                self.assertEqual(device.calls[-1], 'restore')
                self.assertIn('stop', collector.calls)
                if backend.cache is not None:
                    self.assertEqual(backend.cache.format, CacheFormat.RELEASED)
                if fail in ('pack', 'decode', 'start'):
                    self.assertFalse(row['timing']['valid'])
                    self.assertIsNone(row['timing']['mean_tpot_s'])
                if fail == 'restore':
                    self.assertTrue(row['timing']['valid'])

    def test_signal_after_pack_preserves_incomplete_prefix_and_releases(self):
        row, backend, device, collector, _ = self.case(signal_stop=True)
        self.assertEqual(row['status'], 'interrupted')
        self.assertFalse(row['timing']['valid'])
        self.assertIsNone(row['timing']['mean_tpot_s'])
        self.assertEqual(row['token_ids'], [1])
        self.assertNotIn('decode', backend.calls)
        self.assertEqual(backend.cache.format, CacheFormat.RELEASED)
        self.assertEqual(device.calls[-1], 'restore')

    def test_schedule_and_frozen_request_validation(self):
        self.assertEqual(ARMS, ((64, 918), (64, 1122), (128, 1122)))
        schedule = case_schedule()
        self.assertEqual(schedule, [(arm, kind, cap) for arm in ARMS
                                    for kind, cap in (('warmup', 512), ('measured', 2), ('measured', 512))])
        spec = RequestSpec('m4-cap512-3', (1,)*8192, 'a'*64, 512)
        original = {'request_id': spec.request_id, 'length': 8192, 'nmax': 512,
                    'prompt_sha256': spec.prompt_sha256, 'token_ids': list(spec.token_ids),
                    'token_ids_sha256': spec.token_sha256}
        validate_request(spec, original)
        for key, value in (('length', 1024), ('nmax', 2), ('prompt_sha256', 'b'*64),
                           ('token_ids_sha256', 'b'*64), ('token_ids', [2]*8192)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(spec, {**original, key: value})
