"""Feasibility arithmetic and policy-scope checks, without GPU dependencies."""
from dataclasses import asdict, fields, make_dataclass, replace
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from kvolt_runtime.policy.types import Bracket, DecodePrediction, KneeResult
from tools import latest_tpot_audit as audit

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / 'data/luts/latest'
CONFIGURED_REFERENCE = os.environ.get('KVOLT_LUT_BUNDLE')
REFERENCE = (Path(CONFIGURED_REFERENCE) if CONFIGURED_REFERENCE is not None else
             ROOT.parent / 'KVolt-lut-source/data/luts/agx2-repeat5-20261103')


class FixedKnee:
    def detect(self, rows):
        return KneeResult('found', (8,))


class LatestFeasibilityTests(unittest.TestCase):
    def setUp(self):
        # Frozen historical counts remain regression evidence under the strict rule.
        self.buffer_patch = patch('kvolt_runtime.config.TPOT_BUFFER_RATIO', 0.0)
        self.buffer_patch.start()
        self.addCleanup(self.buffer_patch.stop)

    @classmethod
    def setUpClass(cls):
        if CONFIGURED_REFERENCE is not None:
            if not CONFIGURED_REFERENCE:
                raise ValueError('KVOLT_LUT_BUNDLE must name a nonempty sealed reference path')
            # An explicit reference is required evidence, never an optional skip.
            # Validate bytes before the dependency-based real-policy skip too.
            audit.load_repository(LATEST, REFERENCE)

    def assert_sparse_source_evidence(self, row, full):
        dc = row['decode']
        source = [asdict(prediction) for prediction in full]
        if dc['knee']['status'] == 'found':
            active = set(dc['knee']['groups'])
        elif dc['knee']['status'] == 'not_found':
            active = {32}
        else:
            active = {8, 16, 32, 64, 128}
        passing = [p for p in source if p['group'] in active and p['tpot_s'] <= dc['budget_s']]
        fastest = min(p['tpot_s'] for p in source)
        expected = passing or [p for p in source if p['tpot_s'] == fastest]
        self.assertEqual([c['prediction'] for c in dc['candidates']], expected)
        self.assertEqual(dc['all40_time_feasible_count'], sum(p['tpot_s'] <= dc['budget_s'] for p in source))
        self.assertEqual(dc['knee_active_count'], sum(p['group'] in active for p in source))
        self.assertEqual(dc['knee_active_time_feasible_count'], len(passing))
        self.assertEqual(sum(c['selected'] for c in dc['candidates']), 1)
        self.assertEqual([c['prediction'] for c in dc['candidates'] if c['selected']], [dc['selected']])
        for candidate in dc['candidates']:
            prediction = candidate['prediction']
            self.assertEqual(candidate['active_knee'], prediction['group'] in active)
            self.assertEqual(candidate['meets_budget'], prediction['tpot_s'] <= dc['budget_s'])
            self.assertEqual(candidate['eligible'], prediction in passing)

    def test_sparse_counts_cover_original_active_grid_in_all_knee_states(self):
        from kvolt_runtime.policy.mock import MockLUTRepository
        from kvolt_runtime.policy.types import PERFORMANCE_LENGTHS
        class Knee:
            def __init__(self, result): self.result = result
            def detect(self, rows): return self.result
        for knee in (KneeResult('found', (8,)), KneeResult('not_found', reason='flat_quality'),
                     KneeResult('error_recovered', reason='detector_error')):
            for target in (100.0, .0001):
                with self.subTest(knee=knee.status, target=target):
                    repo = MockLUTRepository(decode_target={n: target for n in PERFORMANCE_LENGTHS})
                    original = repo.decode
                    with patch.object(repo, 'decode', wraps=original) as full_query:
                        result = audit.policy_scenarios(repo, (0,), (512,), knee_adapter=Knee(knee))
                    self.assertEqual(full_query.call_count, 8)
                    for row in result:
                        self.assert_sparse_source_evidence(row, original(row['length']))

    def test_legacy_full_trace_counts_keep_recorded_active_groups(self):
        from kvolt_runtime.policy.decode import DecodeController
        from kvolt_runtime.policy.mock import MockLUTRepository
        from kvolt_runtime.policy.types import PERFORMANCE_LENGTHS
        repo = MockLUTRepository(decode_target={n: 100.0 for n in PERFORMANCE_LENGTHS})
        class Knee:
            def detect(self, rows): return KneeResult('not_found', reason='flat_quality')
        class LegacyController:
            def __init__(self, repo, knee_adapter, clock):
                self.controller = DecodeController(repo, knee_adapter=knee_adapter, clock=clock)
            def choose(self, length, *args):
                current = self.controller.choose(length, *args)
                full = repo.decode(length)
                scores = {q.group: q.score for q in current.quality}
                selected = min(full, key=lambda p: (p.energy_j_per_token, -scores[p.group], p.tpot_s, p.group, p.frequency_mhz))
                # Historical normal no-knee control used all G and recorded all40.
                candidates = tuple(replace(current.candidates[0], prediction=p, quality=scores[p.group],
                    active_knee=True, meets_budget=True, eligible=True, selected=p == selected,
                    reasons=() if p == selected else ('not_selected',)) for p in full)
                names = [f.name for f in fields(current) if f.name != 'selection_rule']
                legacy = make_dataclass('LegacyDecodeDecision', [(name, object) for name in names])
                values = {name: getattr(current, name) for name in names}
                values.update(candidates=candidates, selected=selected, selected_quality=scores[selected.group])
                return legacy(**values)
        with patch('kvolt_runtime.policy.decode.DecodeController', LegacyController):
            rows = audit.policy_scenarios(repo, (0,), (512,), knee_adapter=Knee())
        for row in rows:
            dc = row['decode']
            self.assertNotIn('selection_rule', dc)
            self.assertEqual(len(dc['candidates']), 40)
            self.assertEqual(dc['all40_time_feasible_count'], 40)
            self.assertEqual(dc['knee_active_count'], 40)
            self.assertEqual(dc['knee_active_time_feasible_count'], 40)

    def test_complete_latest_grid_against_unchanged_targets(self):
        if not REFERENCE.is_dir():
            self.skipTest('optional sealed reference checkout unavailable; set KVOLT_LUT_BUNDLE')
        report = audit.build_report(LATEST, REFERENCE, run_policy=False)
        self.assertEqual([r['all40']['time_feasible_count'] for r in report['lengths']],
                         [0, 0, 0, 0, 7, 10, 11, 12])
        self.assertEqual([r['length'] for r in report['lengths']], list(range(1024, 8193, 1024)))
        self.assertEqual(report['context']['prediction_source_device'], 'agx1')
        self.assertTrue(report['context']['device_mismatch'])
        self.assertFalse(report['context']['formal_qualification'])
        self.assertEqual(report['policy_execution']['status'], 'not_requested')
        self.assertEqual(report['lengths'][0]['all40']['minimum']['group'], 128)
        self.assertEqual(report['lengths'][0]['all40']['minimum']['frequency_mhz'], 1020)
        self.assertEqual(report['lengths'][-1]['all40']['minimum']['tpot_s'], 0.12509156071624267)
        self.assertEqual(set(report['inputs']['latest_files_sha256']),
                         {'prefill.json', 'decode.json', 'quality.json'})
        json.dumps(report, allow_nan=False)

    def test_agx2_matching_device_metadata_preserves_unvalidated_status(self):
        report = audit.build_report(ROOT / 'data/luts/agx2-runtime-repeat5', ROOT / 'data/luts/agx2-reference-repeat3', run_policy=False)
        self.assertEqual([r['all40']['time_feasible_count'] for r in report['lengths']],
                         [0, 0, 0, 0, 6, 10, 11, 12])
        self.assertEqual(report['context']['prediction_source_device'], 'agx2')
        self.assertEqual(report['context']['target_source_device'], 'AGX2')
        self.assertFalse(report['context']['device_mismatch'])
        self.assertFalse(report['context']['formal_qualification'])
        self.assertFalse(report['context']['runtime_accuracy_validated'])
        self.assertFalse(report['context']['actual_slo_validated'])
        provenance = report['inputs']['provenance']
        metadata = json.loads(provenance['metadata_json'])
        self.assertFalse(metadata['compact_manifest']['validation']['quality_runtime_eligible'])
        self.assertEqual(metadata['compact_manifest']['context']['prefill_aggregation']['samples_per_cell'], 25)
        self.assertEqual(provenance['manifest_sha256'],
                         'b421fcd10ecb57df7d6962aa9a5f48e264e90defc0c76a3544218d31685f783b')

    def test_comparison_does_not_round_at_display_precision(self):
        target = .1
        row = DecodePrediction(1024, 8, 408, target + 1e-12, 1, Bracket(1024, 1024, 0))
        summary = audit.summarize_decode((row,), target)
        self.assertEqual(summary['time_feasible_count'], 0)
        self.assertLess(summary['minimum_margin_ms'], 0)
        equality = audit.summarize_decode((replace(row, tpot_s=target),), target)
        self.assertEqual(equality['time_feasible_count'], 1)

    def test_policy_distinguishes_global_feasibility_from_knee_restriction(self):
        if not REFERENCE.is_dir():
            self.skipTest('optional sealed reference checkout unavailable; set KVOLT_LUT_BUNDLE')
        repo = audit.load_repository(LATEST, REFERENCE)
        result = audit.policy_scenarios(repo, (0, 2_000_000), (2, 128, 512), knee_adapter=FixedKnee())
        self.assertEqual(len(result), 48)
        long = next(r for r in result if r['length'] == 7168 and r['nmax'] == 512 and r['assumed_elapsed_ns'] == 0)
        self.assertEqual(long['decode']['all40_time_feasible_count'], 11)
        self.assertEqual(long['decode']['knee_active_time_feasible_count'], 0)
        self.assertEqual(long['decode']['knee_active_count'], 8)
        self.assert_sparse_source_evidence(long, repo.decode(long['length']))
        self.assertEqual(long['decode']['knee']['groups'], [8])
        self.assertEqual(long['decode']['selection_path'], 'latency_fallback')
        # Fallback searches all40: feasibility can be true despite empty Knee candidates.
        self.assertTrue(long['decode']['predicted_feasible'])
        self.assertEqual(long['decode']['selected']['group'], 128)
        short = next(r for r in result if r['length'] == 1024 and r['nmax'] == 2 and r['assumed_elapsed_ns'] == 2_000_000)
        self.assertEqual(short['decode']['budget_s'], short['decode']['target_s'] - .002)
        self.assertEqual(short['decode']['elapsed_ns'], 2_000_000)
        self.assertFalse(short['decode']['predicted_feasible'])
        self.assertEqual(long['prefill']['elapsed_ns'], 0)

    def test_missing_pinned_policy_is_visible_and_never_synthetic_success(self):
        if not REFERENCE.is_dir():
            self.skipTest('optional sealed reference checkout unavailable; set KVOLT_LUT_BUNDLE')
        from kvolt_runtime.policy.types import PolicyDependencyError
        with patch.object(audit, 'make_knee_adapter', side_effect=PolicyDependencyError('missing pins')):
            report = audit.build_report(LATEST, REFERENCE)
        self.assertEqual(report['policy_execution']['status'], 'unavailable')
        self.assertIn('missing pins', report['policy_execution']['error'])
        self.assertEqual(report['policy_scenarios'], [])
        self.assertEqual(len(report['lengths']), 8)

    def test_real_pinned_policy_if_available(self):
        from kvolt_runtime.policy.types import PolicyDependencyError
        if not REFERENCE.is_dir():
            self.skipTest('optional sealed reference checkout unavailable; set KVOLT_LUT_BUNDLE')
        try:
            audit.make_knee_adapter()
        except PolicyDependencyError as error:
            self.skipTest(str(error))
        report = audit.build_report(LATEST, REFERENCE)
        self.assertEqual(report['policy_execution']['status'], 'completed')
        self.assertEqual(len(report['policy_scenarios']), 48)
        repo = audit.load_repository(LATEST, REFERENCE)
        full = {length: repo.decode(length) for length in audit.LENGTHS}
        for row in report['policy_scenarios']:
            dc = row['decode']
            self.assert_sparse_source_evidence(row, full[row['length']])
            self.assertEqual(dc['predicted_feasible'], dc['selected']['tpot_s'] <= dc['budget_s'])
            self.assertEqual(dc['elapsed_ns'], row['assumed_elapsed_ns'])

    def test_historical_bundle_cannot_be_mislabeled_latest(self):
        if not REFERENCE.is_dir():
            self.skipTest('optional sealed reference checkout unavailable; set KVOLT_LUT_BUNDLE')
        with self.assertRaises(ValueError):
            audit.load_repository(REFERENCE, REFERENCE)

    def test_invalid_scenario_inputs_rejected(self):
        for elapsed, caps in [((-1,), (512,)), ((True,), (512,)), ((0,), (1,)), ((0,), (513,)), ((), (512,))]:
            with self.subTest(elapsed=elapsed, caps=caps), self.assertRaises(ValueError):
                audit.validate_scenarios(elapsed, caps)


if __name__ == '__main__':
    unittest.main()
