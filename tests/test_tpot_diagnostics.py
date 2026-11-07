"""Independent CPU checks of TPOT decomposition and archived r4 evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from tools import diagnose_tpot


ROOT = Path(__file__).resolve().parents[1]
R4 = ROOT / "tests/fixtures/tpot-r4.json"


def fixture(times=(1_000_000_000, 1_300_000_000, 1_400_000_000, 1_600_000_000)):
    return {
        "request": {"request_id": "synthetic", "length": 1024, "nmax": 512},
        "generation": {"status": "completed", "n_output_tokens": len(times), "token_ready_ns": list(times)},
        "decisions": {"decode": {
            "target_s": 0.2, "budget_s": 0.19,
            "t1_ns": times[0], "td_ns": times[0] + 10_000_000,
            "elapsed_ns": 10_000_000,
            "selection_path": "latency_fallback",
            "selected": {"group": 8, "frequency_mhz": 1122, "tpot_s": 0.21},
            "candidates": [
                {"eligible": False, "prediction": {"group": 8, "frequency_mhz": 1122, "tpot_s": 0.21}},
                {"eligible": False, "prediction": {"group": 128, "frequency_mhz": 1122, "tpot_s": 0.22}},
            ],
        }},
    }


class TpotArithmeticTests(unittest.TestCase):
    def test_first_and_steady_gaps_reconstruct_full_decode(self):
        row = diagnose_tpot.analyze_result(fixture())
        self.assertEqual(row["status"], "valid", row["errors"])
        self.assertAlmostEqual(row["first_intertoken_s"], 0.3)
        self.assertEqual(row["steady_gap_count"], 2)
        self.assertAlmostEqual(row["steady_mean_s"], 0.15)
        self.assertAlmostEqual(row["steady_min_s"], 0.1)
        self.assertAlmostEqual(row["steady_max_s"], 0.2)
        self.assertAlmostEqual(row["decode_duration_s"], 0.6)
        self.assertAlmostEqual(row["mean_tpot_s"], 0.2)
        self.assertAlmostEqual(row["first_intertoken_s"] + row["steady_gap_count"] * row["steady_mean_s"], row["decode_duration_s"])
        self.assertTrue(row["actual_slo_met"])
        self.assertEqual(row["selection_path"], "latency_fallback")
        self.assertFalse(row["actual_slo_met"] is False, "Fallback is not proof of actual SLO violation")
        self.assertEqual(row["selected_prediction"]["tpot_s"], 0.21)
        self.assertEqual(row["candidate_count"], 2)
        self.assertEqual(row["eligible_count"], 0)
        self.assertAlmostEqual(row["dc_elapsed_s"], 0.01)

    def test_n_one_has_no_decode_or_steady_statistics(self):
        row = diagnose_tpot.analyze_result(fixture((100,)))
        self.assertEqual(row["status"], "valid", row["errors"])
        self.assertEqual(row["steady_gap_count"], 0)
        for key in ("first_intertoken_s", "steady_mean_s", "steady_min_s", "steady_max_s", "decode_duration_s", "mean_tpot_s", "actual_slo_met"):
            self.assertIsNone(row[key], key)

    def test_n_two_has_one_first_gap_and_no_steady_sample(self):
        row = diagnose_tpot.analyze_result(fixture((1_000_000_000, 1_500_000_000)))
        self.assertEqual(row["status"], "valid", row["errors"])
        self.assertEqual(row["steady_gap_count"], 0)
        self.assertAlmostEqual(row["first_intertoken_s"], 0.5)
        self.assertAlmostEqual(row["mean_tpot_s"], 0.5)
        for key in ("steady_mean_s", "steady_min_s", "steady_max_s"):
            self.assertIsNone(row[key], key)
        self.assertFalse(row["actual_slo_met"])

    def test_large_monotonic_origin_preserves_integer_nanosecond_difference(self):
        origin = 9_007_199_254_740_992
        row = diagnose_tpot.analyze_result(fixture((origin, origin + 1, origin + 4)))
        self.assertEqual(row["status"], "valid", row["errors"])
        self.assertEqual(row["first_intertoken_s"], 1e-9)
        self.assertEqual(row["steady_mean_s"], 3e-9)
        self.assertEqual(row["mean_tpot_s"], 2e-9)

    def test_invalid_token_timestamps_never_create_computed_latency(self):
        for times in ([], [100, 100], [100, 99], [-1, 10], [False, 1], [1.0, 2.0], ["1", 2], [None, 2]):
            with self.subTest(times=times):
                raw = fixture()
                raw["generation"]["token_ready_ns"] = times
                raw["generation"]["n_output_tokens"] = len(times)
                row = diagnose_tpot.analyze_result(raw)
                self.assertEqual(row["status"], "invalid")
                self.assertTrue(row["errors"])
                for key in ("mean_tpot_s", "decode_duration_s", "first_intertoken_s", "actual_slo_met"):
                    self.assertIsNone(row[key], key)

    def test_missing_timestamps_and_mismatched_count_are_invalid(self):
        for change in ("missing", "count", "boolean_count", "float_count"):
            with self.subTest(change=change):
                raw = fixture()
                if change == "missing":
                    del raw["generation"]["token_ready_ns"]
                else:
                    raw["generation"]["n_output_tokens"] = {"count": 3, "boolean_count": True, "float_count": 4.0}[change]
                row = diagnose_tpot.analyze_result(raw)
                self.assertEqual(row["status"], "invalid")
                self.assertTrue(row["errors"])

    def test_duplicate_timing_timestamps_must_match_generation(self):
        for field, value in (("token_ready_ns", [1, 2, 3, 4]), ("t1_ns", 999), ("tN_ns", 999), ("n_output_tokens", 2)):
            with self.subTest(field=field):
                raw = fixture()
                raw["timing"] = {field: value}
                row = diagnose_tpot.analyze_result(raw)
                self.assertEqual(row["status"], "invalid")
                self.assertTrue(row["errors"])

    def test_duplicate_timing_cannot_weaken_integer_timestamp_contract(self):
        for timing in ({"t1_ns": 1_000_000_000.0}, {"tN_ns": 1_600_000_000.0}, {"token_ready_ns": [1e9, 1.3e9, 1.4e9, 1.6e9]}):
            with self.subTest(timing=timing):
                raw = fixture()
                raw["timing"] = timing
                row = diagnose_tpot.analyze_result(raw)
                self.assertEqual(row["status"], "invalid")
                self.assertIsNone(row["mean_tpot_s"])

    def test_first_gap_partition_keeps_uninstrumented_work_honest(self):
        raw = fixture()
        row = diagnose_tpot.analyze_result(raw)
        self.assertIsNone(row["all40_min_prediction"])
        self.assertEqual(row["candidate_coverage"], "incomplete")
        part = row["first_gap_partition"]
        self.assertAlmostEqual(part["dc_preselection_s"], 0.01)
        self.assertIsNone(part["initial_pack_s"])
        self.assertIsNone(part["first_decode_s"])
        # An absent device receipt cannot be assumed to consume zero time.
        self.assertGreaterEqual(part["uninstrumented_s"], 0.0)

    def test_optional_stage_times_partition_without_double_counting(self):
        raw = fixture()
        raw["device"] = {"decode": {"start_ns": 1_010_000_000, "end_ns": 1_030_000_000, "verified": True}}
        raw["generation"]["stage_timing"] = {
            "initial_pack": {"start_ns": 1_030_000_000, "end_ns": 1_080_000_000},
            "first_decode": {"start_ns": 1_080_000_000, "end_ns": 1_280_000_000},
        }
        row = diagnose_tpot.analyze_result(raw)
        self.assertEqual(row["status"], "valid", row["errors"])
        part = row["first_gap_partition"]
        self.assertAlmostEqual(part["dc_preselection_s"], 0.01)
        self.assertAlmostEqual(part["device_apply_s"], 0.02)
        self.assertAlmostEqual(part["initial_pack_s"], 0.05)
        self.assertAlmostEqual(part["first_decode_s"], 0.2)
        self.assertAlmostEqual(part["uninstrumented_s"], 0.02)
        self.assertAlmostEqual(sum(part[key] for key in ("dc_preselection_s", "device_apply_s", "initial_pack_s", "first_decode_s", "uninstrumented_s")), 0.3)

    def test_host_return_after_token_ready_clips_only_partition(self):
        raw = fixture()
        stage = {"initial_pack": None, "first_decode": {"start_ns": 1_100_000_000, "end_ns": 1_305_000_000}}
        raw["generation"]["stage_timing"] = copy.deepcopy(stage)
        row = diagnose_tpot.analyze_result(raw)
        self.assertEqual(row["status"], "valid", row["errors"])
        self.assertEqual(row["stage_timing"], stage)
        self.assertAlmostEqual(row["first_gap_partition"]["first_decode_s"], 0.2)
        self.assertAlmostEqual(row["mean_tpot_s"], 0.2)
        self.assertIn("first_decode_span_clipped_to_token_ready", row["warnings"])

    def test_failed_generation_cannot_publish_full_request_tpot(self):
        raw = fixture()
        raw["generation"]["status"] = "failed"
        row = diagnose_tpot.analyze_result(raw)
        self.assertEqual(row["status"], "invalid")
        self.assertTrue(row["errors"])
        self.assertIsNone(row["mean_tpot_s"])
        self.assertIsNone(row["actual_slo_met"])

    def test_analysis_does_not_mutate_raw_record(self):
        raw = fixture()
        original = copy.deepcopy(raw)
        diagnose_tpot.analyze_result(raw)
        self.assertEqual(raw, original)


class ArchivedR4Tests(unittest.TestCase):
    def test_three_archived_sessions_recompute_21_decode_values_and_19_violations(self):
        totals = {key: 0 for key in ("request_count", "valid_count", "invalid_count", "decode_evaluable_count", "actual_tpot_violation_count")}
        fixture = json.loads(R4.read_text())
        self.assertEqual(fixture['source_commit'], '8fae164')
        for name in ("run-first", "run-repeat", "run-performance"):
            session = fixture['sessions'][name]
            analyzed = diagnose_tpot.analyze_session(session, source=name)
            self.assertEqual(len(analyzed["results"]), 8)
            for key in totals:
                totals[key] += analyzed["summary"][key]
            for raw, row in zip(session["results"], analyzed["results"]):
                with self.subTest(session=name, request=raw["request"]["request_id"]):
                    self.assertEqual(row["status"], "valid", row["errors"])
                    self.assertEqual(row["actual_slo_met"], raw["slo"]["mean_tpot_met"])
                    times = raw["generation"]["token_ready_ns"]
                    if len(times) == 1:
                        self.assertIsNone(row["mean_tpot_s"])
                        continue
                    expected = (times[-1] - times[0]) / 1_000_000_000 / (len(times) - 1)
                    self.assertAlmostEqual(row["mean_tpot_s"], expected, places=14)
                    self.assertAlmostEqual(row["mean_tpot_s"], raw["timing"]["mean_tpot_s"], places=14)
                    self.assertAlmostEqual(row["first_intertoken_s"], (times[1] - times[0]) / 1_000_000_000)
                    self.assertEqual(row["steady_gap_count"], len(times) - 2)
                    if len(times) > 2:
                        self.assertAlmostEqual(row["steady_mean_s"], (times[-1] - times[1]) / 1_000_000_000 / (len(times) - 2))
                    self.assertEqual(row["selection_path"], raw["decisions"]["decode"]["selection_path"])
                    self.assertEqual(row["selected_prediction"], raw["decisions"]["decode"]["selected"])
                    self.assertEqual(row["candidate_count"], 40)
            self.assertEqual(analyzed["summary"]["actual_tpot_violation_count"], 7 if name == "run-first" else 6)
        self.assertEqual(totals, {"request_count": 24, "valid_count": 24, "invalid_count": 0, "decode_evaluable_count": 21, "actual_tpot_violation_count": 19})

    def test_help_runs_without_gpu_dependencies(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/diagnose_tpot.py"), "--help"], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("session", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
