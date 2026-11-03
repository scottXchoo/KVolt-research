"""CPU protocol tests; no model weights, sensors or CUDA are required."""
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.eval_pilot import runner


class FakeBackend:
    name = "cpu_protocol_fake"

    def __init__(self, tokens=(11, 128009), fail_at=None):
        self.tokens, self.fail_at = list(tokens), fail_at
        self.forwards = 0
        self.events = []

    def inference_mode(self): return nullcontext()
    def synchronize(self): self.events.append("synchronize")
    def prepare_input(self, ids):
        self.events.append("prepare_input")
        return ids
    def begin_finite_check(self): return True
    def forward(self, model, tokens, cache):
        self.forwards += 1
        if self.forwards == self.fail_at:
            raise RuntimeError("injected forward failure")
        return SimpleNamespace(past_key_values=(len(tokens) if cache is None else cache + 1))
    def select(self, output, state, suppressed=()):
        self.events.append("select")
        return [self.tokens[(self.forwards - 1) % len(self.tokens)]], self.tokens[(self.forwards - 1) % len(self.tokens)], state
    def finish_finite_check(self, state): return state
    def cache_length(self, cache, method): return cache


def handle(backend):
    return SimpleNamespace(backend=backend, active_request=False, released=False,
                           model=object(), method="fp16", precision="fp16", group=None)


def apply(fake_handle, condition):
    fake_handle.method = condition["method"]
    fake_handle.precision = condition["name"]
    return [{"layer": i, "k_bits": condition["k_bits"], "v_bits": condition["v_bits"],
             "group": condition["group"], "residual_length": condition["residual_length"]}
            for i in range(32)]


ITEM = {"id": "one", "input_ids": [1, 2, 3], "length": 3, "answers": ["a"]}


class FakeClock:
    def __init__(self): self.now = 0
    def monotonic_ns(self): return self.now
    def sleep(self, seconds): self.now += int(seconds * 1e9)


class AdmissionTests(unittest.TestCase):
    def test_paired_blocks_preserve_all_180_requests_and_deterministic_order(self):
        items = [{"id": str(i)} for i in range(30)]
        schedule = runner.paired_schedule(items)
        self.assertEqual(schedule, runner.paired_schedule(items))
        self.assertNotEqual(schedule, runner.paired_schedule(items, seed=20261104))
        self.assertEqual(len(schedule), 180)
        self.assertEqual(len({(item["id"], condition["name"], condition["group"])
                              for item, condition in schedule}), 180)
        for offset in range(0, 180, 6):
            block = schedule[offset:offset + 6]
            self.assertEqual(len({item["id"] for item, _ in block}), 1)
            self.assertEqual({runner.canonical(c) for _, c in block},
                             {runner.canonical(c) for c in runner.CONDITIONS})
        self.assertGreater(len({tuple(c["group"] for _, c in schedule[i:i + 6])
                                for i in range(0, 180, 6)}), 1)

    def test_temperature_cools_to_inclusive_band_without_request_time(self):
        clock = FakeClock()
        temperatures = iter((56000, 55000))
        def reader():
            value = next(temperatures)
            return {"t_ns": clock.now, "temperatures_mc": {"cpu-thermal": value, "gpu-thermal": 35000}}
        result = runner.wait_temperature(reader, clock_ns=clock.monotonic_ns, sleep=clock.sleep)
        self.assertTrue(result["passed"])
        self.assertEqual(result["wait_s"], 1)
        self.assertFalse(result["forced_heating"])
        self.assertEqual(len(result["observations"]), 2)

    def test_cold_idle_allowed_only_for_warmup_and_formal_times_out(self):
        clock = FakeClock()
        reader = lambda: {"t_ns": clock.now, "temperatures_mc": {"cpu-thermal": 34000, "gpu-thermal": 34000}}
        warmup = runner.wait_temperature(reader, minimum_mc=None, clock_ns=clock.monotonic_ns, sleep=clock.sleep)
        self.assertTrue(warmup["passed"])
        formal = runner.wait_temperature(reader, timeout_s=2, clock_ns=clock.monotonic_ns, sleep=clock.sleep)
        self.assertFalse(formal["passed"])
        self.assertEqual(formal["reason"], "temperature_admission_timeout")
        self.assertEqual(formal["wait_s"], 2)

    def test_missing_or_stale_temperature_cannot_admit(self):
        for sample in ({"t_ns": -3_000_000_000, "temperatures_mc": {"cpu-thermal": 40000, "gpu-thermal": 40000}},
                       {"t_ns": 0, "temperatures_mc": {"cpu-thermal": 40000}}):
            clock = FakeClock()
            result = runner.wait_temperature(lambda: sample, timeout_s=0,
                                            clock_ns=clock.monotonic_ns, sleep=clock.sleep)
            self.assertFalse(result["passed"])


class GenerationTests(unittest.TestCase):
    def run_request(self, backend, **kwargs):
        return runner.generate_request(handle(backend), ITEM, runner.CONDITIONS[0],
                                       apply_condition=apply, **kwargs)

    def test_natural_eos_and_gpu_completed_boundaries(self):
        backend = FakeBackend()
        row = self.run_request(backend)
        self.assertEqual(row["output_ids"], [11, 128009])
        self.assertEqual(row["termination_reason"], "eos")
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["decode_forward_count"], 1)
        self.assertEqual(row["final_cache_length"], 4)
        self.assertTrue(row["generation_valid"])
        self.assertEqual(backend.events[backend.events.index("select") + 1], "synchronize")
        self.assertLess(row["t0_ns"], row["t1_ns"])
        self.assertGreater(row["tN_ns"], row["t1_ns"])

    def test_first_eos_preserved_and_decode_metrics_null(self):
        row = self.run_request(FakeBackend(tokens=(128009,)))
        self.assertEqual(row["n"], 1)
        self.assertEqual(row["t1_ns"], row["tN_ns"])
        self.assertIsNone(runner.latencies(row)["tpot_s"])
        self.assertIsNone(runner.latencies(row)["decode_s"])
        self.assertEqual(row["decode_forward_count"], 0)

    def test_cap_includes_first_token(self):
        row = self.run_request(FakeBackend(tokens=(11,)), max_new_tokens=3)
        self.assertEqual(row["n"], 3)
        self.assertEqual(row["termination_reason"], "max_new_tokens")
        self.assertEqual(row["decode_forward_count"], 2)

    def test_warmup_also_preserves_natural_eos(self):
        row = self.run_request(FakeBackend(tokens=(128009,)), warmup=True)
        self.assertEqual(row["n"], 1)
        self.assertEqual(row["termination_reason"], "eos")
        self.assertFalse(row["eos_suppressed"])
        self.assertEqual(row["purpose"], "warmup")

    def test_partial_failure_retains_first_token_and_releases_handle(self):
        model = handle(FakeBackend(fail_at=2))
        with self.assertRaises(runner.GenerationError) as caught:
            runner.generate_request(model, ITEM, runner.CONDITIONS[0], apply_condition=apply)
        self.assertEqual(caught.exception.partial_result["output_ids"], [11])
        self.assertEqual(caught.exception.partial_result["error"]["message"], "injected forward failure")
        self.assertFalse(model.active_request)

    def test_wrong_layer_bits_fail_without_forward(self):
        def bad_apply(model, condition):
            rows = apply(model, condition)
            rows[31]["k_bits"] = 4
            return rows
        backend = FakeBackend()
        with self.assertRaises(runner.GenerationError):
            runner.generate_request(handle(backend), ITEM, runner.CONDITIONS[1], apply_condition=bad_apply)
        self.assertEqual(backend.forwards, 0)

    def test_clock_mismatch_stops_before_gpu_input(self):
        backend = FakeBackend()
        with self.assertRaises(runner.GenerationError):
            self.run_request(backend, verify_device=lambda: {"passed": False})
        self.assertNotIn("prepare_input", backend.events)


class PersistenceTests(unittest.TestCase):
    def test_resume_skips_only_technical_success_and_retains_attempts(self):
        with TemporaryDirectory() as temp:
            journal = runner.AttemptJournal(Path(temp) / "requests.jsonl")
            first = journal.start("a", {"formal": True})
            journal.finish({"request_key": "a", "attempt": first, "technical_valid": False})
            self.assertFalse(journal.completed("a"))
            self.assertEqual(journal.start("a", {}), 2)
            resumed = runner.AttemptJournal(journal.path)
            self.assertEqual(resumed.start("a", {}), 3)
            resumed.finish({"request_key": "a", "attempt": 3, "technical_valid": True, "score": 0})
            self.assertTrue(runner.AttemptJournal(journal.path).completed("a"))
            self.assertEqual(len(journal.path.read_text().splitlines()), 5)

    def test_corrupt_journal_is_not_silently_overwritten(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "requests.jsonl"
            path.write_text('{"type":"attempt_start"}\npartial')
            with self.assertRaises(ValueError): runner.AttemptJournal(path)
            self.assertTrue(path.read_text().endswith("partial"))


class WarmupTests(unittest.TestCase):
    def test_unsupported_jit_fails_after_one_preserved_pass(self):
        with TemporaryDirectory() as temp:
            row = {"generation_valid": True, "n": 1, "jit": {"observation_supported": False},
                   "warm_timing_valid": False}
            with patch.object(runner, "generate_request", return_value=row) as generation:
                with self.assertRaises(RuntimeError):
                    runner._warmup(None, ITEM, runner.CONDITIONS[0], Path(temp), None)
            self.assertEqual(generation.call_count, 1)
            self.assertTrue((Path(temp) / "pass-1.json").exists())
            self.assertFalse((Path(temp) / "pass-2.json").exists())

    def test_second_warmup_only_for_cache_miss(self):
        with TemporaryDirectory() as temp:
            first = {"generation_valid": True, "n": 3,
                     "jit": {"observation_supported": True, "cache_misses": ["kernel"]},
                     "warm_timing_valid": False}
            second = {"generation_valid": True, "n": 3,
                      "jit": {"observation_supported": True, "cache_misses": []},
                      "warm_timing_valid": True}
            with patch.object(runner, "generate_request", side_effect=[first, second]) as generation:
                proof = runner._warmup(None, ITEM, runner.CONDITIONS[0], Path(temp), None)
            self.assertEqual(generation.call_count, 2)
            self.assertEqual(proof["actual_output_tokens"], [3, 3])
            self.assertTrue(proof["natural_eos"])


class PowerTests(unittest.TestCase):
    def test_post_token_guard_waits_for_every_rail_midpoint(self):
        clock = FakeClock()
        def status(stamps):
            return {"rails": {rail: {"valid": True, "t_ns": stamp}
                              for rail, stamp in zip(runner.RAILS, stamps)}}
        logger = SimpleNamespace(status=unittest.mock.Mock(side_effect=[
            status([99, 99, 99]), status([101, 101, 99]), status([101, 101, 100])]))
        guard = runner.wait_power_coverage(logger, 100, clock_ns=clock.monotonic_ns, sleep=clock.sleep)
        self.assertTrue(guard["passed"])
        self.assertEqual(guard["polls"], 3)
        self.assertAlmostEqual(guard["wait_s"], 0.04)

    def test_post_token_guard_timeout_preserves_last_sensor_status(self):
        clock = FakeClock()
        stale = {"rails": {rail: {"valid": True, "t_ns": 99} for rail in runner.RAILS}}
        logger = SimpleNamespace(status=lambda: stale)
        guard = runner.wait_power_coverage(logger, 100, timeout_s=2, poll_s=1,
                                          clock_ns=clock.monotonic_ns, sleep=clock.sleep)
        self.assertFalse(guard["passed"])
        self.assertEqual(guard["wait_s"], 2)
        self.assertEqual(guard["reason"], "post_tN_power_coverage_timeout")
        self.assertEqual(guard["last_status"], stale)

    def test_independent_lut_rails_sum_and_n1_decode_null(self):
        def series(rows, rail): return {"points": [{"t_ns": 0}, {"t_ns": 3_000_000_000}]}
        def integrate(value, boundaries, rules):
            return {"status": "valid", "rails": {rail: {"request_j": 2, "prefill_j": 2,
                        "decode_j": 0} for rail in runner.RAILS},
                    "request_j": 6, "prefill_j": 6, "decode_j": 0}
        row = {"t0_ns": 1_000_000_000, "t1_ns": 2_000_000_000,
               "tN_ns": 2_000_000_000, "n": 1}
        energy = runner.measure_energy([], row, series_builder=series, integrator=integrate)
        self.assertEqual(len(energy["rails"]), 3)
        self.assertEqual(energy["request_j"], 6)
        self.assertEqual(energy["primary_rails"], list(runner.RAILS))
        self.assertIsNone(energy["decode_j"])
        self.assertIsNone(energy["decode_j_per_token"])
        self.assertTrue(energy["rail_sum_used"])
        self.assertFalse(energy["overlapping_aggregate_included"])

    def test_bad_coverage_fails_energy_without_filtering_generation(self):
        def broken(*args): raise ValueError("Power samples must bracket both requested boundaries.")
        energy = runner.measure_energy([], {"t0_ns": 1, "t1_ns": 2, "tN_ns": 3, "n": 2},
                                      series_builder=lambda rows, rail: {}, integrator=broken)
        self.assertEqual(energy["status"], "invalid")
        self.assertIsNone(energy["request_j"])
        self.assertEqual(len(energy["reasons"]), 1)


if __name__ == "__main__": unittest.main()
