import json
from pathlib import Path
import tempfile
import time
import signal
import unittest
from unittest.mock import Mock, patch

from scripts.eval_pilot.device import EXPECTED, configure, verify_environment, stop_worker_group, _live_group_members


class DeviceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        policy = self.root / "devices/system/cpu/cpufreq/policy0"
        gpu = self.root / "class/devfreq/17000000.gpu"
        for path in (policy, gpu):
            path.mkdir(parents=True)
        for name, value in {"affected_cpus": "0 1 2 3", "scaling_min_freq": "2201600",
                            "scaling_max_freq": "2201600"}.items():
            (policy / name).write_text(value)
        for name, value in {"governor": "nvhost_podgov", "min_freq": "306000000",
                            "max_freq": "1122000000"}.items():
            (gpu / name).write_text(value)
        self.monitor = self.root / "telemetry.jsonl"
        self.evidence = self.root / "device-evidence.json"
        self.evidence.write_text(json.dumps({"ready": True, "expected": EXPECTED,
                                           "monitor_path": str(self.monitor)}))
        self.sample()

    def sample(self, **changes):
        row = {"t_ns": time.monotonic_ns(), "emc": {"rate_hz": 3199000000, "lock": 1, "halt": 1}}
        row.update(changes)
        self.monitor.write_text(json.dumps(row) + "\n")

    def check(self):
        return verify_environment(self.evidence, sys_root=self.root)

    def test_expected_controls_and_fresh_root_telemetry_pass(self):
        self.assertTrue(self.check()["passed"])

    def test_gpu_stock_maximum_is_rejected(self):
        (self.root / "class/devfreq/17000000.gpu/max_freq").write_text("1300500000")
        self.assertFalse(self.check()["passed"])

    def test_unpinned_cpu_is_rejected(self):
        (self.root / "devices/system/cpu/cpufreq/policy0/scaling_min_freq").write_text("729600")
        self.assertFalse(self.check()["passed"])

    def test_stale_monitor_is_not_live_evidence(self):
        self.sample(t_ns=time.monotonic_ns() - 10_000_000_000)
        self.assertFalse(self.check()["passed"])

    def test_wrong_emc_is_rejected(self):
        self.sample(emc={"rate_hz": 1600000000, "lock": 1, "halt": 1})
        self.assertFalse(self.check()["passed"])

    def test_missing_privileged_evidence_does_not_pass(self):
        self.evidence.unlink()
        self.assertFalse(self.check()["passed"])

    def test_partial_monitor_tail_uses_last_complete_sample(self):
        with self.monitor.open("a") as stream:
            stream.write('{"t_ns":')
        self.assertTrue(self.check()["passed"])


class DevicePreparationTests(unittest.TestCase):
    def controller(self, root):
        controller = Mock()
        controller.sys_root = root
        controller.apply_maxn.return_value = {"status": "applied"}
        controller.prepare_emc_and_fan.return_value = {"emc_hz": 3199000000}
        controller.snapshot = {"boot_id": "test-boot"}
        controller.domains = [
            {"kind": "cpu", "path": "/cpu"},
            {"kind": "gpu", "path": "/gpu", "governor_name": "governor"}]
        controller.start_monitor.return_value.path = root / "telemetry.jsonl"
        controller.start_monitor.return_value.latest_sample.return_value = {}
        return controller

    def test_reboot_required_stops_before_clock_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.controller(Path(temporary))
            controller.apply_maxn.return_value = {"status": "reboot_required"}
            with self.assertRaisesRegex(RuntimeError, "reboot"):
                configure(controller)
            controller.prepare_emc_and_fan.assert_not_called()
            controller._set_range.assert_not_called()

    def test_different_emc_cap_stops_before_cpu_gpu_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.controller(Path(temporary))
            controller.prepare_emc_and_fan.return_value = {"emc_hz": 2040000000}
            with self.assertRaisesRegex(RuntimeError, "EMC cap"):
                configure(controller)
            controller._set_range.assert_not_called()
            controller.start_monitor.assert_not_called()

    def test_frozen_cpu_gpu_and_emc_are_applied(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.controller(Path(temporary))
            evidence = configure(controller)
            self.assertTrue(evidence["ready"])
            self.assertEqual(evidence["expected"], EXPECTED)
            calls = controller._set_range.call_args_list
            self.assertEqual([(c.args[0]["kind"], c.args[1:]) for c in calls],
                             [("cpu", (2201600, 2201600)), ("gpu", (306000000, 1122000000))])
            controller.start_monitor.assert_called_once_with(20)


class WorkerGroupTests(unittest.TestCase):
    def test_descendants_terminate_even_when_leader_already_exited(self):
        child = Mock(pid=123)
        child.poll.return_value = 3
        members = Mock(side_effect=[[456], []])
        current_time = [0.0]
        def sleep(seconds):
            current_time[0] += seconds
        with patch("scripts.eval_pilot.device.os.killpg") as kill:
            result = stop_worker_group(child, group_members=members, clock=lambda: current_time[0], sleep=sleep)
        kill.assert_called_once_with(123, signal.SIGTERM)
        self.assertTrue(result["passed"])
        self.assertEqual(result["leader_returncode"], 3)

    def test_group_ignoring_term_is_killed_with_bounded_wait(self):
        child = Mock(pid=123)
        child.poll.return_value = -9
        members = Mock(side_effect=[[456], []])
        with patch("scripts.eval_pilot.device.os.killpg") as kill:
            result = stop_worker_group(child, timeout_s=0, group_members=members)
        self.assertEqual([call.args[1] for call in kill.call_args_list], [signal.SIGTERM, signal.SIGKILL])
        self.assertTrue(result["passed"])

    def test_nonexistent_group_is_already_stopped(self):
        child = Mock(pid=123)
        child.poll.return_value = 0
        with patch("scripts.eval_pilot.device.os.killpg", side_effect=ProcessLookupError):
            result = stop_worker_group(child, group_members=lambda _: [])
        self.assertTrue(result["passed"])

    def test_zombies_are_not_running_group_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pid, state, group in ((11, "Z", 123), (12, "S", 123), (13, "S", 999)):
                directory = root / str(pid)
                directory.mkdir()
                (directory / "stat").write_text(f"{pid} (command with ) spaces) {state} 1 {group} 0 0\n")
            self.assertEqual(_live_group_members(123, proc_root=root), [12])


if __name__ == "__main__":
    unittest.main()
