"""Scoped AGX2 preparation, live readback, and restoration for the pilot.

Only ``run`` needs sudo. The GPU worker is started as the original non-root
user; the existing project HardwareController owns all reversible changes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import time

EXPECTED = {"cpu_khz": 2201600, "emc_hz": 3199000000,
            "gpu_governor": "nvhost_podgov", "gpu_min_hz": 306000000,
            "gpu_max_hz": 1122000000}


def _read(path):
    return Path(path).read_text().strip()


def _last_complete_sample(path):
    with Path(path).open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 65536))
        data = stream.read()
    lines = data.split(b"\n")[:-1]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    raise ValueError("No complete root-monitor sample")


def verify_environment(evidence=None, *, sys_root="/sys", max_age_s=2.0):
    """Verify readable controls and fresh privileged EMC telemetry, no writes."""
    root = Path(sys_root)
    result = {"passed": False, "expected": dict(EXPECTED), "cpu": {}, "gpu": {}, "errors": []}
    try:
        proof = json.loads(Path(evidence).read_text()) if not isinstance(evidence, dict) else evidence
        if not proof or proof.get("ready") is not True or proof.get("expected") != EXPECTED:
            raise ValueError("Missing or mismatched device evidence")
        policies = sorted((root / "devices/system/cpu/cpufreq").glob("policy*"))
        for path in policies:
            if not _read(path / "affected_cpus"):
                continue
            observed = {"min": int(_read(path / "scaling_min_freq")),
                        "max": int(_read(path / "scaling_max_freq"))}
            result["cpu"][path.name] = observed
            if observed != {"min": EXPECTED["cpu_khz"], "max": EXPECTED["cpu_khz"]}:
                result["errors"].append(f"Unpinned CPU policy: {path.name}")
        if not result["cpu"]:
            result["errors"].append("No active CPU policy")
        gpus = list((root / "class/devfreq").glob("*gpu*"))
        if len(gpus) != 1:
            raise ValueError("Expected exactly one GPU devfreq")
        path = gpus[0]
        result["gpu"] = {"governor": _read(path / "governor"),
                         "min": int(_read(path / "min_freq")), "max": int(_read(path / "max_freq"))}
        if result["gpu"] != {"governor": EXPECTED["gpu_governor"],
                             "min": EXPECTED["gpu_min_hz"], "max": EXPECTED["gpu_max_hz"]}:
            result["errors"].append("GPU governor/range mismatch")
        sample = _last_complete_sample(proof["monitor_path"])
        age = (time.monotonic_ns() - sample["t_ns"]) / 1e9
        result["monitor_age_s"] = age
        if not 0 <= age <= max_age_s:
            result["errors"].append("Root monitor is stale or from another clock")
        result["emc"] = sample.get("emc")
        if result["emc"] is None or any(result["emc"].get(k) != v for k, v in
                {"rate_hz": EXPECTED["emc_hz"], "lock": 1, "halt": 1}.items()):
            result["errors"].append("EMC rate/lock mismatch")
        result["monitor_path"] = proof["monitor_path"]
        if proof.get("boot_id") and root == Path("/sys"):
            if proof["boot_id"] != _read("/proc/sys/kernel/random/boot_id"):
                result["errors"].append("Device rebooted after preparation")
        result["passed"] = not result["errors"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result["errors"].append(str(exc))
    return result


def _save(path, value):
    path = Path(path)
    with path.open("x") as stream:
        os.fchmod(stream.fileno(), 0o644)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def configure(controller):
    mode = controller.apply_maxn()
    if mode.get("status") == "reboot_required":
        raise RuntimeError("MAXN requires reboot; no inference started")
    env = controller.prepare_emc_and_fan(mode)
    if env.get("emc_hz") != EXPECTED["emc_hz"]:
        raise RuntimeError("Installed MAXN EMC cap differs from the frozen protocol")
    for domain in controller.domains:
        if domain["kind"] == "cpu":
            controller._set_range(domain, EXPECTED["cpu_khz"], EXPECTED["cpu_khz"])
        elif domain["kind"] == "gpu":
            controller._write(Path(domain["path"]) / domain["governor_name"], EXPECTED["gpu_governor"])
            controller._set_range(domain, EXPECTED["gpu_min_hz"], EXPECTED["gpu_max_hz"])
    power = controller.sys_root / "devices/platform/17000000.gpu/power/control"
    if power.exists():
        controller._write(power, "on")
    monitor = controller.start_monitor(20)
    return {"schema": "kvolt.eval_pilot.device.v1", "ready": True,
            "expected": dict(EXPECTED), "boot_id": controller.snapshot["boot_id"],
            "monitor_path": str(monitor.path), "prepared_ns": time.monotonic_ns(),
            "preparation": env, "initial_sample": monitor.latest_sample()}


def worker_prefix(source, user):
    return ["/usr/sbin/runuser", "-u", user, "--", "/usr/bin/env",
            "LD_LIBRARY_PATH=" + str(source / ".local/cusparselt/lib") + ":/usr/local/cuda/lib64",
            "TRITON_CACHE_DIR=" + str(source / ".cache/triton"),
            "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "USE_TORCH=1", "USE_TF=0",
            sys.executable, "-u", "-m"]


def _live_group_members(group_id, *, proc_root="/proc"):
    """Ignore zombies, which cannot execute or retain active CUDA work."""
    root = Path(proc_root)
    members = []
    if root.is_dir():
        for path in root.glob("[0-9]*/stat"):
            try:
                # comm is parenthesized and may itself contain spaces or ')'.
                fields = path.read_text(errors="replace").rsplit(")", 1)[1].split()
                if int(fields[2]) == group_id and fields[0] not in ("Z", "X"):
                    members.append(int(path.parent.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
        return members
    # macOS CPU validation has no procfs. Only PID/group/state are requested.
    result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,stat="],
                            check=True, capture_output=True, text=True, timeout=2)
    for line in result.stdout.splitlines():
        pid, group, state = line.split()
        if int(group) == group_id and not state.startswith(("Z", "X")):
            members.append(int(pid))
    return members


def stop_worker_group(child, *, timeout_s=10, poll_s=0.05,
                      group_members=None, clock=time.monotonic, sleep=time.sleep):
    """Terminate descendants even after the runuser/group leader has exited."""
    group_members = group_members or _live_group_members
    start = clock()
    result = {"group_id": child.pid, "passed": False, "signals": [], "timeout_s": timeout_s}

    def send(sig):
        try:
            os.killpg(child.pid, sig)
            result["signals"].append(sig.name)
        except ProcessLookupError:
            pass

    def live():
        child.poll()  # Reap our direct child; orphan zombies are ignored above.
        result["live_members"] = group_members(child.pid)
        return bool(result["live_members"])

    send(signal.SIGTERM)
    deadline = start + timeout_s
    while live() and clock() < deadline:
        sleep(min(poll_s, max(0, deadline - clock())))
    if result["live_members"]:
        send(signal.SIGKILL)
        deadline = clock() + 2
        while live() and clock() < deadline:
            sleep(min(poll_s, max(0, deadline - clock())))
    result["passed"] = not result["live_members"]
    result["wait_s"] = clock() - start
    result["leader_returncode"] = child.poll()
    return result


def run_stage(controller, command, source, session, stage):
    """Stop the entire child group before returning to clock restoration."""
    _save(session / f"worker-command-{stage}.json", {"argv": command})
    child = None
    try:
        with (session / f"worker-{stage}.log").open("x") as log:
            os.fchmod(log.fileno(), 0o644)
            child = subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            controller.worker = child
            result = child.wait()
    finally:
        if child is not None:
            stopping = stop_worker_group(child)
            _save(session / f"worker-group-stop-{stage}.json", stopping)
            if not stopping["passed"]:
                raise RuntimeError(f"Worker group still has live members before restoration: {stopping['live_members']}")
    _save(session / f"worker-result-{stage}.json", {"returncode": result, "finished_ns": time.monotonic_ns()})
    return result


def run(args):
    if os.geteuid() != 0:
        raise PermissionError("Device preparation requires sudo; inference will run as the non-root user")
    account = pwd.getpwnam(args.user)
    if account.pw_uid == 0:
        raise ValueError("Inference worker must be non-root")
    source = Path(args.source_root).resolve()
    inputs, output, snapshot = map(lambda x: Path(x).resolve(), (args.inputs, args.output, args.snapshot))
    if not inputs.is_file() or not (snapshot / "config.json").is_file():
        raise FileNotFoundError("Prepared inputs or local model snapshot is missing")
    if not output.is_dir() or output.stat().st_uid != account.pw_uid:
        raise ValueError("Create the output directory as the experiment user before sudo")
    document = json.loads(inputs.read_text())
    largest = max(document["items"], key=lambda item: item["length"])
    smoke_document = {**document, "items": [largest], "qualification_only": True}
    smoke_inputs = output / "smoke-inputs.json"
    if smoke_inputs.exists():
        if json.loads(smoke_inputs.read_text()) != smoke_document:
            raise ValueError("Existing smoke inputs differ from the frozen cohort")
    else:
        _save(smoke_inputs, smoke_document)
    smoke_output = output if args.smoke else output / "smoke"
    if not smoke_output.exists():
        smoke_output.mkdir()
        os.chown(smoke_output, account.pw_uid, account.pw_gid)
    if smoke_output.stat().st_uid != account.pw_uid:
        raise ValueError("Smoke output directory must belong to the experiment user")
    os.chdir(source)
    from scripts.p2.hardware_control import HardwareController
    session = output / ("device-session-" + str(time.monotonic_ns()))
    lock_path = Path("/tmp/kvolt-eval-pilot-device.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = HardwareController(session)
        with controller:
            proof = configure(controller)
            proof_path = session / "device-evidence.json"
            _save(proof_path, proof)
            verification = verify_environment(proof_path)
            _save(session / "verification.json", verification)
            if not verification["passed"]:
                raise RuntimeError("Prepared device failed readback")
            stages = [("smoke", smoke_inputs, smoke_output, True)]
            if not args.smoke:
                stages.append(("formal", inputs, output, False))
            for stage, stage_inputs, stage_output, smoke in stages:
                command = worker_prefix(source, args.user) + [
                    "scripts.eval_pilot.runner", "--inputs", str(stage_inputs), "--output", str(stage_output),
                    "--snapshot", str(snapshot), "--device-evidence", str(proof_path)]
                if smoke:
                    command.append("--smoke")
                returncode = run_stage(controller, command, source, session, stage)
                if returncode:
                    break
        if returncode == 0 and not args.smoke:
            returncode = subprocess.run(worker_prefix(source, args.user) + [
                "scripts.eval_pilot.analyze", "--output", str(output)], cwd=source).returncode
        return returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--source-root", default="/mnt/kvolt-data/kvolt")
    parser.add_argument("--user", default=os.environ.get("SUDO_USER", "agx31"))
    parser.add_argument("--smoke", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
