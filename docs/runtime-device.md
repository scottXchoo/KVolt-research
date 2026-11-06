# M3 device control

`kvolt_runtime.device` is CPU-only. Importing it does not import torch, CUDA, or
the model. `DeviceController` implements the shared C5 Device protocol and only
writes the selected GPU devfreq directory's `governor`, `min_freq`, and
`max_freq` nodes. Device checks never set CPU, EMC, MAXN, fan, services, ownership,
permissions, or sudo credentials.

## Ownership and receipts

Acquire `HardwareLock()` before any GPU import, loading, warmup, or mutation,
then pass the held object as `DeviceController(profile, lock=lock)`. It opens the
existing `/run/lock/kvolt-p3-hardware.lock` in `rb` mode and uses
`LOCK_EX | LOCK_NB`. Missing, busy, or replaced lock inodes fail immediately.
It never creates, truncates, unlinks, replaces, or waits for the lock. The path
constructor argument exists for isolated tests; production uses the common path.
Do not wrap the probe in another `flock` process.

Every operation returns a JSON-compatible receipt with host monotonic integer
`start_ns`/`end_ns`, full explicit profile, physical serial SHA-256, boot ID,
fresh observations, successful changes, individual write attempts, verification
sample counts, and status. `ExecutionError`/`PreparationError` retain the partial
receipt and original exception cause. `changed` means at least one setter
returned successfully. A failed setter can have partially mutated hardware;
`mutation_uncertain=true` records that case and restoration remains necessary.
Failed write attempts retain an error and their own completion timestamp.

Call `capture()` before any GPU changes and preserve its successful receipt.
`verify_prepared()` checks all three CPU policies at min=max=2201600 kHz, EMC
rate/max_rate/cap=3199000000 Hz, EMC rate lock=1 and bandwidth-manager halt=1,
MAXN and explicit mode ID, explicit TPC mask, fan PWM=255, Tj `user_space`
policy, and inactive `nvfancontrol`. Fixed read-only `nvpmodel -q` and
`systemctl is-active nvfancontrol` subprocesses use the profile's finite query
timeout. Missing nodes, permissions, invalid responses, and mismatches fail
closed. The original GPU range may differ from the common start range.

`prepare_request()` sets/verifies `nvhost_podgov`, min=306000000 Hz and
max=1122000000 Hz before t0. Its observed instantaneous `cur_freq` is recorded
without requiring an identical DVFS number. `apply(frequency_mhz)` accepts only
408/510/612/714/816/918/1020/1122 MHz, freshly checks driver availability, then
sets min=max to the selected frequency. It confirms fresh governor/min/max and
exact `cur_freq` for the explicitly requested consecutive sample count. A setter
is skipped only when all four initial conditions freshly match. Even that path
performs the full fresh confirmation and records its cost. Bounds are written
in an order that preserves min<=max when moving either direction.

`restore(snapshot)` accepts the original capture receipt from the same profile,
physical device, and boot. It restores and confirms original governor/min/max;
it does not reproduce a transient DVFS `cur_freq`. Runtime must preserve
generation/measurement results independently of later restore errors. No device
failure chooses another frequency or continues degraded generation.

## Explicit profiles and probe commands

`DeviceProfile.from_json(path)` requires every field shown by discovery. It
rejects unknown/missing/duplicate fields, malformed paths, nonfinite/noninteger
timings, and impossible match schedules. Required timing fields are
`timeout_ns`, `poll_ns`, `consecutive_matches`, and `query_timeout_ns`. There are
no hardware timing defaults. `identity_expected` is SHA-256 of the stripped
physical serial, not a boot ID; profiles remain reusable after a reboot while
capture/restore must remain within one boot.

The following read-only command discovers exact paths, writes a provisional
profile and reports preparation mismatches. Every timing argument must come
from an explicitly recorded development proposal or hardware evidence:

```sh
python3 tools/probe_device.py --output DISCOVERY.json discover \
  --name agx2 --evidence 'Explicit development probe; not M4 qualification' \
  --timeout-ns TIMEOUT_NS --poll-ns POLL_NS --matches MATCHES \
  --query-timeout-ns QUERY_TIMEOUT_NS --profile-output DEVICE_PROFILE.json
python3 tools/probe_device.py --output READONLY.json probe --profile DEVICE_PROFILE.json
```

Only the explicitly selected `qualify` action changes GPU settings. After
read-only preparation succeeds it establishes the common GPU start, visits all
eight candidates ascending and descending, immediately repeats each apply to
exercise the no-op path, and restores the original capture in `finally`:

```sh
python3 tools/probe_device.py --output GPU_TRANSITIONS.json qualify --profile DEVICE_PROFILE.json
```

Run a privileged probe only through the separately authorized, bounded
administrator procedure. The needed writes are limited to those three GPU
nodes; access to restricted EMC debug nodes is read-only. An administrator
separately prepares CPU/EMC/MAXN/fan using the existing trusted preparation
procedure and preserves their original state. Neither global passwordless sudo,
sysfs permission changes, nor a second lock path is needed. Administrator
authentication is handled interactively outside artifacts.

Both profile statuses, `provisional` and `device_verified`, remain distinct from
formal M4 runtime qualification. Driver `cur_freq` is a driver report, not an
independent physical-clock measurement or proof of fixation throughout a request.
Unit fixtures prove software behavior only. Actual AGX transition timing,
restoration, and preparation evidence must be recorded separately per device.
