# M3 timing and INA3221 energy

This module implements development measurement plumbing. It does not qualify
sensor response, runtime performance, or entry into the M4 formal evaluation.
It imports only the Python standard library and CPU-only runtime errors.

## Timing boundaries

The runner records `t0_ns = time.monotonic_ns()` immediately before PC. The
backend returns integer token-ready timestamps from that same host clock, after
GPU synchronization. The measurement helpers consume those timestamps without
calling the clock again. Wall clock, `perf_counter`, predicted LUT values, and
GPU event clocks do not replace these boundaries.

`timing_metrics(t0_ns, token_ready_ns, *, completed)` returns TTFT
`(t1_ns - t0_ns) / 1e9` and mean TPOT
`(tN_ns - t1_ns) / (N - 1) / 1e9`. Subtract integers before conversion. `N` counts
actual outputs including EOS. The Decode interval includes DC, frequency
application/confirmation, initial quantization/packing, cache replacement, and
later Decode forwards. Do not add those costs a second time. Collector startup,
right-bracket waiting, cleanup, and result serialization lie outside token timing.

No output gives `valid=false`, `reason=no_output_tokens`, and null measurements.
For `N=1`, Decode duration, mean TPOT, Decode energy, and J/Decode token are null,
with `decode_reason=no_decode_tokens`; Prefill/request measurements remain valid.
An incomplete generation has null normal measurement fields and
`reason=incomplete_generation`. Measurable prefix diagnostics remain under
`prefix`, together with original output timestamps. A prefix is not a normal
complete-request observation.

## Request memory diagnostics

Every attempted request also records `memory` (`kvolt-memory-v1`) independently
of power collection. `RequestMemoryMonitor` uses only the standard library and
the backend's already loaded `torch` object; importing telemetry does not import
torch, initialize CUDA, or add a dependency. The backend fixes inference to
`cuda:0`, and every allocator read/reset explicitly selects that device.

The runner captures `before` after device preparation, before power collector
startup and `t0`; it captures `after` after generation, before decision
serialization and cache release. Each snapshot has monotonic read-start/end
timestamps because the counters are read sequentially, not atomically. On a
failed or interrupted generation, snapshots remain diagnostics and
`generation_status` identifies the incomplete request. An unattempted request
has `status=not_run` and null snapshots.

| Section | Values | Scope |
| --- | --- | --- |
| `system` | `total_bytes`, `available_bytes`, `used_bytes`, `usage_percent`, `swap_total_bytes`, `swap_free_bytes`, `swap_used_bytes` | `/proc/meminfo`; used = total minus available; usage = 100 × used / total |
| `process` | `rss_bytes`, `hwm_bytes`, `swap_bytes` | `/proc/self/status`: VmRSS, VmHWM, VmSwap |
| `cuda` | `allocated_bytes`, `reserved_bytes`, `max_allocated_bytes`, `max_reserved_bytes`, `device` | PyTorch CUDA allocator for the inference process/device |

Linux `kB` values are converted to bytes using 1024. `MemAvailable` is the
kernel's estimate of memory available without swapping. It is not interchangeable
with `MemFree`. VmRSS is the kernel's approximate resident-set accounting;
VmHWM is the process-lifetime RSS high-water mark, including earlier model load,
warmup and requests. Neither two RAM snapshots nor VmHWM is a request-specific
RAM peak or an average over the inference interval.

CUDA peaks are reset once after the before snapshot, before power startup and
`t0`. Only the after snapshot publishes reset-scoped maxima. The explicit
`cuda_peak_reset` receipt and `scope.cuda_peak_window` identify the window;
it surrounds, but is not exactly, `t0..tN`. Peaks include resident model weights
and any transient allocation during Prefill/packing/Decode. Earlier warmup
high-water marks are excluded, but memory retained by the caching allocator
at reset remains part of the baseline. No `empty_cache`, extra synchronize,
device-memory query, or per-token memory sampling is performed. Reset failure
leaves request peaks null instead of publishing an earlier lifetime maximum.

`allocated` measures tensor allocations; `reserved` includes the allocator's
cached blocks. Neither is device-wide usage or an isolated KV-cache size.
System RAM, process RSS and CUDA allocator values can overlap, especially on
the shared-memory Jetson, and must not be summed. This change does not measure
KV tensor storage separately, memory bandwidth, or a continuous RAM trace.

Each section retains available fields and per-field error reasons. Missing proc
files (including macOS), missing/uninitialized CUDA, malformed fields, and failed
reads remain null with explicit reasons. Overall `status` is `valid`, `partial`,
or `unavailable`, according to the available diagnostics. Unexpected monitor
exceptions are recorded inside `memory.errors` with `reason=collection_failed`.
Memory diagnostic failures do not change generation, timing, energy, SLO, or
cleanup validity. `units` and `scope` travel with the stored evidence. `clock`
identifies `time.monotonic_ns` or an `injected-test-clock`; fixture timestamps
are never labeled as production measurements.

The complete record is saved in `requests/NNNNNN/result.json` and the matching
`session.json.results` entry. Stdout request summaries add `memory_status` and
retain the existing artifact links. Historical records are not rewritten.
CPU tests cover fixture arithmetic, unavailable/malformed counters, peak reset
isolation, timing boundaries, errors/interruption/N=1, and JSON persistence.
Actual AGX allocator readings and diagnostic overhead require hardware
validation; existing formal qualification does not transfer to changed source.

Counter semantics follow the [Linux proc documentation](https://docs.kernel.org/filesystems/proc.html)
and [PyTorch 2.5 CUDA memory API](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/cuda/memory.py).

## Sensor scope and units

Energy sums exactly these three independent INA3221 rails:

- `VDD_GPU_SOC`
- `VDD_CPU_CV`
- `VIN_SYS_5V0`

This reproduces the existing LUT/pilot scope. Do not add `VDD_IN`, diagnostic
rails, or any overlapping supply. This is module-rail sensor energy, not wall
power. There is no idle subtraction or clipping. Signed finite current readings
are preserved, consistent with the previous logger. Voltage input is mV and
current input is mA; `power_w = voltage_mv * current_ma / 1e6`.

Each flat sample contains `rail`, `timestamp_ns`, `read_start_ns`, `read_end_ns`,
`current_ma`, `voltage_mv`, `power_w`, and `error`. Timestamp is integer midpoint
`(read_start_ns + read_end_ns) // 2` of that rail's label/voltage/current read
window. This preserves read uncertainty; it does not assert simultaneous rail
reads or instantaneous sensor response. Labels are checked on each read.
Sensor response/averaging/update latency remains unvalidated.

## Integration and validity

`energy_metrics(samples, t0_ns, t1_ns, tN_ns, n, *, completed, max_gap_ns)`
integrates each rail independently before summing. Every rail must have strictly
increasing integer timestamps, a valid point strictly before `t0_ns`, and a valid
point strictly after `tN_ns`. The two interval boundaries are interpolated
linearly between their adjacent sensor samples. Every intersecting sample pair
must have gap at most `max_gap_ns`. The clipped linear segments are integrated by
the trapezoid rule; there is no extrapolation, zero filling, or interpolation
across an invalid row. Any malformed/error/nonfinite row in the supplied trace
invalidates the trace conservatively, including rows outside the integration
window. Keep the complete raw trace to diagnose that outcome.

The result contains `prefill_energy_j`, `decode_energy_j`, `total_energy_j`, and
`decode_j_per_token`, both as the rail sum and inside `rails[rail]`.
Prefill integrates `[t0,t1]`, Decode integrates `[t1,tN]`, and total integrates
`[t0,tN]`. Decode J/token divides by actual `N-1`. Disabled collection is passed
as `samples=None`, returning null energy with `reason=not_collected`. An empty
collected trace is invalid, not zero energy. Invalid and incomplete measurements
never substitute predicted LUT energy. `valid` is measurement validity, separate
from generation, restoration, and SLO statuses.

## Explicit collector profile

`PowerProfile.from_json(path)` requires a UTF-8 JSON object, without duplicate
keys or nonfinite constants. Every interval and deadline, including polling,
is an integer from `1` through `86400000000000` ns (24 hours). This bound avoids
overflow and impractical timeout values. Unknown fields and extra/missing rails
are rejected. Required fields:
`rails`, `interval_ns`, `max_gap_ns`, `ready_timeout_ns`,
`coverage_timeout_ns`, and `join_timeout_ns`. Optional `poll_interval_ns`
defaults to `1000000`; `name` and `evidence` describe development provenance.
Every rail requires absolute and unique `label_path`, `current_path`, and
`voltage_path`; there is no implicit discovery or overlapping power-input sum.
Paths, `name`, and `evidence` must be nonempty valid UTF-8 strings; lone
surrogates are rejected during preflight. Paths also reject embedded NUL bytes.
The validated rail mapping has immutable tuple storage and immutable `RailPaths`
values. Mutating constructor dictionaries or fresh `to_dict()` output cannot
change the sampled profile. The mapping remains pickleable for spawn workers.

Example AGX2 development profile (paths must be checked on the selected host):

```json
{
  "name": "agx2-ina3221-development",
  "evidence": "20ms interval and 100ms maximum gap reused from pilot; M4 unqualified",
  "interval_ns": 20000000,
  "max_gap_ns": 100000000,
  "ready_timeout_ns": 5000000000,
  "coverage_timeout_ns": 1000000000,
  "join_timeout_ns": 1000000000,
  "poll_interval_ns": 1000000,
  "rails": {
    "VDD_GPU_SOC": {
      "label_path": "/sys/class/hwmon/hwmon1/in1_label",
      "current_path": "/sys/class/hwmon/hwmon1/curr1_input",
      "voltage_path": "/sys/class/hwmon/hwmon1/in1_input"
    },
    "VDD_CPU_CV": {
      "label_path": "/sys/class/hwmon/hwmon1/in2_label",
      "current_path": "/sys/class/hwmon/hwmon1/curr2_input",
      "voltage_path": "/sys/class/hwmon/hwmon1/in2_input"
    },
    "VIN_SYS_5V0": {
      "label_path": "/sys/class/hwmon/hwmon1/in3_label",
      "current_path": "/sys/class/hwmon/hwmon1/curr3_input",
      "voltage_path": "/sys/class/hwmon/hwmon1/in3_input"
    }
  }
}
```

The actual AGX2 mapping was reported as `hwmon1` / `name=ina3221`, channels 1–3
in the order above. Software polling at 20ms does not establish sensor bandwidth.
The 100ms maximum gap is provisional development policy from the existing pilot,
not a newly qualified M4 limit. Readiness/coverage/join deadlines in the example
are explicit development choices and also require M4 validation.

## Collector lifecycle and storage

`INAPowerCollector(profile, *, clock=time.monotonic_ns)` uses multiprocessing
`spawn`, preventing inherited CUDA/fork state and isolating the parent GIL. Its
worker owns raw JSONL writes; it sends no per-sample objects through parent IPC.
Only fixed-size latest-valid timestamps and readiness/update events are shared.
The parent reads the trace after bounded stop/join. `samples_csv(samples)` is a
post-collection serializer for the recorder. Persist trace/metadata outside the
critical token interval.

`start()` waits under `ready_timeout_ns` until every rail has a valid fresh
sample, at most `max_gap_ns` old. The runner must require returned `ready=true`
before recording t0. A failure returns errors and stops the worker. Repeated
start and start after stop are contract errors.

`finish(end_ns)` waits under `coverage_timeout_ns` for every rail to have a valid
sample strictly after the final boundary, then stops/joins. Coverage failure is
an explicit error; the trace remains available. `stop()` is idempotent, also
before start, and is used when generation produces no token. Graceful join,
termination, and kill each use at most `join_timeout_ns`; escalation is recorded
as a technical error. Worker failures, raw read errors, startup/coverage/join
failures, skipped polling deadlines, and raw rows are retained in receipts.
The runner keeps cleanup errors separate from the generation outcome.

Receipt `trace_sha256` hashes the exact `raw_trace_jsonl` UTF-8 bytes written by
the worker (`trace_hash_format="worker JSONL bytes"`). If the recorder stores CSV,
it must compute and label that CSV artifact's own SHA256; the JSONL hash is not
the CSV hash. Keep the explicit profile, source/model/LUT/device provenance,
clock/unit/scope, and `sensor_response_validated=false` with recorded artifacts.
An injected clock supports CPU tests; its receipts say `injected-test-clock`
and must not be presented as production host measurements. All lifecycle
**deadlines** use real `time.monotonic_ns` even when sample clocks are injected.

## Existing source audit and verification boundary

Read-only AGX2 inspection found `scripts/eval_pilot/runner.py:429` imports
`RequestPowerLogger` from `scripts.p2.measurement`; that module reuses
`scripts.p1.power_logger.PowerLogger`. Its `TOTAL_RAILS` matches this module,
reads mV/mA, preserves negative current, and collects in a spawned process.
The new runtime does not import or modify those legacy scripts.

CPU analytical and synthetic-file process tests cover exact linear integrals,
per-rail sums, N−1 normalization, high integer clock origins, missing/nonfinite/
nonmonotonic/error rows, missing brackets, excessive gaps, N=1, partial/no-output,
no collection, readiness failure, successful right coverage, and idempotent stop.
Actual AGX2 read-only samples and independent tester evidence are tracked under
`artifacts/m3-t1/`. No GPU settings or INA sensor configuration are written.
