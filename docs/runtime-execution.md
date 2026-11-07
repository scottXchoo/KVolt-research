# M3 sequential execution

M3 connects the already verified Backend and PC/DC without changing their
numerical paths or selection policy. M4 remains the formal qualification gate.

## Ownership and boundaries

* Input preflight reads every manifest entry, fixes UTF-8 bytes/token IDs and
  validates complete LUT support before model loading. No online interpolation
  or Knee detection is moved into preflight.
* SessionRunner holds the existing nonblocking hardware lock before any GPU
  import, model load, warmup, or setting mutation. The same lock inode is reused.
* The device validates separately prepared CPU/EMC/MAXN/fan conditions and only
  writes GPU governor/min/max. It captures original GPU settings before changes.
* One resident model serves requests sequentially. Every request owns fresh KV.
* One monotonic integer-nanosecond clock is shared by Backend, PC/DC, Runner,
  device receipts and power sampling. Backend timestamps follow GPU completion.
* `t0` is read immediately before PC. PC lookup/selection, frequency application,
  input preparation and FP16 Prefill are included in `t0..t1`.
* First EOS or Nmax=1 skips DC, frequency application for Decode, initial
  packing and Decode forwards. Otherwise DC uses the original `L` and `t1`,
  followed by frequency application, one initial pack, and Decode forwards.
* A normal output of N tokens has N-1 Decode forwards and final KV length
  L+N-1. The final token-ready timestamp fixes the measurement end. Cleanup and
  result serialization never move this boundary or add time/energy twice.
* PC/DC selection stays inside the measured generation interval. Conversion of
  their immutable decisions and recovered-Knee event evidence to result dicts
  runs after generation ends, including failure/interruption paths, before
  timing/SLO evaluation and recording. Selected candidates and skip metadata
  are preserved without nested evidence serialization delaying GPU work.
* Additive `generation.stage_timing.initial_pack` and `first_decode` diagnostics
  each contain `start_ns` and `end_ns` from the shared monotonic clock around
  the corresponding Backend call. `end_ns` is recorded even if the call fails;
  a call never started is `null`. These spans partition existing execution
  work; their durations are never added to token-based timing or energy.
  Backend records readiness before returning, so a first Decode that produces
  the final token can have a wrapper `end_ns` after `tN`; `tN` still ends the
  normal measurement interval.
* Generation, timing, energy and cleanup statuses remain separate. Valid normal
  measurements survive a later restoration failure. Technical execution errors
  abort the session; later requests are `not_run`; the CLI never retries.
* Signal handlers set a flag only. Runner stops additional generation after the
  current forward, saves its prefix and restores GPU settings. SIGINT returns
  130 and SIGTERM returns 143. No recovery guarantee exists for SIGKILL.

## Interface allocation

`execution/contracts.py` is the shared CPU-only boundary. Input/result code owns
`execution/inputs.py` and `recording/`. Device code owns `device/`. Measurement
code owns `telemetry/`. The main agent owns common contracts and Runner wiring;
feature agents own one logical task with independent test-agent verification.

Device operations return JSON-compatible receipts, including a fresh observed
state, integer ns boundaries, applied changes and verification. Device errors
carry any partial receipt. The collector starts before t0 and finishes with a
right-side bracket after tN, preserving raw trace and failure metadata.

Development permits explicit synthetic LUTs, marked synthetic, and AGX1 use of
AGX2 LUTs with device mismatch recorded. Performance requires measured data,
device/condition match, power collection and M4 qualification evidence. M3 does
not turn source `runtime_eligible=false` into a formal performance approval.

Hardware deadlines, polling and sampling settings must be explicit in a recorded
device/collector profile. Proposed development settings are not a completed
device qualification. AGX1 remains untouched while another job owns its lock.
