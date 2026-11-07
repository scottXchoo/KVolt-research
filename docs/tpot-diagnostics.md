# TPOT diagnosis and evidence

`tools/diagnose_tpot.py` is an offline, standard-library-only reader. It does not
load a model, modify a device, rerun the controller, or change targets or LUTs.
It accepts recorded `session.json` files and emits machine JSON plus an optional
compact Markdown report:

```sh
python3 tools/diagnose_tpot.py \
  artifacts/m4/review-r4/r4/run-first/session.json \
  artifacts/m4/review-r4/r4/run-repeat/session.json \
  artifacts/m4/review-r4/r4/run-performance/session.json \
  --output /private/tmp/kvolt-tpot-r4.json \
  --report /private/tmp/kvolt-tpot-r4.md
```

Exit status is 0 for valid traces, including real SLO violations; 1 when any
request trace is technically invalid; 2 for input or output errors. Raw files
are read without modification. The archived r4 qualification source remains
`8fae164`; a diagnostic report is not new qualification evidence.

## Measurement contract

The authoritative [Method](https://app.notion.com/p/40be0acd77f88014a36beb5285250208)
and [Evaluation Plan](https://app.notion.com/p/40de0acd77f880f98165f98215f51e20)
were fetched for this task. Method defines actual TPOT as
`(tN_ns - t1_ns) / 1e9 / (N - 1)`, including control, frequency application,
initial packing once, and subsequent Decode. N is the actual output count,
including EOS. `Nmax - 1` is the controller's planning denominator and is not
substituted for actual `N - 1` in measured TPOT.

The diagnostic reports:

- First intertoken interval: `t2 - t1`.
- Remaining intertoken mean/min/max: gaps from `t2` onward, using exactly
  `N - 2` gaps. These are descriptive measurements, not a claim of stationarity
  or GPU-kernel-only latency.
- Full Decode duration and TPOT, target, selected LUT prediction, selection
  path, eligible and budget-passing counts, and the minimum prediction across
  all 40 combinations when coverage is complete and unique.
- Recorded DC preselection time `td - t1` and the Decode device receipt,
  including requested/observed clocks, verification, and elapsed time.
- Optional `generation.stage_timing.initial_pack` and `first_decode` host-call
  spans, each `{start_ns, end_ns}`. The full spans are preserved. First-gap
  partitioning uses only their intersection with `t1..t2`; a first Decode call
  can return after its backend-stamped token-ready boundary, and clipping is
  recorded explicitly. Nested or overlapping spans are not summed.

Missing stage timestamps leave initial-pack and first-Decode durations null.
The uninstrumented remainder includes every unmeasured operation, including
packing and first Decode in r4. It must not be labeled measured quantization
latency. DC ends at `td` before filtering/selection; remaining controller work
can also be in that remainder. Partition components are never added to TPOT.

`N=1` preserves null Decode metrics. `N=2` has one first gap and no remaining
sample. Token timestamps and supplied duplicate timestamp fields must be
nonnegative integers and strictly increasing where applicable. Contradictory,
partial, failed, or technically invalid records retain selection evidence but
publish null computed TPOT and SLO results. Invalid optional stage spans produce
warnings rather than inventing a partition. Malformed/incomplete candidate
coverage cannot support an all-40 minimum claim.

## Archived r4 findings

The three sessions contain 24 completed requests: 3 N=1 requests and 21 requests
with measurable Decode. Recomputing token timestamps reproduces **19/21 TPOT
violations**: 7/7 in run-first, 6/7 in run-repeat, and 6/7 in run-performance.
Device Decode apply receipts are verified in all 21 measurable requests.

All values below are milliseconds; ranges span these three archived sessions.

| L | Actual N | Path / selected G / MHz | Target | First gap | Remaining mean | Actual TPOT |
|---:|---|---|---:|---:|---:|---:|
| 1024 | 2 | latency_fallback / 128 / 1122 | 103.211 | 169.973–171.396 | null | 169.973–171.396 |
| 1024 | 50 | latency_fallback / 128 / 1122 | 103.211 | 170.329–171.827 | 121.579–122.214 | 122.574–123.214 |
| 4096 | 46 | latency_fallback / 128 / 1020 | 117.439 | 249.074–250.492 | 121.544–122.452 | 124.378–125.295 |
| 8192 | 2 | energy_min / 128 / 918 | 136.512 | 391.253–396.388 | null | 391.253–396.388 |
| 8192 | 48 | energy_min / 128 / 918 | 136.512 | 391.145–395.629 | 130.853–131.160 | 136.391–136.710 |

At L1024 and L4096, all 40 LUT candidates already exceed the target: global
minimum predictions are 118.726 ms and 119.206 ms respectively. Fallback is the
recorded, expected policy outcome. Remaining-gap means also exceed the target;
removing initial packing alone would not close the observed steady-gap deficit.
This is an observation about the recorded workload, not proof that another
unmeasured implementation cannot meet the target.

At L8192, the selected prediction is 131.651 ms; the global minimum across all
40 is 124.947 ms. Both are below the 136.512 ms target. The controller's normal
energy-minimum choice does not promise actual SLO satisfaction. The first gap
is roughly 392 ms, so N=2 exposes the startup cost directly. At N=48 it is
amortized sufficiently to straddle the target: only run-first violates.

DC preselection costs 1.519–2.064 ms and verified device apply costs
6.598–10.376 ms. These account for a small part of the first gap. Existing r4
records do not isolate initial packing, the first Decode call, or remaining
host overhead, so their individual contributions remain unresolved.

## LUT source evidence and interpretation

The immutable source is
`../KVolt-lut-source/data/luts/agx2-repeat5-20261103/decode-lut.json`.
For each selected cell, `metrics.T_init_s.mean`, `tpot_actual_s.mean`, and
`output_tokens.raw` give:

| L / G / MHz | Initial packing mean (ms) | LUT TPOT (ms) | Outputs in each of 5 repeats |
|---|---:|---:|---|
| 1024 / 128 / 1122 | 40.829 | 118.726 | 512, 512, 512, 512, 512 |
| 4096 / 128 / 1020 | 115.396 | 119.206 | 512, 512, 512, 512, 512 |
| 8192 / 128 / 918 | 250.595 | 131.651 | 512, 512, 512, 512, 512 |

Source provenance explicitly states Decode boundary `t1..tN` includes initial
K2V2 packing once. `init_start..tN` is nested in the tail and must not be added
again. The LUT aggregates each request's ratios, then averages the five
technical repeats of one document. It does not validate arbitrary mixed-phase
prediction, cross-frequency transitions, or fixed-output workloads.

Thus **packing is present in the LUT**, amortized over 511 Decode outputs for
these cells. r4 ends at actual N=2/46/48/50. That difference supports startup
amortization as a contributor, especially at L8192, without establishing an
exclusive causal attribution or authorizing LUT/target/policy changes.
New stage measurements and backend profiling are needed to separate that cost
from the observed remaining-gap deficit.

## Verification

`tests/test_tpot_diagnostics.py` independently checks synthetic hand-calculated
intervals, N=1/N=2, integer-nanosecond precision beyond a binary64 exact integer
origin, malformed/duplicate timestamps, partial results, stage partitioning,
no raw mutation, and all 24 real r4 requests. Portable CPU tests use the tracked
reduced fixture `tests/fixtures/tpot-r4.json`, retaining the analyzer-consumed
fields while excluding prompts and token IDs. Its source commit is `8fae164`.
It records SHA-256 hashes of all three original sessions, `audits.json`, and
`qualification.json`; the latter is
`49210efcbb4a996bf07b84e81838ecbdce93e10d9b9d43309d929cb0b0cbbc82`.
The CLI command above separately checks the complete forensic archives when
available. Tests do not require the ignored archives or skip their absence.
The tool runs without GPU packages.
The independent tester also checks the development profiling driver's wrapper
return/exception behavior and validator restoration in
`tests/test_profile_tpot.py`. Live AGX2 profiling and any new qualification are
separate runs owned by the main agent; none are implied by these offline tests.

## Development profiling driver

`tools/profile_tpot.py` wraps existing backend calls and pure metadata validators
without changing their results or suppressing their exceptions. It uses a
hardware lock, formal warmup, development-mode power collection, and the existing
hardware preparation/restoration scope. Host wrapper overhead is included.
Nested validator durations overlap their parent backend call and must not be
added to it. Its output is explicitly non-evaluation diagnostic evidence.

The administrator invocation requires a previously reviewed plan digest:

```sh
sudo python3 tools/profile_tpot.py \
  --plan /absolute/original-r4-directory/plan.json \
  --plan-sha256 <reviewed-plan-sha256> \
  --runtime-root /absolute/clean-runtime-checkout \
  --output-dir /absolute/new-diagnostic-directory
```

`--plan-sha256` binds the original non-evaluation plan. Before hardware work,
the driver verifies its schema, original root/path, fixed manifest/profile
paths, and each frozen input file hash. Tampering with a plan or manifest is
rejected. The selected Runtime and driver checkouts must have clean code, tests,
and tools; their commits are recorded separately. A different Runtime commit
is allowed for baseline/fixed comparison. This does not assert that the
diagnostic source has passed M4 qualification.
