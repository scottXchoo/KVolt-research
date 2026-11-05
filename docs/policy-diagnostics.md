# M2 CPU policy diagnostics

`kvolt diagnose` predicts the PC/DC choices for an assumed input length, output
cap, first token and elapsed phase times. It performs no tokenization, inference,
GPU frequency operation, device lock or power collection. `run` belongs to M3
and is not exposed by this CLI.

Install the optional policy dependencies on the target Python 3.10 environment:

```sh
python -m pip install '.[policy]'
kvolt diagnose --length 1536 --max-new-tokens 512 \
  --pc-elapsed-ns 1000000 --dc-elapsed-ns 2000000 \
  --lut-bundle /path/to/agx2-repeat5-20261103
```

The extra pins NumPy 1.26.4, SciPy 1.15.3 and kneed 0.9.0. Core dependencies
remain empty. `kvolt --help` and `kvolt diagnose --help` load no NumPy, SciPy,
kneed, torch, tokenizer or model library. Active Decode requires the actual
pinned detector; a missing/broken package fails with a dependency error rather
than a detector recovery. `loaded_versions` records module versions actually
loaded by this process, while `installed_distribution_versions` records package
metadata separately. A null loaded version does not assert a package was used.

Equivalent source-checkout commands are:

```sh
PYTHONPATH=src python -m kvolt_runtime diagnose --length 1536 \
  --pc-elapsed-ns 1000000 --dc-elapsed-ns 2000000 --mock baseline
python tools/diagnose_policy.py --length 1024 --max-new-tokens 1 \
  --pc-elapsed-ns 0 --mock exact_ties --output /tmp/new-policy-report.json
```

Exactly one source is mandatory: `--lut-bundle PATH` or `--mock SCENARIO`.
Synthetic scenarios are `baseline`, `flat_quality`, `nonmonotonic` and
`exact_ties`. A rejected measured bundle never becomes a synthetic source.
The measured loader verifies the frozen manifest and all ten source file hashes.

L must be an integer in [1024,8192], Nmax in [1,512], and elapsed times must be
nonnegative integer nanoseconds. `--max-new-tokens` defaults to 512.
`--first-token-id` defaults to **assumed token 17**, not a generated token.
An assumed EOS (128001,128008,128009) or Nmax=1 uses the real DC skip path:
Quality/Decode lookup, detector initialization/detection and the DC clock do not
run. EOS takes precedence over Nmax=1. `--dc-elapsed-ns` may be omitted on skip;
if supplied it is recorded as not applied. Full LUT support preflight still runs.
An active DC without this time is rejected before source loading or lookup.

After full-support preflight, PC performs its online lookups before reading the
injected clock, with t0=0 and tp equal to the assumed PC elapsed ns. Active DC
performs Quality, Kneedle, Decode and target queries before its injected clock,
with t1=0 and td equal to the assumed DC elapsed ns. These are separate synthetic
phase origins, not a request timeline. Budgets use the controllers' unchanged
binary64 arithmetic and original candidate/energy/tie policies. No time is
measured by this diagnostic and no value is rounded before selection.

Each successful invocation prints one JSON record to stdout, even with
`--output`. The schema is `kvolt.policy_diagnostic.v1`, record type
`policy_diagnostic`, with `synthetic_assumptions=true`, `predictions_only=true`
and `measurements=null`. Source kind (`measured` or `synthetic`) is independent
of these synthetic timing/token assumptions. `prefill` preserves eight candidate
records; active `decode` preserves five raw Quality rows, Knee result/reasons,
40 candidate records, target, budget, selected prediction, flags and reasons.
The target/budget fields are the Method's tau_p/Bp and tau_d/Bd. A skipped DC
retains null target/budget/elapsed/td/selection/Knee/feasibility and empty rows.
Prediction feasibility is not an observed TTFT/TPOT SLO result.

Source metadata preserves the original flags, measurement conditions, model,
limitations, source commit, manifest hash and ten file hashes. Bulk Quality rows,
bootstrap strata and raw-record hash maps are omitted; the original sealed files
remain the full metadata reference. In particular `runtime_eligible=false` and
`runtime_validation_required=true` remain source facts. A report establishes no
formal Runtime accuracy, SLO, energy or frequency-transition qualification.
Runtime metadata includes the actual Python and dependency versions, runtime
source hashes and Git HEAD/dirty state when this is an actual Git checkout.
An installed/non-Git source snapshot records null Git fields; it never invents a
commit. Source hashes identify the bytes used even for a dirty checkout.

`--output` exclusively creates a new file, including for a structured rejected
attempt, and refuses any existing path (including symlinks). It never overwrites
a prior attempt. A nonexistent parent or write failure is an output rejection;
stdout contains the structured error when possible. Diagnostics go to stderr.
JSON uses `allow_nan=False`. Valid-syntax input, LUT, dependency and output errors
return one error/rejected JSON record and exit 2. Argument syntax errors follow
argparse (stderr usage, exit 2). Successful prediction, fallback or skip exits 0.

Development checks run separately from target qualification. Local macOS Python
3.13 has NumPy 2.3.1 and no working pinned SciPy/kneed stack (the isolated SciPy
1.15.3 wheel fails to load `_spropack`). Local fake-detector unit tests do not
replace the mandatory actual-detector CLI tests. The target integration tests
require Python 3.10, NumPy 1.26.4, SciPy 1.15.3, kneed 0.9.0 and the sealed bundle;
missing/wrong policy dependencies fail those tests rather than skip them.
