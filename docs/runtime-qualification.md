# M4 scoped runtime qualification

M4 verifies implementation, policy, data, timing/power, device control, recovery
and reproducibility before formal evaluation. It sets no SLO attainment or energy
saving target. Normal SLO violations remain valid observations.

An evidence report is scoped to the Runtime Python source hashes, model and
tokenizer pins, KIVI sources and native binary pins, Python/platform/dependencies,
physical AGX2 identity and boot ID, sealed measured LUT hashes, device profile
and power profile. Source/profile/device changes require matching new evidence.
The original LUT's `runtime_eligible=false` flags are never rewritten.

The report loader uses only the standard library. It verifies every required
machine-check summary, all referenced artifacts and their SHA-256s, and exact
binding equality. Artifact paths cannot leave the report directory or traverse
symlinks. These hashes identify the supplied evidence; they do not authenticate
an untrusted report author. Use reports produced from the reviewed qualification
driver and inspect its original logs. See [qualification-evidence.md](qualification-evidence.md).

## Frozen execution plan

Use the pinned AGX2 Python/environment and a clean committed checkout.
`tools/qualify_m4.py prepare` freezes copies of the existing development cohort
and natural first-EOS cohort, retains Nmax=1/2 cases, adds maximum-cap checks,
and executes the CPU suite and opt-in GPU cache/Prefill correctness tests. It
records the commands, exit codes, JUnit results, source and input hashes. These
diagnostic inputs remain excluded from the formal evaluation.

```sh
python tools/qualify_m4.py prepare \
  --model-dir SNAPSHOT --lut-dir SEALED_AGX2_LUTS \
  --device-profile DEVICE.json --power-profile POWER.json \
  --kivi-root KIVI --baseline-root INDEPENDENT_REFERENCE_ROOT \
  --manifest DEVELOPMENT_MANIFEST.json \
  --eos-manifest FIRST_EOS_MANIFEST.json --m1-cohort PINNED_M1_INPUTS.json \
  --output-dir NEW_M4_DIRECTORY
```

Only the explicit `execute PLAN.json --plan-sha256 PREPARE_EMITTED_SHA256` stage temporarily prepares the existing
CPU/EMC/fan controls and writes GPU settings. Run this stage with the necessary
administrator access. It never switches MAXN, reboots, edits system configuration
or changes file permissions. It uses the existing nonblocking common lock and
restores captured controls in `finally`. A busy lock rejects immediately.
The digest is printed by `prepare` and must identify that reviewed plan exactly.
The driver validates frozen source, paths, profiles, test counts and input hashes
before hardware access; recorded commands are provenance and are never executed.

The integrated checks cover all eight GPU frequencies and repeated no-op
applications, formal preparation and repeated adaptive requests, naturally
reached first EOS, exact native reference/repeat comparison, real SIGINT/SIGTERM
handling, saved prefixes and later `not_run` requests, and final restoration.
An independent audit recalculates energy from raw integer mV/mA readings using
exact rational arithmetic. It checks clock boundaries, N-1 normalization,
strict outside trace brackets, maximum sample gaps, policy budgets and selection
order, selected-clock readbacks, saved inputs and results.

The three INA3221 rails retain their sensor scope. Qualification of collection
and integration does not calibrate physical sensor response or measure wall
power. Missing/failed traces never become zero energy. N=1 Decode metrics stay
null. Actual SLO failures never become technical failures.

Every plan permits one execution attempt. A failed attempt keeps its evidence;
freeze a new plan/output directory for a corrected attempt. No automatic retry
or silent cohort extension occurs.

After the integrated checks and first full restoration, the driver creates an
immutable candidate report. Under a new common lock and temporary external
preparation, it runs the actual `--mode performance --qualification CANDIDATE
--warmup formal` session and independently audits its saved results. This is a
diagnostic qualification session using frozen excluded inputs. Its new proof
files do not rewrite the candidate's already hashed artifacts. After this second
restoration succeeds, the same candidate bytes are published as
`qualification.json`. Any performance validation, restoration or publication
failure removes both candidate and final report paths. Failed raw evidence is
retained, but does not leave an available performance qualification.

The `tests.run=1` count in a category summary records one executed independent
category audit, containing multiple assertions. It is not a substituted CPU/GPU
test-suite count. Original JUnit counts, including every skipped CPU-suite GPU
test, remain in the frozen plan and original JUnit files. Separately selected
GPU correctness tests must execute with zero skips.
The pinned M1 input file is copied into this attempt. Both existing GPU opt-in
flags and the common lock path are explicitly set; CPU-stage opt-ins are cleared.
The CPU suite uses the supplied measured LUT directory and permits only the
existing 103 integration GPU skips. The separately selected 102 cache/Prefill
GPU checks must all pass. Their Prefill report is written inside this attempt.
`--baseline-root` identifies the independent reference Python helpers separately
from the KIVI checkout. Both helper hashes are checked before correctness tests,
recorded in the attempt, and checked again before execution. The AGX2 reference
root is `/home/agx31/kvolt-m1-reference`; it is never inferred from `--kivi-root`.

## Per-session formal preparation

`--warmup formal` runs outside timed requests. It uses actual frozen inputs and
all five Gs through natural EOS or their largest requested output cap. Contexts
with different prompt bytes or metadata remain distinct. The ledger records
visited cache/residual shapes, forwards, token IDs, cache release and restoration.
Natural EOS is never suppressed to force coverage of unreachable shapes.

Development mode can run formal preparation to gather qualification evidence.
That does not itself grant performance eligibility. Performance mode requires
a valid report, measured AGX2 LUTs, collected power, native profile `agx2` and
formal preparation. Inputs are limited to the report's explicit length/maxN
coverage; no extrapolated qualification is inferred.

```sh
kvolt run --mode performance --qualification M4/qualification.json \
  --warmup formal --collect-power --native-profile agx2 \
  --model-dir SNAPSHOT --lut-dir SEALED_AGX2_LUTS \
  --device-profile DEVICE.json --power-profile POWER.json \
  --kivi-root KIVI --manifest EVALUATION_MANIFEST.json --output-dir NEW_SESSION
```

The evidence gate runs before hardware acquisition or GPU imports. The Runner
rechecks locked physical identity, actual device/collector settings and request
coverage. Preparation failure or interruption prevents timed performance work.
Formal evaluation execution, quality statistics and energy/SLO effect claims
remain separate work after M4 qualification.
