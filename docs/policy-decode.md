# M2 Knee and Decode policy

`KneeAdapter()` loads and checks `kneed==0.9.0` and `SciPy==1.15.3` at startup.
Import failures or version mismatches raise `PolicyDependencyError`; they do not
produce a detector recovery. The target Jetson environment uses NumPy 1.26.4.
An explicit `locator_factory` is an isolated unit-test seam, not a production
alternative to the pinned detector.

`detect(quality)` requires five unique original G values `(8,16,32,64,128)`, one
supported L, finite raw RULER mean scores in `[0,1]`, and valid containing
interpolation brackets. Invalid Quality data raises `ContractError` before the
detector recovery boundary. It preserves raw nonmonotonic values and original G
coordinates. Exactly flat values return `not_found/flat_quality` without invoking
the locator. Other curves use `S=1.0`, `curve='concave'`,
`direction='decreasing'`, `interp_method='interp1d'`, and `online=True`.

The adapter reads `all_knees`, which kneed 0.9.0 exposes as a **list**. It accepts
list, tuple, set or frozenset containers, validates every finite numeric original
candidate, then applies Method's `SET(result)` and sorts unique G values. Exact
NumPy integer/float candidates are accepted; booleans, strings, dictionaries,
generators, arrays, nonfinite values and values outside G are invalid. Valid
endpoints and multiple Knees are preserved. An empty supported container returns
`not_found/no_knee`. Detector exceptions or invalid returns yield
`error_recovered`, empty groups, and error type/message with `detector_error`.
`KeyboardInterrupt` and `SystemExit` propagate. Empty groups cause DC to use all G.

`DecodeController(repo, knee_adapter=None, *, clock=time.monotonic_ns)` constructs
the actual adapter by default. `choose(L,nmax,first_token_id,t1_ns)` validates
strict integer L in `[1024,8192]`, Nmax in `[1,512]`, first token in `[0,128255]`,
and nonnegative integer t1. EOS IDs are `(128001,128008,128009)`. First EOS or
Nmax 1 returns `status='skipped'`, `selection_path='skipped'` and
`skip_reason='first_token_eos'` or `'nmax_one'` (EOS takes precedence). This path
calls no repository method, detector or clock. Selection, target, budget, elapsed,
td, Knee and feasibility are null; Quality and candidates are empty. It preserves
the supplied t1.

The active online order is Quality lookup, Knee detection, all 40 Decode lookup,
Decode target lookup, td clock, elapsed/budget calculation, then selection:

```python
elapsed_ns = td_ns - t1_ns
budget_s = target_s - (elapsed_ns / 1_000_000_000) / (nmax - 1)
```

Integer subtraction precedes conversion to seconds. Backwards or noninteger
clocks are fatal. Negative budgets are preserved. Failed detection/recovery costs
remain before td and therefore in elapsed time. Full support preflight belongs
to Runner; it must not move online interpolation/detection before the first token.
DC validates the actual five Quality and 40 Decode rows and positive target, even
with an injected repository. It does not filter away missing or invalid rows.

Detected Knees restrict the initial active G set; `not_found` and
`error_recovered` use all five G. Eligible combinations satisfy the active G
condition and `tpot_s <= budget_s`. A nonempty eligible set minimizes
`(energy_j_per_token,-quality,tpot_s,G,frequency_mhz)`. Otherwise DC releases the
Knee restriction and minimizes all 40 candidates by
`(tpot_s,-quality,energy_j_per_token,G,frequency_mhz)`. Both paths compare full
Python binary64 values directly, with no rounding, approximate ties or energy
normalization. `predicted_feasible` is the chosen TPOT comparison with the budget;
it may be true on a fallback outside the original Knee. It does not assert an
actual measured SLO result.

Active decisions use `status='selected'` and selection path `energy_min` or
`latency_fallback`. The frozen decision preserves raw Quality predictions, Knee
status, target, budget, integer times, selected Quality, and all 40 candidate
records. Flags describe the original Knee/time filter even after fallback, so a
selected fallback outside Knees has `eligible=False`. Reasons include
`knee_excluded`, `time_budget_exceeded`, `not_selected`, and for a selected fallback
`latency_fallback`. These are prediction/selection reasons, not technical request
invalidity. M2 performs no GPU imports, model calls, frequency changes, file
writes, or per-request result caching.

Unit scenarios use explicit factories and manual candidate expectations.
`test_actual_pinned_kneedle_differential` is mandatory: missing/wrong package pins
fail rather than silently skip. Development Python/NumPy differences and target
Python 3.10/NumPy 1.26.4 qualification must be reported separately.
`test_actual_published_quality_curves_differential` checks all 15 Quality knots
and 14 interior lengths against official Kneedle, after the measured adapter
checks bundle pins and hashes. `KVOLT_LUT_BUNDLE` supplies that published bundle;
without it, tests also check the sibling `KVolt-lut-source` checkout. Absence of
the optional data source is reported as a data test skip, never as a dependency
pass. An explicitly configured bad path fails. Published-table Decode examples
inject an explicit no-Knee result and preserve the distinction from actual
Kneedle differential tests.

The initial set-only adapter assumption failed both actual-package tests on the
target and was corrected before acceptance. Installed kneed 0.9.0
`knee_locator.py` documents `all_knees : list` at line 107, initializes `[]` at
line 153 and appends at line 346. Its SHA-256 is
`212aafaf453413649025de834d872c890d5e2815d5dca5367197179fb08e2611`.
This API correction changes neither detector parameters nor selection policy.
