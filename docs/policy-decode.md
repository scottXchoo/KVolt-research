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
`KeyboardInterrupt` and `SystemExit` propagate. DC preserves these detector
results: normal empty/flat results activate the median G, while detector recovery
activates all G.

`DecodeController(repo, knee_adapter=None, *, clock=time.monotonic_ns)` constructs
the actual adapter by default. `choose(L,nmax,first_token_id,t1_ns)` validates
strict integer L in `[1024,8192]`, Nmax in `[1,512]`, first token in `[0,128255]`,
and nonnegative integer t1. EOS IDs are `(128001,128008,128009)`. First EOS or
Nmax 1 returns `status='skipped'`, `selection_path='skipped'` and
`skip_reason='first_token_eos'` or `'nmax_one'` (EOS takes precedence). This path
calls no repository method, detector or clock. Selection, target, budget, elapsed,
td, Knee and feasibility are null; Quality and candidates are empty. It preserves
the supplied t1.

The active online order is Quality lookup, Knee detection and active G selection,
`repo.decode_target(L)` Reference lookup, td clock, elapsed/budget calculation,
`repo.decode_within_budget(L, groups, budget_s)`, then energy selection:

```python
elapsed_ns = td_ns - t1_ns
budget_s = target_s - (elapsed_ns / 1_000_000_000) / (nmax - 1)
```

This 2026-11-08 user revision supersedes Method's Decode-before-td order.
Integer subtraction precedes conversion to seconds. The positive finite target
is validated before td; backwards or noninteger clocks fail before the Decode
query. Negative budgets are preserved. Failed detection/recovery costs remain
before td. Conditional Decode lookup and selection costs occur after td, so Bd
does not subtract them, while actual Runner t1-to-tN timing still includes them.
Bd is a planning rule, not an actual SLO upper bound. Full support preflight
belongs to Runner and must not move online interpolation/detection before the
first token.

The conditional query interpolates TPOT for active G first and returns all rows
with TPOT <= Bd, interpolating energy only for those rows. If none pass, it
examines the remaining global candidates, reuses TPOT probes, and returns all
exact global minimum-TPOT ties with their energies. It never queries energy for
rejected rows or re-queries all details to build a trace. Existing full Decode queries remain available for offline comparisons.

DC validates an immutable nonempty tuple of at most 40 unique original (G,f)
rows, requested L, containing shared brackets and finite metrics. A normal
return must consist entirely of active/time-eligible rows; a fallback return
must contain no eligible rows and only equal TPOT values. Mixed eligibility,
unequal fallback times and malformed rows fail after td instead of being
silently filtered. Completeness and global minimality belong to the repository
query/preflight contract; the controller cannot establish omitted candidates
without defeating conditional lookup.
Offline source-backed auditing independently evaluates the full 40-row grid to
prove returned-set completeness, global fastest ties and the selected argmin.
An audit without the repository can check only internal trace consistency.
Validation constructs fresh `Bracket` and prediction objects, then uses those
canonical copies for detection, comparisons and returned traces. It revalidates
nested bracket endpoints and weight, normalizes accepted numeric metrics to
binary64, and leaves foreign source objects unchanged. Thus a foreign integer
`2**53+1` compares as its binary64 value `float(2**53)`, preserving the exact-tie
rules. Invalid Quality metadata fails before detector access; invalid Decode
metadata fails at the conditional query boundary after td. Neither is a detector
recovery.

Detected Knees restrict the initial active G set. For `not_found` (`no_knee` or
`flat_quality`), DC activates only the middle original candidate in sorted
`GROUPS=(8,16,32,64,128)`: G=32. This is the 2026-11-08 user revision to Method's
previous all-G normal-miss rule. The recorded Knee remains `not_found` with empty
detected groups; DC does not fabricate a detected Knee. `error_recovered` still
activates all five G. Eligible combinations satisfy the active G
condition and `tpot_s <= budget_s`. A nonempty eligible set minimizes
`(energy_j_per_token,-quality,tpot_s,G,frequency_mhz)`. Otherwise DC releases the
active G restriction, including a median-only restriction, and selects from the
returned global fastest TPOT ties by
`(tpot_s,-quality,energy_j_per_token,G,frequency_mhz)`. Both paths compare full
Python binary64 values directly, with no rounding, approximate ties or energy
normalization. `predicted_feasible` is the chosen TPOT comparison with the budget;
it may be true on a fallback outside the original Knee. It does not assert an
actual measured SLO result.

Active decisions use `status='selected'` and selection path `energy_min` or
`latency_fallback`. The frozen decision preserves raw Quality predictions, Knee
status, target, budget, integer times, selected Quality, and only the queried
detail candidates (1..40). `selection_rule="budget_filtered_decode"` marks this
contract, including skipped decisions. Historical unmarked traces retain their
full-40 interpretation. Flags describe the original active G/time filter even
after fallback, so a
selected fallback outside the detected or median G has `eligible=False`. The
existing `active_knee` flag records membership in that active G set. Reasons include
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
