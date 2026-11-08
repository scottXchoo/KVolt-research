# Prefill Controller

`PrefillController(repo: LUTRepository, *, clock=time.monotonic_ns)` exposes
`choose(length: int, t0_ns: int) -> PrefillDecision`. It runs on the CPU and
imports no numerical, model or GPU libraries. The controller does not write
files or device settings. M3 supplies the Backend's host monotonic clock and
records `t0_ns` at PC entry. A deterministic diagnostic may inject assumed
integer-nanosecond times and must label them synthetic.

The caller performs `repo.preflight(L)` before entering the measured phase.
Every call to `choose` then performs these operations, without caching:

1. Validate integer `L` in `[1024, 8192]` and nonnegative integer `t0_ns`.
2. Query `repo.prefill_lookup(L, latency_only=True)` once for the target and
   all eight TTFTs, using a shared interpolation bracket. Validate the immutable
   `PrefillLookup` wrapper, requested L, positive finite target, bracket and rows.
3. Read `prefilter_ns = clock()` and compute the screening budget as
   `target_s - (prefilter_ns - t0_ns) / 1_000_000_000`.
4. Query `repo.prefill_candidates(L, frequencies)` only for frequencies whose
   interpolated TTFT is at most the screening budget. If none pass, query only
   all ties at the global minimum TTFT for the latency fallback's energy tie.
5. Validate those details against the requested subset and earlier TTFT/bracket.
6. Read `tp_ns = clock()` after all queries and compute the final
   `budget_s = target_s - (tp_ns - t0_ns) / 1_000_000_000`.
7. Recheck the final budget, select a prediction and record every candidate.

The combined lookup's `predictions` must be an immutable tuple containing
exactly one `PrefillLatencyPrediction` for each of `408, 510, 612, 714, 816, 918, 1020,
1122` MHz. Its `energy_j` is null because energy has not been queried. Details
must be an immutable tuple of `PrefillPrediction` rows covering exactly the
requested subset, with finite energy and exactly the earlier TTFT and bracket.
Every row must have the requested length, valid metrics and a valid containing
interpolation bracket. The controller revalidates foreign repository rows,
including the wrapper and nested bracket fields. Every row bracket must match
the lookup bracket. A foreign wrapper or invalid target fails before the first
clock or details query. The controller never falls back to separate target or
latency methods. Missing, duplicate, unsupported, inconsistent
or invalid rows reject the request with `ContractError`; valid metric values
are normalized to binary64 before selection. Full LUT support is still checked
before the measured phase. Null in the decision trace means energy was not
queried, never that a source cell was missing or replaced by zero. No Quality,
Decode, Knee or preflight method runs inside this online phase.

The feasible set contains rows with `ttft_s <= budget_s`, including equality.
For a nonempty set, `energy_min` selects the minimum
`(energy_j, ttft_s, frequency_mhz)`. Otherwise `latency_fallback` selects the
minimum `(ttft_s, energy_j, frequency_mhz)` over all eight frequencies. Every
global minimum-TTFT tie is included in a nonempty screening set; an empty
screening set explicitly fetches those ties. The fallback therefore needs no
extra lookup after `tp_ns`. A row passing screening can still fail the final
budget because detailed lookup consumed time. Python
binary64 values are compared directly. There is no rounding, approximate
equality, epsilon or energy normalization. Negative budgets stay negative.
An invalid clock or backward movement from `t0_ns` or between the two clock
reads raises `ContractError`, rather than selecting a fallback.

The frozen `PrefillDecision` contains the selected prediction, target, budget,
integer elapsed time, `t0_ns`, `prefilter_ns`, `tp_ns`, selection path and
`predicted_feasible`. Its immutable candidate tuple is sorted by original
frequency and retains every prediction with `meets_budget`, `selected` and
reasons. Queried rows retain their full prediction, including rows that no
longer meet the final budget. Other rows retain latency-only evidence with
`energy_j=null`. The selected prediction always has a numeric energy. The
offline auditor verifies the screening timestamps and exact queried subset,
then checks available values against the pinned LUT; historical full-detail
records without `prefilter_ns` remain readable. A failing time condition adds
`time_budget_exceeded`; every unchosen
row adds `not_selected`; the chosen fallback row adds `latency_fallback`.
For PC a fallback prediction cannot meet the budget. Predicted feasibility
is separate from actual TTFT SLO evaluation against the frozen target.

`tests/test_policy_prefill.py` covers manual energy/latency choices, both
tie orders, binary64 neighbors, exact budget equality, negative budgets,
clock ordering and integer precision, malformed foreign inputs and repeated
online calls, partial queries, final-budget expiry and backwards movement
between screening and completed lookup. A subprocess import guard rejects NumPy/SciPy/kneed/torch/
transformers imports. Sealed measured integration uses `KVOLT_LUT_BUNDLE`
(or the sibling source bundle when present), with independently transcribed
1024/2048 knots and their 1536 midpoint. These CPU policy checks do not
establish GPU execution, actual SLOs or formal runtime qualification.

The combined lookup returns the Reference target and all eight interpolated
TTFTs before screening and detailed energy lookup. There is no TTFT index or
assumed frequency monotonicity.

Only energy/detail lookup is reduced; no runtime speedup is claimed without
measurement. Source LUT and reference files are unchanged.
