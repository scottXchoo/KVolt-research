# Prefill Controller

`PrefillController(repo: LUTRepository, *, clock=time.monotonic_ns)` exposes
`choose(length: int, t0_ns: int) -> PrefillDecision`. It runs on the CPU and
imports no numerical, model or GPU libraries. It does not write files or device
settings. The caller performs full-support `repo.preflight(L)` before the
measured phase. Every request repeats these six steps without decision caching:

1. Runner records `t0_ns` at PC entry using the Backend's host monotonic clock.
   The controller validates integer `L` in `[1024, 8192]` and nonnegative integer
   `t0_ns`.
2. Query `repo.prefill_target(L)` for the Reference LUT's frozen TTFT target
   `target_s`, and validate that it is positive and finite.
3. Immediately read `tp_ns = clock()`. No Prefill candidate query precedes it.
4. Subtract the integer nanosecond timestamps first:
   `elapsed_ns = tp_ns - t0_ns`, then compute
   `budget_s = target_s - elapsed_ns / 1_000_000_000` once. Preserve finite zero
   and negative budgets without clamping.
5. Query `repo.prefill_within_budget(L, budget_s)` for all full-detail candidates
   with `ttft_s <= budget_s`, including equality. If none pass, return all exact
   ties at the global minimum TTFT for the existing latency fallback. Validate
   the returned `PrefillBudgetLookup`, then record `lookup_end_ns = clock()`.
   This second reading measures lookup and validation; it does not recompute or
   reduce the budget.
6. Select the minimum raw Prefill energy among passing candidates using
   `(energy_j, ttft_s, frequency_mhz)`. If none pass, select the minimum
   `(ttft_s, energy_j, frequency_mhz)` across all global minimum-TTFT ties.

This is the user's corrected 2026-11-08 Reference-time budget rule. Reference
lookup and validation costs affect the threshold. Candidate lookup, filtering,
selection and device application occur after `tp_ns`; they are not subtracted
again from this planning budget. Actual TTFT still includes those costs. This
rule does not establish an actual TTFT SLO bound. DC retains its existing rule.

The repository's immutable `PrefillBudgetLookup` contains `length`,
`predictions`, the supplied signed finite `budget_s` and a shared interpolation
`bracket`. Each prediction is a full `PrefillPrediction` with valid finite
metrics and the requested length. The controller revalidates foreign frozen
wrappers, nested brackets, row types, matching budget, unique original
frequencies and matching brackets. It rejects mixed passing/rejected subsets;
a fallback result must contain only equal-TTFT rows. Full LUT support remains
mandatory before PC entry. No Quality, Decode, Knee or preflight method runs
inside this measured phase.

The filtered query reuses one performance-length bracket. Tables whose TTFTs
are nonincreasing across frequency at every knot use binary boundary search;
linear interpolation preserves that property. Valid nonmonotonic tables use an
exact scan. TTFT probes are reused within the query, and energy is interpolated
only for returned candidates. The controller never requests all eight latencies,
a combined all-frequency lookup, the positive-target filtering API, or a second
detail lookup. Python binary64 comparisons are exact: no rounding, approximate
equality, epsilon or energy normalization changes eligibility or ties.

The frozen decision carries `selection_rule="reference_budget"`, positive
`target_s`, signed `budget_s`, selected prediction, `t0_ns`, `tp_ns`,
`elapsed_ns`, `lookup_end_ns`, selection path and `predicted_feasible`.
`prefilter_ns` is an exact compatibility alias for `tp_ns`. Require
`t0_ns <= tp_ns <= lookup_end_ns`; invalid or backward clocks raise
`ContractError`. The immutable candidate tuple is sorted by frequency and
records only queried full predictions with `meets_budget`, `selected` and
reasons. Fallback rows have `time_budget_exceeded`; every unchosen row has
`not_selected`; the chosen fallback also has `latency_fallback`. Predicted
feasibility is separate from actual SLO evaluation against the frozen target.

The offline auditor validates the Reference boundary, fixed budget, lookup end,
application after lookup completion, metrics, candidate flags, reasons and
selection. With a repository it derives the exact passing set or all fastest
ties from the full source LUT and checks all returned values and the target.
Without a repository it checks internal consistency only; it cannot prove no
eligible row or fastest tie was omitted. Unknown selection rules are rejected.

Historical `selection_rule="target_ttft"` traces retain their direct-target
meaning (`budget_s=target_s`, elapsed measured through the candidate lookup).
Historical unmarked traces retain their full-detail or partial-energy
elapsed-budget audit, including their post-lookup budget boundary. Neither
historical rule is reinterpreted as `reference_budget`. Compatibility
`prefill_within_target(L, target_s)` remains positive-target-only; current online
PC uses the distinct signed-budget API.

Tests cover the six-step order, Reference cost, target-passing/budget-failing
rows, exact equality, expensive candidate lookup without a second subtraction,
zero/negative fallback, exact ties, sparse evidence, malformed inputs, backward
clocks, nanosecond precision and repeated requests. Audit tests preserve both
historical policies and reject current boundary/candidate tampering. Filtered
LUT tests cover monotonic search, nonmonotonic scans and interpolated lengths.
A subprocess import guard rejects NumPy/SciPy/kneed/torch/transformers imports.
These CPU checks do not establish GPU execution, runtime speedup, actual SLOs
or formal qualification. Original LUT and Reference bytes are unchanged.
