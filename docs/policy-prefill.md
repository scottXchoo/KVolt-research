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
2. Query `repo.prefill_target(L)` first. Validate the positive finite target.
3. Read `prefilter_ns = clock()` after target validation.
4. Query `repo.prefill_within_target(L, target_s)`. The immutable `PrefillLookup`
   contains full `PrefillPrediction` details for every frequency with
   `ttft_s <= target_s`, including equality. If none pass, it contains every
   exact tie at the global minimum TTFT for the existing latency fallback.
5. Validate the lookup wrapper, requested L, matching target, shared bracket,
   unique original frequencies and full finite metrics. Revalidate foreign
   frozen rows and nested brackets. Passing and rejected rows cannot be mixed;
   a fallback result must contain only equal-TTFT rows.
6. Read `tp_ns = clock()` after lookup validation and record
   `elapsed_ns = tp_ns - t0_ns`. Keep `budget_s = target_s`.
7. Select the minimum raw energy among passing rows, or use latency fallback,
   and record only the returned detailed candidates.

This is the user's 2026-11-08 target-based PC rule. The published Method's
historical PC rule subtracted elapsed control time from the target. Current PC
selection does not subtract elapsed time, even when lookup time exceeds the
target. Both clocks and elapsed time remain measured evidence, and actual TTFT
still includes control and execution costs. DC retains its existing elapsed-time
budget rule.

The feasible set contains rows with `ttft_s <= target_s`. For a nonempty set,
`energy_min` selects the minimum `(energy_j, ttft_s, frequency_mhz)`. Otherwise
`latency_fallback` selects the minimum `(ttft_s, energy_j, frequency_mhz)` across
the global minimum-TTFT ties. Python binary64 values are compared directly;
there is no rounding, approximate equality, epsilon or energy normalization.
Targets and metrics are normalized to binary64 before comparison. Invalid or
backward clocks from `t0_ns` or between the two reads raise `ContractError`.

The filtered repository query uses one performance-length bracket for its rows
and supplied threshold. Loaded tables whose TTFTs are nonincreasing with
frequency at every length knot use a binary boundary search. That property is
preserved by linear interpolation. Valid nonmonotonic tables use an exact scan;
they are not rejected or silently treated as monotonic. Energy is interpolated
only for returned rows, and TTFT probes are reused within the query. The online
controller does not request all eight latencies, a combined all-frequency lookup,
or a second detail lookup. Full LUT support remains mandatory before PC entry.
No Quality, Decode, Knee or preflight method runs inside this phase.

The frozen `PrefillDecision` carries `selection_rule="target_ttft"`, the selected
prediction, target, equal budget, integer elapsed time, `t0_ns`, `prefilter_ns`,
`tp_ns`, selection path and `predicted_feasible`. Its immutable candidate tuple
is sorted by frequency and contains only queried full predictions with
`meets_budget`, `selected` and reasons. There are no latency-only placeholders.
A fallback row has `time_budget_exceeded`; every unchosen row has `not_selected`;
the chosen fallback also has `latency_fallback`. Predicted feasibility remains
separate from actual TTFT SLO evaluation against the frozen target.

The offline auditor rejects unknown selection rules and validates sparse trace
metrics, clock boundaries, flags, reasons and selection. With a repository it
independently derives the exact passing set, or all global fastest ties, from
the full source LUT and checks every value and target. Without a repository it
checks internal consistency only; it cannot establish that no eligible row or
fastest tie was omitted. Historical records with no `selection_rule` retain the
old elapsed-budget audit: full-detail records and partial-energy records with
`prefilter_ns` remain readable.

`tests/test_policy_prefill.py` covers target-first order, target equality,
nonzero and large elapsed times, sparse traces, exact ties, binary64 neighbors,
clock ordering and integer precision, malformed foreign inputs and repeated
online calls. The filtered LUT tests additionally check monotonic boundary
search, nonmonotonic scans and interpolated lengths. Audit tests retain legacy
coverage and reject sparse trace tampering, including source-proven omissions.
A subprocess import guard rejects NumPy/SciPy/kneed/torch/transformers imports.
Sealed measured integration uses `KVOLT_LUT_BUNDLE` (or the sibling source bundle
when present), with independently transcribed knots and their midpoint.
These CPU checks do not establish GPU execution, runtime speedup, actual SLOs
or formal runtime qualification. Source LUT and Reference bytes are unchanged.
