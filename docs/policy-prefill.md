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
2. Query `repo.prefill(L)` and validate all eight original frequencies.
3. Query and validate `repo.prefill_target(L)`.
4. Read `tp_ns = clock()` and subtract `t0_ns` using integer arithmetic.
5. Compute `budget_s = target_s - elapsed_ns / 1_000_000_000`.
6. Select a prediction and record every candidate's decision evidence.

Prefill predictions must be an immutable tuple containing exactly one row
for each of `408, 510, 612, 714, 816, 918, 1020, 1122` MHz. Every row must
have the requested length, valid finite metrics and a valid containing
interpolation bracket. The controller revalidates foreign repository rows,
including nested bracket fields. A missing, duplicate, unsupported or invalid
row rejects the request with `ContractError`; valid metric values are
normalized to binary64 before selection. Candidates are never silently
removed and missing values are never replaced by zero. No Quality, Decode,
Knee or preflight method runs inside this online phase.

The feasible set contains rows with `ttft_s <= budget_s`, including equality.
For a nonempty set, `energy_min` selects the minimum
`(energy_j, ttft_s, frequency_mhz)`. Otherwise `latency_fallback` selects the
minimum `(ttft_s, energy_j, frequency_mhz)` over all eight rows. Python
binary64 values are compared directly. There is no rounding, approximate
equality, epsilon or energy normalization. Negative budgets stay negative.
An invalid or backward clock raises `ContractError`, rather than selecting a
fallback.

The frozen `PrefillDecision` contains the selected prediction, target, budget,
integer elapsed time, `t0_ns`, `tp_ns`, selection path and
`predicted_feasible`. Its immutable candidate tuple is sorted by original
frequency and retains every prediction with `meets_budget`, `selected` and
reasons. A failing time condition adds `time_budget_exceeded`; every unchosen
row adds `not_selected`; the chosen fallback row adds `latency_fallback`.
For PC a fallback prediction cannot meet the budget. Predicted feasibility
is separate from actual TTFT SLO evaluation against the frozen target.

`tests/test_policy_prefill.py` covers manual energy/latency choices, both
tie orders, binary64 neighbors, exact budget equality, negative budgets,
clock ordering and integer precision, malformed foreign inputs and repeated
online calls. A subprocess import guard rejects NumPy/SciPy/kneed/torch/
transformers imports. Sealed measured integration uses `KVOLT_LUT_BUNDLE`
(or the sibling source bundle when present), with independently transcribed
1024/2048 knots and their 1536 midpoint. These CPU policy checks do not
establish GPU execution, actual SLOs or formal runtime qualification.
