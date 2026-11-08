# M2 policy contracts

`policy/types.py` is the shared CPU-only interface. It does not import NumPy,
SciPy, kneed, torch or a GPU model. Existing M1 contracts and arithmetic remain
unchanged. Python float is binary64; selection compares the full values directly.

`LUTRepository` provides combined `prefill_lookup(L, latency_only=False)` and
`decode_lookup(L)` methods. Frozen `PrefillLookup` and `DecodeLookup` contain
`length`, immutable `predictions`, positive finite `target_s` and a shared
`bracket`. Predictions and Reference target reuse one performance-length bracket
calculation. Separate `prefill`, `prefill_latencies`, `prefill_candidates`,
`quality`, `decode`, `prefill_target`, and `decode_target` remain available for
compatibility. `preflight(L)` checks full support
without calling those online interpolation methods. All immutable predictions
carry L, original candidate identifiers and a containing interpolation bracket.
Quality uses raw RULER mean_score in [0,1], never retention q. The raw metadata
and source pins are retained in `LUTProvenance`; an adapter may not turn
runtime_eligible=false into a declaration of formal Runtime qualification.

PC receives Runner's `t0_ns`, queries `prefill_target(L)` and validates the
positive finite Reference target. It immediately reads `tp_ns`, computes
`elapsed_ns=tp_ns-t0_ns` using integer subtraction, and fixes
`budget_s=target_s-elapsed_ns/1_000_000_000`. `prefill_within_budget(L,budget_s)`
returns `PrefillBudgetLookup(length,predictions,budget_s,bracket)` with all
`ttft_s <= budget_s` full rows or all exact global fastest ties if none pass.
Finite zero/negative budgets are valid. After validation PC records
`lookup_end_ns`; it does not subtract lookup time again. Both clocks must be
nondecreasing. `prefilter_ns` equals `tp_ns` as a compatibility alias. This
2026-11-08 corrected user rule replaces the intermediate direct-target policy.
DC checks Nmax/first-EOS before lookup, then performs Quality lookup, Knee
selection and active G determination, `decode_target(L)` Reference lookup and
positive finite target validation, td clock and Bd calculation, then
`decode_within_budget(L, groups, budget_s)`. This 2026-11-08 user revision moves
Decode lookup after td. The query accepts a finite signed budget and a nonempty
tuple of unique original G and canonicalizes their order. It returns all
active/time-eligible details, or all exact global fastest TPOT ties if none
qualify; energy is queried only
for returned rows. DC revalidates immutable nonempty unique rows, requested L,
finite metrics and containing shared brackets. Mixed eligibility or unequal
fallback TPOT values are fatal. Query/preflight owns completeness and global
minimality; DC does not perform a second full-grid lookup. Invalid targets and
clocks fail before the Decode query; invalid returned rows fail after td.
Negative DC budgets stay negative. Query time is outside Bd's elapsed deduction
and inside Runner's actual t1-to-tN measurement.
Clock subtraction happens as integers before conversion to seconds. A clock
moving backwards is a contract error, not a Knee recovery.

Controllers return frozen decisions with candidate eligibility, selection and
reasons. PC records `selection_rule="reference_budget"` and only queried full
`PrefillPrediction` rows; it does not emit latency-only placeholders. DC records
`selection_rule="budget_filtered_decode"` and only returned detail rows (1..40).
Legacy unmarked full-40 Decode traces retain historical audit semantics.
The auditor validates PC's fixed Reference-time budget and requires device
application after `lookup_end_ns`. It checks exact sparse PC/DC sets against a
repository when available; without source evidence it checks only internal
consistency. For DC it independently evaluates the full source grid, proving
eligible-set completeness or global fastest ties and the selected argmin outside
online control. Historical PC `target_ttft` traces retain budget equal to target;
unmarked PC traces retain full/partial-energy post-lookup budget auditing.
Unknown rule markers are rejected. The positive-target `prefill_within_target`
API remains compatible.
Missing source cells still fail full support validation before PC entry.
PC and DC do not write files or device settings. CLI
diagnosis injects assumed times and marks them synthetic; later M3 passes the
same host monotonic clock used by the Backend. There is no request-result cache
that moves online interpolation or detection ahead of its phase boundary.

Knee states are found, not_found, error_recovered. Detected groups are sorted
unique original G values. The 2026-11-08 user revision makes normal empty/flat
`not_found` results activate only the median original G=32 from sorted
`(8,16,32,64,128)`, preserving `not_found` and empty detected groups in the trace.
Invalid detector outputs still recover to all G. A nonempty active time-eligible
set uses the existing energy/tie rules; an empty one releases the active G
restriction and uses the existing global 40-combination latency fallback. The
returned candidate flags retain the original active/time filter. Invalid LUT data remains an error.
Missing/wrong dependency versions fail adapter startup with
PolicyDependencyError instead of masquerading as normal Knee detection.
