# M2 policy contracts

`policy/types.py` is the shared CPU-only interface. It does not import NumPy,
SciPy, kneed, torch or a GPU model. Existing M1 contracts and arithmetic remain
unchanged. Python float is binary64; selection compares the full values directly.

`LUTRepository` provides `prefill`, `quality`, `decode`, `prefill_target`,
`decode_target`, and `decode_within_budget(L, groups, budget_s)` methods. `preflight(L)` checks full support
without calling those online interpolation methods. All immutable predictions
carry L, original candidate identifiers and a containing interpolation bracket.
Quality uses raw RULER mean_score in [0,1], never retention q. The raw metadata
and source pins are retained in `LUTProvenance`; an adapter may not turn
runtime_eligible=false into a declaration of formal Runtime qualification.

PC calls Prefill lookup and the Prefill target before reading its injected
integer-nanosecond clock. DC checks Nmax/first-EOS before lookup, then performs Quality lookup, Knee
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
reasons. DC records `selection_rule="budget_filtered_decode"` and only returned
detail rows (1..40). Unmarked historical full-40 traces retain legacy auditing.
With a repository, the auditor independently checks the full grid for exact
returned-set completeness, global fastest ties and the selected argmin; without
source evidence it checks internal consistency. Unknown rule markers are rejected.
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
