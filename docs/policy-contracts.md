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

PC queries `prefill_target(L)` first, validates the target, then reads its
integer-nanosecond `prefilter_ns` clock. `prefill_within_target(L, target_s)`
returns full details only for all `ttft_s <= target_s` candidates, or every exact
global minimum-TTFT tie if none pass. PC validates the lookup and reads `tp_ns`.
Its `budget_s` equals the target; elapsed control time remains measured but does
not reduce eligibility. Both clock boundaries must be nondecreasing. This
2026-11-08 user-directed rule replaces historical elapsed-budget PC selection.
DC checks Nmax/first-EOS before lookup, then performs
quality lookup, Knee detection, combined Decode predictions/target lookup,
and its clock. Controllers revalidate combined wrapper type, requested length,
finite positive target and nested bracket metadata; each row bracket must match
the combined bracket. Malformed output is fatal without a separate-API fallback.
Only then are budgets and candidates selected. Negative DC budgets stay negative.
Clock subtraction happens as integers before conversion to seconds. A clock
moving backwards is a contract error, not a Knee recovery.

Controllers return frozen decisions with candidate eligibility, selection and
reasons. PC records `selection_rule="target_ttft"` and only queried full
`PrefillPrediction` rows; it does not emit latency-only placeholders. DC retains
all 40 predictions. `prefilter_ns` records target validation independently of
final `tp_ns`. The auditor checks the exact sparse PC set against a repository
when available; without source evidence it checks only internal consistency.
Historical PC traces without `selection_rule` retain full/partial-energy
elapsed-budget auditing. Unknown rule markers are rejected.
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
40 candidate flags retain the original active/time filter. Invalid LUT data remains an error.
Missing/wrong dependency versions fail adapter startup with
PolicyDependencyError instead of masquerading as normal Knee detection.
