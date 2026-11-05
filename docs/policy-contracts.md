# M2 policy contracts

`policy/types.py` is the shared CPU-only interface. It does not import NumPy,
SciPy, kneed, torch or a GPU model. Existing M1 contracts and arithmetic remain
unchanged. Python float is binary64; selection compares the full values directly.

`LUTRepository` provides separate `prefill`, `quality`, `decode`,
`prefill_target`, and `decode_target` methods. `preflight(L)` checks full support
without calling those online interpolation methods. All immutable predictions
carry L, original candidate identifiers and a containing interpolation bracket.
Quality uses raw RULER mean_score in [0,1], never retention q. The raw metadata
and source pins are retained in `LUTProvenance`; an adapter may not turn
runtime_eligible=false into a declaration of formal Runtime qualification.

PC calls Prefill lookup and the Prefill target before reading its injected
integer-nanosecond clock. DC checks Nmax/first-EOS before lookup, then performs
quality lookup, Knee detection, all Decode lookup, Decode target, and its clock.
Only then are budgets and candidates selected. Negative budgets stay negative.
Clock subtraction happens as integers before conversion to seconds. A clock
moving backwards is a contract error, not a Knee recovery.

Controllers return frozen decisions with every candidate, its eligibility,
selection and reasons. PC and DC do not write files or device settings. CLI
diagnosis injects assumed times and marks them synthetic; later M3 passes the
same host monotonic clock used by the Backend. There is no request-result cache
that moves online interpolation or detection ahead of its phase boundary.

Knee states follow current Method: found, not_found, error_recovered. Detected
groups are sorted unique original G values. Empty/flat results use all G;
invalid detector outputs recover to all G. Invalid LUT data remains an error.
Missing/wrong dependency versions fail adapter startup with
PolicyDependencyError instead of masquerading as normal Knee detection.
