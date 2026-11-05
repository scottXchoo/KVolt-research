# M2 measured and synthetic LUT repositories

`MeasuredLUTRepository(bundle_path)` loads the published AGX2 repeat5 bundle
from source commit `d6cf87a60984aa167c5b5d72de354fd3d587c2d5`. Its
`bundle-manifest.json` must have SHA-256
`3b9755054b485ce81350bfbd4a675c4deff871c92784e5a20370211cac4029d9`.
The constructor verifies all ten sealed files, including both CSVs, and rejects
unknown manifests, missing files, changed bytes and paths escaping the bundle.
It rejects duplicate JSON keys and nonfinite JSON numbers. The sealed source
pin establishes the accepted distribution; this adapter does not accept a
user-supplied replacement manifest hash or automatically fall back to Mock.

The canonical grids contain 64 Prefill cells, 320 Decode cells and 75 Quality
cells. Performance/reference lengths are 1024 through 8192 by 1024; Quality
lengths use steps of 512. Frequencies are 408, 510, 612, 714, 816, 918, 1020,
1122 MHz and groups are 8, 16, 32, 64, 128. Every required candidate and target
is validated at construction. `preflight(L)` validates integer length support
without calling online lookup, interpolation or detection. Full support is
required even when a later controller skips Decode for Nmax=1 or first EOS.

Online methods are `prefill(L)`, `quality(L)`, `decode(L)`, `prefill_target(L)`
and `decode_target(L)`. Predictions are immutable and retain the request length,
original candidate and `Bracket(lower, upper, weight)`. Exact knots return the
original Python binary64 value directly; intermediate values use
`(1 - weight) * lower_value + weight * upper_value`. Only length is interpolated.
There is no extrapolation, G/f interpolation, rounding, smoothing, or
request-result interpolation cache across phases.

Prefill uses published PWL `ttft_s` and `prefill_energy_j`; Decode uses
`tpot_actual_s` and `epsilon_actual_j_per_token`, preserving the published
measurement boundaries. Quality uses `entries[].mean_score` in [0,1], never
FP16 retention `q`. Targets use `R_p_ttft_s` and `R_d_tpot_s`: independent raw
maxima over three FP16 requests per length, with multiplier 1.0.

`provenance` retains measured/synthetic kind, source device, source commit,
manifest hash and all ten file hashes. Its `metadata_json` includes the source
bundle, original Quality object, export/validation metadata and reference
metadata. In particular, `runtime_eligible=false`,
`runtime_validation_required=true` and the source limitations remain unchanged.
Loading the bundle does not establish new Runtime accuracy, SLO or phase
frequency transition qualification.

`MockLUTRepository(scenario='baseline')` is explicitly synthetic. Scenarios
`flat_quality`, `nonmonotonic`, and `exact_ties` provide deterministic boundary
fixtures. Optional keyword arguments `prefill`, `decode`, `quality`,
`prefill_target`, `decode_target` replace a complete table; unspecified tables
keep scenario defaults. Table keys are `(L,f)`, `(L,G,f)`, `(L,G)`, `L`, and `L`,
respectively. Performance cells contain `(time_seconds, energy)`; Quality and
target cells contain scalar numbers. For Decode, energy means joules per token.
Fixtures must cover the entire canonical grid with strict integer identifiers,
finite positive times/targets, nonnegative energies and raw quality in [0,1].
Booleans, strings and missing values are rejected, never coerced. Supplied
mappings are copied into private immutable tables at construction.

CPU tests discover real data through `KVOLT_LUT_BUNDLE`, or the known sibling
`KVolt-lut-source/data/luts/agx2-repeat5-20261103`. If unavailable, measured
integration tests explicitly skip. They compare all exact/midpoint lookups,
raw five-repeat means/request ratios, raw independent reference maxima, source
flags and sealed files. No GPU or scientific package is imported by the loader.
