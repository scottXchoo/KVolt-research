# LUT lookup data and Runtime repositories

## Latest default for LUT value lookups

Use [`data/luts/latest/`](../data/luts/latest/) for future LUT value lookups:
[`prefill.json`](../data/luts/latest/prefill.json),
[`decode.json`](../data/luts/latest/decode.json) and
[`quality.json`](../data/luts/latest/quality.json). These are the latest
user-provided AGX1 compact tables, received on 2026-11-07 and preserved byte
for byte. Their positional row contracts, grids, source documents, hashes and
limits are recorded in [`data/luts/README.md`](../data/luts/README.md).
Use the older AGX2 sealed bundle for explicitly requested historical
reproduction, rather than as the default source of current LUT values.

## Latest compact Runtime adapter

`CompactLUTRepository(compact_path, *, reference_repository)` loads the latest
AGX1 predictions. The recovered original `manifest.json` is copied byte for
byte into `data/luts/latest/`; its SHA-256 is
`86b2343eed65432d1ee1f071fa8188a33789bdcf0309f325e689ec0bae37591a`.
The adapter checks that pin and the three table hashes/byte counts before
validating schema, units, positional row widths, strict integer identifiers,
all 459 cells, positive times, nonnegative energies and raw quality in [0,1].
It shares the length-only interpolation and support contract below.

The original compact export contains no FP16 target curves. An explicit
`reference_repository` must be an actual sealed `MeasuredLUTRepository`.
Each target knot is copied unchanged from that repository's `prefill_target`
and `decode_target`; no compact performance cell substitutes for a target.
The current runtime device and its frozen FP16 goals remain AGX2. Applying
AGX1 predictions against those separately supplied AGX2 references is an
explicit development comparison, with a device mismatch recorded.

`load_lut_repository(path, *, reference_lut_dir=None)` in
`kvolt_runtime.policy.compact` selects the sealed or compact distribution.
Compact inputs require `reference_lut_dir` pointing to the sealed AGX2 bundle.
Missing/unknown/ambiguous formats fail; there is no synthetic fallback.
Passing a reference directory for a sealed input or `--mock` is rejected.
Both CLI commands expose the explicit reference option:

```sh
kvolt diagnose --lut-bundle data/luts/latest --reference-lut-dir /path/to/agx2-repeat5-20261103 \
  --length 1024 --max-new-tokens 1 --pc-elapsed-ns 0
kvolt run --lut-dir data/luts/latest --reference-lut-dir /path/to/agx2-repeat5-20261103 \
  --prompt-file prompt.txt --model-dir /path/to/0e9e39f249a16976918f6564b8830bc894c89659 \
  --device-profile device.json --native-profile agx2 --output-dir /path/to/new-output
```

Compact provenance has `kind=measured`, `source_device=agx1`, the original
manifest/table hashes and `source_commit=null` (the original manifest does not
identify a commit). Metadata preserves the entire original compact manifest,
including `performance_runtime_accuracy_or_SLO_validated=false` and
`quality_runtime_eligible=false`. `target_repository` separately retains the
sealed AGX2 source/commit, hashes, source conditions and FP16 curve provenance;
`target_source_device=AGX2` and `cross_device_reference=true` identify the
comparison. The runtime also records prediction source versus execution
device in `lut_device_conditions`. No Runtime accuracy, SLO or AGX2 formal
qualification is established by loading these AGX1 measurements. Existing
AGX2 qualification evidence cannot qualify this changed prediction source.

## Existing M2 sealed measured Runtime adapter

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

Online DC queries `quality(L)` and Knee first, then `decode_target(L)`, records
td and computes Bd before calling `decode_within_budget(L, groups, budget_s)`.
The latter accepts finite signed budgets and a nonempty tuple of unique
original G, canonicalized in sorted order. It interpolates active-G TPOT first
and returns every TPOT <= Bd row; if none qualify, it scans the remaining global combinations and returns
all exact minimum-TPOT ties. TPOT probes are reused within the request; energies
are interpolated only for returned rows. The result is an immutable tuple of
full `DecodePrediction` details, with no target query inside this API. Full
source-grid constructor/preflight validation remains mandatory. No G/f
interpolation, extrapolation, approximate boundary or missing-value substitution
is introduced. See [DC order and budget](policy-decode.md).

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
Compact tests additionally compare every original numeric row, validate
separate target provenance and exercise both CLI paths and rejection before
GPU ownership when old AGX2 qualification is supplied for the new AGX1 LUT.
