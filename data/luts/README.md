# Latest LUT lookup data

Use [`current/selection.json`](current/selection.json) as KVolt's fixed default
LUT entry point. `kvolt run` and `kvolt diagnose` read it automatically when
`--lut-dir`/`--lut-bundle` and `--mock` are omitted, independently of the working
directory in a source checkout. Installed packages without these data files
fail explicitly; provide a source directory override in that case. It selects
[`agx2-runtime-repeat5/`](agx2-runtime-repeat5/) for
predictions and [`agx2-reference-repeat10/`](agx2-reference-repeat10/) for FP16
targets. The user supplied the runtime four-file bundle on 2026-11-08. Its
original bytes are tracked without modification. The directory named
`latest/` below preserves the previous AGX1 export for explicit reproduction.

The `kvolt.lut-selection.v1` selector records each sibling directory name and
its manifest SHA-256. KVolt verifies these with the existing pinned loaders.
Missing or invalid selections fail before tokenizer or hardware setup;
there is no directory scan, mtime-based choice or Mock fallback. Original
versioned bundles remain unchanged. Source metadata records the selector path,
SHA-256 and selection alongside the actual runtime and target provenance.
Selecting a bundle establishes no new accuracy, SLO, quality or formal
qualification, and does not allow an old grant to authorize changed sources.

## Current AGX2 runtime bundle

Each data file is an array of positional numeric rows. All 459 cells remain
unchanged, with the same axes, units, interpolation and range checks described
below for the previous export.

| File | Rows | Column order | Bytes |
| --- | ---: | --- | ---: |
| [`prefill.json`](agx2-runtime-repeat5/prefill.json) | 64 | `[length, gpu_mhz, ttft_s, prefill_energy_j]` | 2818 |
| [`decode.json`](agx2-runtime-repeat5/decode.json) | 320 | `[length, group, gpu_mhz, tpot_actual_s, epsilon_actual_j_per_token]` | 16702 |
| [`quality.json`](agx2-runtime-repeat5/quality.json) | 75 | `[length, group, mean_score]` | 1764 |

Performance is from AGX2 (A2), with five repeats per group. Each Prefill cell
is the arithmetic mean of all 25 observations: G=8/16/32/64/128, five requests
each. This covers 1,600 raw requests and one document per length. Decode
preserves the source cell means over five repeats. Quality remains the AGX1
(A1) RULER raw score over 100 items per length. It does not become AGX2
quality evidence when the performance device matches AGX2.

The original [`manifest.json`](agx2-runtime-repeat5/manifest.json) records
the full aggregation and generation context, source paths, hashes, byte counts,
units and measurement boundary. Model revision is
`0e9e39f249a16976918f6564b8830bc894c89659`, FP16 weights, K2V2, residual128,
CPU2201600kHz and EMC3199000000Hz. Decode's t1..tN boundary includes remaining
controller work and initial K2V2 packing once. Original validation flags
`performance_runtime_accuracy_or_SLO_validated=false` and
`quality_runtime_eligible=false` remain intact in provenance.

The prediction bundle has no FP16 targets. The latest target source is
[`agx2-reference-repeat10/`](agx2-reference-repeat10/). The user supplied its
310-byte `reference.json` on 2026-11-08; those bytes are preserved. Its eight
rows are `[length, r_p_ttft_s, r_d_tpot_s]`, lengths 1024..8192, step 1024.
Targets are independent maxima of TTFT and request mean TPOT over ten
requests per length (80 total): the baseline 24, first supplement 16 and
second supplement 40 requests. Conditions remain FP16 KV, default GPU DVFS
nvhost_podgov 306..1122MHz, CPU2201.6MHz and EMC3199MHz. Only N>1 enters TPOT.
Prefill/Decode policy multipliers remain 1.0. Interpolate only length; never
extrapolate.

The new `manifest.json` is a locally generated integration envelope, not a
user-provided original manifest. It records the exact original table hash,
source paths and hashes, context, aggregation and verification scope. The
tracked `source-validation.json` records the 80 canonical result hashes,
t0/t1/tN timestamps, N, repeat, timing metrics, per-request control checks and
the three source runs' declared metadata. The 80 canonical result bytes and
hashes, source-origin/protocol hash bindings, timestamps and request controls
were checked read-only in the AGX2 merged export. The original component
directories are unavailable; their manifest/completion hashes are inherited
declarations and were not newly rechecked
(`original_component_manifests_rechecked=false`). The summary does not contain
original request JSON, prompts or output token IDs. The loader
requires its exact hash and byte count and includes it alongside
`reference.json` in target `files_sha256` binding. This source check preserves
`controller_integration_validated=false` and establishes no new controller,
accuracy, quality, SLO or formal qualification. No new GPU measurements are
part of this registration.

The older [`agx2-reference-repeat5/`](agx2-reference-repeat5/) keeps all three
files unchanged for explicit reproduction of the five-repeat, 40-request
targets and their canonical validation evidence. The older
[`agx2-reference-repeat3/`](agx2-reference-repeat3/) keeps both original files
unchanged for the three-repeat, 24-request targets. Each version has its own
trusted manifest pin and exact measurement context; changing a repeat count
cannot authorize a new export.

`CompactLUTRepository` accepts only the two explicitly trusted prediction
manifests. `--reference-lut-dir` accepts either pinned three/five/ten-repeat
`CompactReferenceRepository` or the unchanged sealed AGX2
`MeasuredLUTRepository` for historical reproduction. Default use:

```sh
kvolt run ...
kvolt diagnose --length 1024 --max-new-tokens 1 --pc-elapsed-ns 0
```

`--lut-dir data/luts/current` and `--lut-bundle data/luts/current` explicitly
select the same entry point. A selector's reference cannot be overridden;
select a versioned compact directory for a different reference instead.
Explicit compact predictions still require `--reference-lut-dir`; a reference
alone or with `--mock` is rejected. Use `--mock baseline` for explicit synthetic
development. Versioned overrides and the audit tool remain available:

```sh
kvolt run --lut-dir data/luts/agx2-runtime-repeat5 --reference-lut-dir data/luts/agx2-reference-repeat10 ...
kvolt diagnose --lut-bundle data/luts/agx2-runtime-repeat5 --reference-lut-dir data/luts/agx2-reference-repeat10 ...
python3 tools/latest_tpot_audit.py --latest-dir data/luts/agx2-runtime-repeat5 --reference-lut-dir data/luts/agx2-reference-repeat10 --output audit.json
```

Provenance retains complete prediction and reference manifests, including the
locally generated ten-repeat envelope, and separate target conditions. Device
mismatch is calculated from their actual source devices. Matching devices establish no accuracy, SLO, quality eligibility or
formal qualification. An existing grant cannot authorize the new source/pins.
The sealed loader's original manifest pin and target contract remain unchanged.

Reference hashes:

| File | SHA-256 |
| --- | --- |
| `agx2-reference-repeat10/manifest.json` | `d332875ccce626dd99700da8c7a128bf4801577333e76ee40a987f4051311e40` |
| `agx2-reference-repeat10/reference.json` | `359fd02c609738755c87847f27b19286f45a3a7cb8d6bef8acd7c369e1c30c2b` |
| `agx2-reference-repeat10/source-validation.json` | `f954289b28ac94e3945d7ffeb0efa26e851fdfdc8025988181a94bdeabe8e8ea` |
| `agx2-reference-repeat5/manifest.json` | `d61c1b6a6ccd0aaa80b0cf4645702eaee09bd406dac084df70f151669a9b74ee` |
| `agx2-reference-repeat5/reference.json` | `0892b309e9124553f0b5a1410241060648b68f5be7c9c02db7e1d3b0fd640151` |
| `agx2-reference-repeat5/source-validation.json` | `a51bcff64f8e291d4121b0578c23ef211bad5c3e772f396ca31098d55400aa93` |
| `agx2-reference-repeat3/manifest.json` | `00d5112eaf9b0cdad2420b7622522c5d0913196e47681a286c99e19eeeeae15c` |
| `agx2-reference-repeat3/reference.json` | `0a1382b6a6a3b64626f4195902c3bdee325b58ff993714a4d4eb9d0693263b17` |

Prediction hashes:

| File | SHA-256 |
| --- | --- |
| `manifest.json` | `b421fcd10ecb57df7d6962aa9a5f48e264e90defc0c76a3544218d31685f783b` |
| `prefill.json` | `bb9e800faf56b3d0d5e4d20081dd62c4638642908fbf121199d83c40ae445a47` |
| `decode.json` | `875c78e7af420b52719b3a29d4c43332a9e53581be1681fa1f5b79e8cda3bb4b` |
| `quality.json` | `7804c10ea01c386cc52fe9367568951715d47b1d017aa790d299837db569235c` |

## Previous AGX1 compact export (explicit reproduction)

Use [`latest/`](latest/) only when explicitly reproducing the previous AGX1 results.
These are the three compact JSON files supplied by the user on 2026-11-07,
moved from the repository root without changing their bytes. Use the older
AGX2 sealed bundle only when explicitly reproducing historical results.

### AGX1 files and row contract

Each file is a JSON array of numeric row arrays. Column order is positional:

| File | Rows | Column order | Bytes |
| --- | ---: | --- | ---: |
| [`prefill.json`](latest/prefill.json) | 64 | `[length, gpu_mhz, ttft_s, prefill_energy_j]` | 3003 |
| [`decode.json`](latest/decode.json) | 320 | `[length, group, gpu_mhz, tpot_actual_s, epsilon_actual_j_per_token]` | 16657 |
| [`quality.json`](latest/quality.json) | 75 | `[length, group, mean_score]` | 1764 |

`length` is the request's input prompt length in tokens; `gpu_mhz` is MHz.
`group` is the quantization group size, not a token count for every axis.
Prefill time and energy are seconds and joules. Decode time and energy are
seconds per Decode token and joules per Decode token. `mean_score` is the raw
RULER score in [0,1], not FP16 retention `q`.

- Prefill/Decode lengths: 1024 through 8192, step 1024 (8 points).
- Quality lengths: 1024 through 8192, step 512 (15 points).
- GPU frequencies: 408, 510, 612, 714, 816, 918, 1020, 1122 MHz.
- Groups: 8, 16, 32, 64, 128.

Require the complete Cartesian grid with no duplicate keys, strict integer
identifiers, finite numeric values, positive times, nonnegative energies and
quality in [0,1]. Exact measured lengths use the stored values. Interpolate
only length, linearly between adjacent points for the same setting; use each
table's own length grid. Do not extrapolate, interpolate G/f, round the source
values or replace missing values with zero.

### AGX1 source and limits

The [Offline Profiler: LUT](https://app.notion.com/p/411e0acd77f88094b9bcdf5272ea776f)
documents this compact export as of 2026-11-07: AGX1 (A1/agx30), CPython
3.10.12/aarch64, model `meta-llama/Llama-3.1-8B-Instruct`. Performance cells
preserve the original three-request means; quality uses the raw mean RULER
score over 100 questions per length. The provided file sizes and first rows
match that document. Policy semantics follow
[Method](https://app.notion.com/p/40be0acd77f88014a36beb5285250208) under
[[연구] KVolt](https://app.notion.com/p/40be0acd77f880a98f05fd393b9e0ccd).

The original [`manifest.json`](latest/manifest.json) was subsequently recovered
from the compact export and preserved byte for byte (4,809 B; SHA-256
`86b2343eed65432d1ee1f071fa8188a33789bdcf0309f325e689ec0bae37591a`).
It seals these three arrays and records original source paths/hashes,
measurement conditions, units and interpolation rules. It does not contain
FP16 target curves or a source commit. Its original
`performance_runtime_accuracy_or_SLO_validated=false` and
`quality_runtime_eligible=false` remain unchanged.

`CompactLUTRepository` now loads these predictions with an explicitly supplied
sealed `MeasuredLUTRepository` for FP16 targets. The current AGX2 runtime and
its frozen FP16 goals remain unchanged. CLI use requires
`--lut-dir latest-path --reference-lut-dir sealed-agx2-path` for `run`, or
`--lut-bundle latest-path --reference-lut-dir sealed-agx2-path` for `diagnose`.
This is an explicitly recorded AGX1-prediction/AGX2-reference development
comparison. It does not turn AGX2 targets into AGX1 targets, establish Runtime
accuracy/SLO, or inherit old AGX2 qualification. No same-source complete AGX1
FP16 curve is supplied or extrapolated. See
[`docs/policy-lut.md`](../../docs/policy-lut.md) for loader and provenance details.

### AGX1 received-file SHA-256

| File | SHA-256 |
| --- | --- |
| `prefill.json` | `26380e92c28fd1f570fe4e83f8829d24a80558cb3e7bb5a7c65792c97f81add5` |
| `decode.json` | `4deacacb51444f941d1b6263dbc667f4dc42cad4afa80c99d8a8745fe8566208` |
| `quality.json` | `7804c10ea01c386cc52fe9367568951715d47b1d017aa790d299837db569235c` |
