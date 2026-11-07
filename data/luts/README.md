# Latest LUT lookup data

Use [`latest/`](latest/) as the default source for future LUT value lookups.
These are the three compact JSON files supplied by the user on 2026-11-07,
moved from the repository root without changing their bytes. Use the older
AGX2 sealed bundle only when explicitly reproducing historical results.

## Files and row contract

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

## Source and limits

The [Offline Profiler: LUT](https://app.notion.com/p/411e0acd77f88094b9bcdf5272ea776f)
documents this compact export as of 2026-11-07: AGX1 (A1/agx30), CPython
3.10.12/aarch64, model `meta-llama/Llama-3.1-8B-Instruct`. Performance cells
preserve the original three-request means; quality uses the raw mean RULER
score over 100 questions per length. The provided file sizes and first rows
match that document. Policy semantics follow
[Method](https://app.notion.com/p/40be0acd77f88014a36beb5285250208) under
[[연구] KVolt](https://app.notion.com/p/40be0acd77f880a98f05fd393b9e0ccd).

The export's original `manifest.json` and FP16 reference target curves were
not supplied in this request. This README records the received files; it does
not recreate or replace that manifest, establish missing provenance, or add
reference targets.

The existing `MeasuredLUTRepository` still accepts the older sealed AGX2
repeat5 bundle contract described in [`docs/policy-lut.md`](../../docs/policy-lut.md).
It does not load these compact arrays. Adopting them as the latest lookup data
does not switch the Runtime loader or establish Runtime accuracy, SLO or AGX2
qualification for this AGX1 data.

## Received-file SHA-256

| File | SHA-256 |
| --- | --- |
| `prefill.json` | `26380e92c28fd1f570fe4e83f8829d24a80558cb3e7bb5a7c65792c97f81add5` |
| `decode.json` | `4deacacb51444f941d1b6263dbc667f4dc42cad4afa80c99d8a8745fe8566208` |
| `quality.json` | `7804c10ea01c386cc52fe9367568951715d47b1d017aa790d299837db569235c` |
