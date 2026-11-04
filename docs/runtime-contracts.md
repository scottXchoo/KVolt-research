# M1 runtime contracts

This is the CPU contract boundary for the C0 → B1 → B2 milestone. Importing
`kvolt_runtime`, `config`, `contracts`, `errors`, or `backend.protocol` does not
import Torch, CUDA, Transformers, or install GPU dependencies.

## Fixed configuration and requests

`RuntimeConfig` fixes Llama-3.1-8B-Instruct snapshot
`0e9e39f249a16976918f6564b8830bc894c89659`, FP16 weights, batch size 1,
greedy generation, K2/V2, residual length 128, and G candidates
`(8, 16, 32, 64, 128)`. EOS IDs are `(128001, 128008, 128009)`.
Values and their types, including tuple member types, must match the defaults.

`resolve_nmax` resolves item value, then CLI value, then 512. Every supplied
value must be an integer from 1 through 512; explicit null, booleans, and floats
are invalid. Nmax includes the first output token and any generated EOS.

`PromptInput.read` hashes original file bytes with SHA-256 and strictly decodes
UTF-8 without changing whitespace or CRLF. `RequestSpec.from_prompt` tokenizes
once with `add_special_tokens=False`, preserving existing special tokens and
adding no automatic BOS, template, or truncation. Input IDs are a nonempty
tuple of integers in `[0, 128255]`. Their hash is SHA-256 of compact UTF-8 JSON
such as `[1,2,3]`. `RequestSpec.length` is the number of actual model input IDs.
LUT support validation belongs to the later preflight/LUT layer.

## Cache ownership and transitions

The core treats payloads as opaque backend-owned objects. Live metadata fixes
32 layers, batch size 1, 8 KV heads, and head dimension 128. The backend must
validate every tensor's shape, dtype, device, post-RoPE meaning, and full
completion before publishing a replacement. CPU metadata validation alone does
not establish those GPU properties.

A handle starts with raw FP16 metadata and a non-null payload. Initial packing
commits one complete packed payload with the same request ID, device, and T.
Each later decode consumes the previous output at position T and commits T+1
with the same G. An invalid transition preserves both old metadata and old
payload. Metadata and payload are published together in one reference update.
The handle is request-scoped and not thread-safe.

The caller and backend must drop aliases to the old full cache after a successful
handoff. Independent packed tails must not retain old full storage through
views. `release` drops this handle's active reference and is idempotent; other
operations on released handles fail. A new `TokenResult` requires a live handle.
After N generated outputs, the cache length is `L + N - 1`.

## Backend and timing

`Backend` defines `load_session`, `prefill_raw`, `pack_initial`, `decode_one`,
and `release_request`. One loaded model instance persists across requests; KV
state is new for each request. Prefill returns y1 with raw post-RoPE FP16 KV at
T=L. If y1 is EOS or Nmax=1, the caller skips DC, initial packing, and decode.
Otherwise, after the caller applies and verifies the selected GPU frequency,
the backend packs all 32 layers and commits only after GPU completion.

`Clock` is a callable returning integer nanoseconds from one monotonic host
clock, for example `time.perf_counter_ns`. Backend `TokenResult.ready_ns` is
recorded after GPU completion using the same clock as future runner events.
EOS has its own output ID and ready timestamp. `generation_stop` gives EOS
priority when EOS also reaches Nmax, rejects output beyond Nmax, and rejects
tokens generated after an earlier EOS. `StopReason.INTERRUPTED` is reserved
for caller-reported user interruption; it is not inferred from output IDs.

Canonical units are integer ns for event timestamps, seconds for elapsed
TTFT, seconds/token for mean TPOT, integer MHz for policy frequencies, integer
Hz for device receipts, J for measured energy, and J/token for normalized
Decode energy. Conversion helpers validate integer inputs. For N=1, Decode
TPOT and J/token must be null rather than zero. Full generation, measurement,
and cleanup result DTOs and energy calculations remain a later C0 expansion
for the runner/recording work; M1 does not implement or claim those results.

## Errors and verification limits

`ContractError` rejects invalid inputs or metadata. `CacheStateError` rejects
invalid transitions or released use. `BackendError` is a technical execution
failure, preserving the original cause; the caller must release the request.
`QualificationError` identifies a required exact comparison or provenance
failure. User interruption, generation failure, measurement failure, and cleanup
failure must remain separate in the later runner's result schema.

CPU tests establish the input, metadata, lifecycle, unit, and import boundary.
They do not establish GPU completion, tensor correctness, single resident model
behavior, or KIVI output equivalence. Those require B1/B2 device qualification.
