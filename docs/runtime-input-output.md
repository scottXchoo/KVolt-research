# M3 input and result evidence

This CPU-only boundary implements the v1 Runtime design sections 6–8. It does
not change M1 tensor paths or M2 selection policy, and does not qualify a run
for formal performance evaluation. Runner owns generation, measurement and
cleanup status construction.

## Input preflight

`execution.inputs.preflight_inputs` takes keyword-only `prompt_file`, `manifest`,
`tokenizer`, `repo`, and optional `cli_nmax=MISSING`. Exactly one input path is
required. The caller supplies its pinned CPU tokenizer and validated LUT
repository before importing/loading GPU code.

The manifest must have exactly `schema_version` and `requests` at its top level.
The schema is `kvolt-request-manifest-v1`; requests is a nonempty array. Each
request has `request_id` and `prompt_file`, with optional `nmax` and `metadata`.
Unknown request fields, duplicate JSON keys at any depth, nonfinite JSON
numbers, invalid UTF-8, blank/duplicate IDs and metadata outside an object are
rejected. Metadata keys are an extension area used only for recording.

Relative prompt paths resolve from the manifest directory. Every supplied
Nmax is a strict integer from 1 through 512, including a CLI value overridden
by an item. Priority is item, CLI, then 512. The direct prompt ID is
`request-000000`.

Prompt files are read once as bytes and strictly decoded as UTF-8. CRLF,
whitespace and existing special-token text remain intact. The tokenizer is
called with `add_special_tokens=False`, without a template, stripping or
truncation. Actual token IDs must have length 1024 through 8192 inclusive.
Each request calls `repo.preflight(length)`, which validates complete support
without online interpolation, quality lookup or Knee detection. Any failure
rejects the complete session before GPU work.

`PreparedInputs` is frozen. It contains a tuple of shared `PreparedRequest`
objects, original manifest path/bytes/SHA-256 when supplied, and an immutable
`resolved_manifest_json` string. The `resolved_manifest` and per-request
`metadata` properties return fresh native dictionaries. Prepared prompt text,
IDs, hashes, L and Nmax are used throughout execution; input files are never
reread to run or save a request.

Prompt SHA-256 hashes original bytes. Ordered token IDs are hashed using
compact JSON UTF-8 with `ensure_ascii=False`, `sort_keys=True`,
`separators=(',', ':')`, and `allow_nan=False`, matching the existing contract.
The original manifest hash likewise covers original bytes, including spacing
and line endings. Resolved evidence makes absolute paths and effective Nmax
explicit while preserving request order and metadata.

## Session storage

Current Runner output uses `kvolt-session-v2` and `kvolt-request-result-v2`.
Both preserve `tpot_policy = {version: "reference-buffer-v1", ratio: 0.20}`;
runtime provenance records the same policy. Active Decode decisions include
`reference_target_s` (raw FP16), `tpot_buffer_ratio`, and `target_s` (allowed
TPOT after one multiplication). N=1/EOS skip target fields remain null.
`slo.mean_tpot_met` compares actual mean TPOT inclusively with this effective
target. The auditor requires matching current policy fields and rejects a
missing or inconsistent identity. Historical v1 evidence retains the recorded
zero-buffer interpretation; it is not reclassified by the current global value.

`recording.SessionRecorder(output_dir)` performs exclusive directory creation.
An existing directory, regular file, symlink or dangling symlink raises a
conflict. The parent must already exist. A rejected output location receives
no error artifact. Runner/CLI reports the failure using stderr/JSONL and null
paths as appropriate.

The recorder owns this layout; request IDs never become path components:

```text
session.json
manifest.original.json          # exact bytes, only for manifest input
manifest.resolved.json
preflight.json
inputs/PROMPT_SHA256.txt
requests/000000/request.json
requests/000000/generation.json  # only when supplied
requests/000000/result.json
requests/000000/output.txt      # only when output_text is supplied
```

`save_preflight(prepared)` saves frozen inputs and original/resolved evidence.
`preflight.json` records manifest identity and every request's original token
IDs, token/prompt hashes, L, Nmax, metadata and saved input path. Input files
with identical byte hashes share one evidence path.

`save_request(index, prepared, result, generation=None, output_text=None)`
uses a zero-based strict integer index formatted to at least six digits.
It writes input identity in `request.json` and preserves caller-supplied
generation and result fields without inferring status or validity. It returns
`request_path`, `generation_path`, `result_path` and `output_path`; optional
files have null paths. An empty output string is a real supplied output and
creates an empty file. `not_run` needs no fabricated generation record, token,
time, TTFT, or output. Valid timing/energy and partial generation prefixes
remain available even when a separate cleanup status is failed.

`save_session(summary)` writes caller-assembled `session.json`. `session_path`
and `path(relative_path)` return a relative path only for an existing regular
artifact, otherwise null. Caller-owned session counts and stdout records use
the same independently constructed generation/timing/energy/cleanup data.
The recorder never retries a request.

Request results additionally contain `memory` (`kvolt-memory-v1`), preserved
unchanged in both `requests/NNNNNN/result.json` and `session.json.results`.
Its `before`/`after` snapshots contain system RAM/swap, process RSS/lifetime HWM,
and CUDA allocator counters; after-snapshot CUDA peaks are reset per request.
Read windows, units, scope and unavailable/partial reasons accompany values.
These diagnostics are independent of timing/energy validity, and unattempted
requests have null snapshots. Stdout request summaries expose `memory_status`.
See [request memory diagnostics](runtime-measurement.md#request-memory-diagnostics)
for metric semantics and the distinction between snapshots and request peaks.

`atomic_json`, `write_text` and `write_bytes` use same-directory temporary files,
file fsync, atomic replacement and parent-directory fsync. Traversal and
symlink artifact destinations are rejected. Directory file descriptors avoid
following replaced directory symlinks; root inode checks reject reuse after
session-directory replacement. Before returning success, the public artifact
path is reopened and its parent/file inode identity is checked, rejecting a
directory swap during publication rather than returning an absent file link.
Storage errors propagate, and unpublished
temporary files are removed. A directory-fsync error after replacement can
leave a visible file whose crash durability is unknown; it remains a storage
failure. Process termination, power loss and storage/stdout errors cannot
guarantee final artifacts or JSONL lines.

`to_json_native`, `json_dumps`, and `json_line` support native finite values,
dataclasses, tuples, enums and paths without GPU or scientific dependencies.
JSON object keys must be strings. `json_line` emits one line with embedded
newlines escaped. Runner/CLI owns stdout record schemas and writes.

## Validation

`tests/test_inputs.py` covers schema/duplicate/nonfinite validation, Nmax
precedence and strict types, original UTF-8/CRLF, immutable evidence, actual
length boundaries, full-request LUT preflight and CPU-only imports.
`tests/test_recording.py` covers output conflicts including dangling symlinks,
frozen input storage, numeric paths, independent statuses, partial prefixes,
`not_run`, atomic replacement/fsync failure, traversal/symlink rejection and
root ownership changes. M3 CPU tests do not claim GPU or M4 qualification.
