# M1 B2 correctness qualification

B2 defers the unchanged patched KIVI model's initial K2V2 quantization until
B1 returns y1 and all 32 post-RoPE FP16 KV pairs. The same resident candidate
model then consumes y1 at position L. For N outputs there are N−1 Decode
forwards and final native cache length L+N−1. EOS IDs are 128001/128008/128009;
Nmax includes y1 and ranges from 1 through 512. First EOS or Nmax=1 skips
candidate packing and Decode. No EOS suppression or forced logits are used.

`qualification/reference.py` calls the unchanged vendored native KIVI model
with raw_prefill=False. Its only control is fixed G before Prefill, direct
forwards, greedy EOS/cap, and records. It does not use the old deferred K2
backend or candidate prefill/packing/decode to produce reference tokens.
`set_group` sets G on the model/config and every attention/config before native
Prefill; the shared loader intentionally starts with G=None.

Every raw layer is checked before the first packing kernel. Every packed field,
all GPU work, and all attention settings must be ready before the handle is
published. Tails own cloned contiguous storage. Packing preserves T. Decode
validates token/timestamp and the full replacement before publishing T+1.
An error stops qualification. A retained old handle does not guarantee that
retrying is safe after a native forward; automatic retries are unsupported.

## CPU checks

```sh
python3 -m pytest -q
PYTHONPATH=src python3 tools/qualify_backend.py --help
python3 -m compileall -q src tools tests
```

The CPU suite uses injected cache/model fixtures, including first EOS and
Nmax=1. A skipped GPU test is not GPU qualification. Natural first-EOS coverage
requires the separately frozen follow-up below; the current frozen RULER cohort
does not establish that branch on the device.

## AGX1 commands

Only the coordinating main agent runs these after checking the agreed source
revision and hardware availability. Do not wrap the harness in an outer flock;
the harness itself takes the nonblocking exclusive lock before importing GPU
libraries or loading a model. A busy lock exits without GPU initialization.
The pytest cache fixture does the same and releases its lock at module end.
No frequency or device-control settings are written by these commands.

```sh
cd /home/agx3/kvolt-runtime
export PYTHONPATH=/home/agx3/kvolt-runtime/src:/home/agx3/kvolt-runtime-test-deps
export LD_LIBRARY_PATH=/home/agx3/kvolt/.local/cusparselt/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export KVOLT_SNAPSHOT=/home/agx3/kvolt/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
KVOLT_RUN_GPU=1 /home/agx3/kvolt/.venv/bin/python -m pytest -q tests/integration/test_cache_equivalence_gpu.py
/home/agx3/kvolt/.venv/bin/python tools/qualify_backend.py --suite smoke --snapshot "$KVOLT_SNAPSHOT" --output artifacts/m1/b2-smoke
/home/agx3/kvolt/.venv/bin/python tools/qualify_backend.py --suite full --snapshot "$KVOLT_SNAPSHOT" --output artifacts/m1/b2-full
/home/agx3/kvolt/.venv/bin/python tools/qualify_backend.py --suite boundaries --snapshot "$KVOLT_SNAPSHOT" --output artifacts/m1/b2-boundaries
```

The artificial cache matrix contains 9 T values × 5 G = 45 cases, each with all 32
native fields. Independent quantizer-axis/bit/constant diagnostics add 5 cases.
A one-attention native residual-forward diagnostic adds 45 cases across the same
9 T × 5 G grid, comparing exact output/cache and T+1, K flush, and V eviction. It
constructs one unchanged attention module with seeded FP16 random weights,
original FP32 rotary buffers, and artificial KV; it is separate from full model natural-EOS generation.
The actual CUDA matmul/GQA diagnostic adds 5 cases × 7 one-hot input positions,
checking every output column and every KV/query head. Its integer and
power-of-two inputs produce exact values, so it uses zero tolerance, including
the distinct G8 metadata for both halves of a packed word.

Smoke uses one frozen input at L=1024/4096/8192 × 5 G, Nmax=4: 15 pairs.
Full uses the frozen 30 inputs × 5 G, Nmax=128: 150 pairs/300 paired generations.
Boundaries use the 3 lengths × 5 G × caps 1/2/512: 45 pairs/90 paired generations.
Each suite also repeats one frozen representative native request per distinct
L/G/cap for baseline repeatability: 15 repeats for smoke, 15 for full, 45 for
boundaries. Total planned generations are 45, 315, 135 respectively.
These diagnostic repeats are extra GPU work and are not independent quality
samples. Caps of 512 have natural EOS enabled, so
actual output count and elapsed GPU work depend on the model. No runtime or
energy estimate is presented as measured performance.

## Explicit AGX2 native build

`load_model` and `InferenceBackend` accept the keyword `native_profile='agx1'`.
Both qualification tools accept `--native-profile {agx1,agx2}` and default to
`agx1`. The selected profile is passed to both native reference and candidate
and recorded in their provenance and the qualification plan. Production code
does not select a profile from environment variables or device/path detection.
An injected backend loader retains its existing `(path, config)` signature;
the caller supplying that loader owns its loading arguments.

These are an immutable whitelist of previously inspected builds. AGX1's
existing binary pins remain unchanged. AGX2 changes only these native hashes:

| Native module | AGX2 SHA256 |
| --- | --- |
| `kivi_gemv` | `59a8cc13a82859612494d3fde0554edd2eee4cdcb7b9cdbb4f1a98297fb9e27a` |
| `flash_attn_2_cuda` | `10e5657f58c462cbc052b1798792e94aab498b47d039bad5ddc305be0a414c11` |

Source, model/tokenizer, and dependency pins are identical for both profiles.
Provenance `baseline_commit` identifies the original pinned source baseline;
it does not assert that AGX2's legacy checkout has that Git HEAD. Actual source
bytes, native builds, model files, import origins, and Runtime revision are
recorded and checked separately.
Unknown profiles and wrong native hashes fail before torch import/model load.
There is no hash override, automatic acceptance, or check-skipping flag.
Imported module origins and binary bytes are rechecked against the selected
profile. Selecting AGX2 is not evidence that its GPU qualification has passed.

Only GPU test entry points read `KVOLT_NATIVE_PROFILE` (default `agx1`),
`KVOLT_BASELINE`, and `KVOLT_KIVI_ROOT` (default `$KVOLT_BASELINE/third_party/KIVI`).
The cache tests still require the exact pinned baseline helper sources. On
AGX2 the coordinator supplies a frozen Python-only reference source snapshot
because its legacy checkout lacks `scripts/k2_decode_lut/backend.py`.
`KVOLT_SNAPSHOT` supplies the model snapshot; B1 also preserves its existing
`KVOLT_MODEL_SNAPSHOT` override. Tools require explicit CLI arguments.

```sh
cd /home/agx31/kvolt-runtime
export PYTHONPATH=/home/agx31/kvolt-runtime/src:/home/agx31/kvolt-runtime-test-deps
export LD_LIBRARY_PATH=/mnt/kvolt-data/kvolt/.local/cusparselt/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export KVOLT_NATIVE_PROFILE=agx2
export KVOLT_BASELINE=/home/agx31/kvolt-m1-reference
export KVOLT_KIVI_ROOT=/mnt/kvolt-data/kvolt/third_party/KIVI
export KVOLT_SNAPSHOT=/mnt/kvolt-data/kvolt/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
/mnt/kvolt-data/kvolt/.venv/bin/python -m pytest -q
KVOLT_RUN_GPU=1 /mnt/kvolt-data/kvolt/.venv/bin/python -m pytest -q tests/integration/test_cache_equivalence_gpu.py
KVOLT_RUN_B1_GPU=1 /mnt/kvolt-data/kvolt/.venv/bin/python -m pytest -q tests/integration/test_backend_prefill_gpu.py
/mnt/kvolt-data/kvolt/.venv/bin/python tools/qualify_backend.py --suite smoke \
  --native-profile agx2 --kivi-root "$KVOLT_KIVI_ROOT" \
  --snapshot "$KVOLT_SNAPSHOT" --output artifacts/m1/agx2-b2-smoke
/mnt/kvolt-data/kvolt/.venv/bin/python tools/qualify_first_eos.py --prepare-only \
  --native-profile agx2 --snapshot "$KVOLT_SNAPSHOT" --output artifacts/m1/agx2-first-eos-frozen
# Retain preparation's exact SHA256 before inference:
/mnt/kvolt-data/kvolt/.venv/bin/python tools/qualify_first_eos.py \
  --native-profile agx2 --kivi-root "$KVOLT_KIVI_ROOT" --snapshot "$KVOLT_SNAPSHOT" \
  --inputs artifacts/m1/agx2-first-eos-frozen/inputs.json \
  --expected-input-sha256 <CPU_PREPARATION_SHA256> --output artifacts/m1/agx2-first-eos-run
```

First-EOS preparation freezes `native_profile` in the policy and provenance;
the plan includes both. Run must select the same profile, and a mismatch fails
before reserving an attempt or loading a model. The shared `POLICY` constant is
not changed during selection. Prepare a new explicit AGX2 freeze after the
final clean commit, retaining the same eight approved prompts. Existing AGX1
frozen evidence remains separate and is not rewritten to select AGX2.

The common hardware lock reuses its existing inode via a read-only descriptor
and takes the same exclusive nonblocking flock. If absent, creation is
exclusive; a concurrent creator's inode is reopened. No lock file is removed,
replaced, chmodded, or bypassed. Do not wrap any command in another flock.
AGX1's busy root-owned lock remains untouched.

Record source/helper archive hashes, device and selected native build with
each result. AGX1's existing 210 exact pairs remain AGX1 evidence. AGX2's
target CPU, cache/GQA, B1, smoke, and eight-input first-EOS results are separate;
do not claim a full 150-pair or 45-boundary AGX2 rerun unless actually executed.
AGX2 results do not qualify AGX1 natural first EOS or the full adaptive Runtime.

## Evidence and storage

The input/manifest SHA256s are frozen in `qualification/compare.py`. All 30
inputs, lengths, IDs, prompt hashes, and token hashes are checked before model
load. The plan records exact cases, source file SHA256s, Git commit/worktree,
config, model/tokenizer snapshot, and batch membership before GPU imports.
The loader additionally pins source/build/dependency/model hashes. Device and
both native/candidate provenance are recorded per batch.

Native and candidate models are sequential. To fit available disk, cases are
split into bounded batches using exact logical cache bytes. The default budget
is 3072 MiB; `--cache-budget-mib` accepts 1..4096. At least budget + 1024 MiB free disk
is required before loading. Each native phase unloads before its candidate
phase; one candidate model serves all requests in that batch. This intentionally
reloads between batches and does not measure session performance.

Native initial caches are copied with
`detach().contiguous().cpu().clone()` before serialization to avoid retaining
full raw storage through views. Candidate comparison checks every tensor byte,
not only digests. A mismatch preserves its native tensor file, candidate tensor
file, first layer/field/byte/element and expected/actual values. Token comparison
preserves every original token ID, stop reason, forward count, final T, and
first mismatch index. A baseline repeat mismatch stops before the candidate
phase. Passed pairs remove their large tensor files and retain per-field
shape/dtype/byte/SHA256 proof. `progress.json` records each completed pair;
`summary.json` is written only when the entire suite passes. Output directories
must be empty; automatic resume/retry is unsupported. Unexpected errors print
a traceback and leave existing evidence in place.

The unchanged helper in `scripts/k2_decode_lut/backend.py` is used only for
artificial cache comparison. Its SHA256 is
`2702735c01b64bd84f146a8b042b96960aa7540ada52b98208caf7421295c859`;
its cache helper `scripts/p6/cache.py` is
`f0de50eae45cf204f63dd29837ba5b7dd33f2b21b0dc9544473e4c8c5258f24f`.
The native greedy reference does not call either deferred helper.

This is M1 backend correctness qualification. It makes no PC/DC, LUT, DVFS,
energy, performance, or full adaptive Runtime qualification claim.

## Bounded natural first-token EOS follow-up

`tools/qualify_first_eos.py` is a separate synthetic diagnostic authorized in
RT-054. It does not modify the frozen 30-input cohort or extend Runtime LUT
support beyond 1024..8192. Eight exact completed-dialogue prompts are constants
in the tool: number-only addition, OK, oxygen symbol, Spanish cat, hello,
opposite of hot, empty JSON, and Korean addition. Each contains its existing
BOS, one user EOT, and an assistant answer without a trailing EOT. There is no
EOS forcing, logit/model change, EOS suppression, padding, or input search.

Preparation uses the pinned snapshot's Rust `tokenizers` encoding with
`add_special_tokens=False`; it imports neither torch nor transformers and loads
no model. It verifies both pinned tokenizer files, records exact prompt/token
IDs and hashes, Git commit/status, repository root, source hashes, tokenizer
revision/version, and all cases in `inputs.json` and `plan.json`. Prepare after
the final clean commit, on the same checkout where the GPU run will execute.
No source, Git state, tokenizer, inputs, or plan edits may occur after freeze.
Both output directories must be empty. The run needs the independently retained
input SHA256 printed by preparation. A one-use `run-attempt.json` reservation
prevents another invocation of that frozen plan, including after a technical
failure or a busy hardware lock; the harness never creates a replacement plan.

```sh
/home/agx3/kvolt/.venv/bin/python tools/qualify_first_eos.py --prepare-only \
  --snapshot "$KVOLT_SNAPSHOT" --output artifacts/m1/first-eos-frozen
# Retain the printed inputs_sha256 before any inference, then use that exact value:
/home/agx3/kvolt/.venv/bin/python tools/qualify_first_eos.py \
  --snapshot "$KVOLT_SNAPSHOT" --inputs artifacts/m1/first-eos-frozen/inputs.json \
  --expected-input-sha256 <CPU_PREPARATION_SHA256> --output artifacts/m1/first-eos-run
```

The fixed plan has 8 prompts × 5 requested G values × Nmax=2 = 40 pairs,
plus one native repeat per prompt at G32: 88 total generations. These are eight
distinct diagnostic inputs, not 40 independent quality examples. Nmax=2 keeps
natural first EOS distinguishable from the Nmax=1 shortcut. Native reference
uses the unchanged loader with `raw_prefill=False` and `generate_reference`;
candidate uses `generate_candidate`. Reference and candidate models run
sequentially, with the reference unloaded before loading the candidate. The
nonblocking hardware lock is acquired before torch/model imports. Stable
fields in `device_report` avoid NVIDIA device-properties repr UTF-8 failures.
Loader provenance and all raw output token IDs are retained. No settings or
frequencies are changed, and there are no performance, energy, or SLO claims.

A delegating spy observes only candidate public request API calls. A natural
first-EOS pair must match exact token IDs, EOS stop, zero Decode forwards, and
T=L; it must have prefill=1, packing=0, decode=0, release=1, RAW cache at release,
RELEASED afterward, and a cleared active cache. First-EOS G is a requested
comparison condition: quantization does not run in that branch. This M1 tool
does not establish integrated DC behavior. Non-first-EOS inputs complete the
normal branch up to Nmax=2 and retain exact IDs/stop/forward/T comparisons;
initial packed-cache tensor proof is outside this follow-up's scope and remains
covered by the existing B2 suites.

All 40 pairs and eight native repeats must pass. Per-G coverage and covered
case IDs are recorded. `first_eos_gpu_qualified` is true only when natural first
EOS is actually observed and passes all checks in every requested G condition.
No observation yields `bounded_search_complete_no_first_eos`, exit 0, and
qualification false; there is no expanded search or retry. Partial G coverage
also leaves qualification false. Technical failures preserve `error.json` with
traceback, any completed outputs and spy evidence, and exit nonzero. Source,
input, plan, Git and tokenizer identities are rechecked before model loading
and at completion. CPU spy fixtures prove the harness checks only; they do not
qualify natural EOS on a GPU.
