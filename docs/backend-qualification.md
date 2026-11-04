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
Nmax=1. A skipped GPU test is not GPU qualification. The GPU first-EOS natural
input policy is pending; the current frozen RULER cohort does not establish
that branch on the device.

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
