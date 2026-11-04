# B1 single-model raw Prefill

`InferenceBackend.load_session()` constructs one pinned `LlamaForCausalLM_KIVI`
with FP16 weights. Its attention instances change only local method dispatch;
parameter and buffer objects remain intact. The unchanged project model source is
vendored byte-for-byte under its MIT license. Quantization/native kernels still
load from the separately preserved AGX1 baseline and pass source, binary and
installed-version checks. Model configuration, tokenizer and all four weight
shards are hashed before model loading. No dependency installation is performed.

`prefill_raw()` uses the original projection, RoPE, conditional FP16 cast,
FlashAttention and output projection order. It returns the greedy first token and
32 post-RoPE FP16 K/V pairs shaped `[1,8,L,128]`. Cache length remains `L`; the
first token has not been consumed. No G is chosen and no quantization runs.
The token-ready timestamp follows device synchronization. Release the cache
before the next request; model loading is not repeated. Packing and packed-cache
Decode are the B2 integration boundary.

CPU verification:

```sh
python3 -m pytest -q tests/test_backend_prefill.py
PYTHONPATH=src python3 -m pytest -q tests/test_backend_prefill.py
```

AGX1 opt-in qualification, from the exact committed checkout:

```sh
KVOLT_RUN_B1_GPU=1 \
LD_LIBRARY_PATH=/home/agx3/kvolt/.local/cusparselt/lib:/usr/local/cuda/lib64 \
PYTHONPATH=src:/home/agx3/kvolt-runtime-test-deps \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/agx3/kvolt/.venv/bin/python -m pytest -q -s \
tests/integration/test_backend_prefill_gpu.py
```

The test acquires `/run/lock/kvolt-p3-hardware.lock` itself, without waiting.
Do not add an outer `flock`. GPU imports and model loading happen after locking.
The command-scoped `LD_LIBRARY_PATH` uses the existing cuSPARSELt and CUDA
libraries; it does not modify the baseline environment or install dependencies.
The frozen `artifacts/m1/cohort/inputs.json` must already exist with SHA-256
`e9f0c05e0d21e13356f6a64d1176b03bcff613c708b0b0fd72228f711aadc20d`.
The three first representative rows at L1024/4096/8192 run, then L1024 repeats.
The test checks raw layout, absent quantization, single candidate load,
parameter/buffer identity, position reset, released KV reachability and exact
first-token equality to the unchanged G32 native model. Candidate and reference
weight instances run sequentially. Native full-generation/all-G packing checks
belong to B2.

The default report is `artifacts/m1/b1/prefill-report.json`. It records git SHA,
source/binary/weight hashes, import paths, effective model settings, actual Python
executable, command-scoped `LD_LIBRARY_PATH` and per-input tokens. Only these two
process environment fields are captured. `KVOLT_B1_REPORT`, `KVOLT_M1_COHORT`
and `KVOLT_MODEL_SNAPSHOT` override paths; input and snapshot identities remain
pinned. A skipped GPU test is
not a qualification pass. These checks do not measure SLO, time or energy gains.
