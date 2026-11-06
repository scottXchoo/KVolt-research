# Standard M3 run CLI

`kvolt run` executes a frozen sequence of development requests through one
resident model. `kvolt diagnose` remains a CPU policy diagnostic using explicit
assumed phase times. Both help paths import neither GPU nor scientific packages.

```sh
kvolt run --prompt-file PROMPT.txt \
  --model-dir /models/0e9e39f249a16976918f6564b8830bc894c89659 \
  --lut-dir SEALED_LUT_BUNDLE --output-dir NEW_SESSION \
  --device-profile DEVICE_PROFILE.json --native-profile agx2 \
  --kivi-root /path/to/KIVI --max-new-tokens 128 \
  --collect-power --power-profile POWER_PROFILE.json
```

Use exactly one of `--prompt-file` and `--manifest`, and exactly one of
`--lut-dir` and `--mock SCENARIO`. Synthetic scenarios are `baseline`,
`flat_quality`, `nonmonotonic`, and `exact_ties`; they require explicit selection
and are recorded as synthetic. Development is the default mode. Performance
always rejects with `M4_QUALIFICATION_REQUIRED` before device or GPU work until
formal M4 evidence support exists. Development device/source mismatches remain
explicit in provenance and grant no performance eligibility.

The standard path fixes FP16, greedy decoding, batch size one, K2V2, residual
length 128, the five Method group candidates, and pinned EOS IDs. It exposes no
sampling, dtype, group, frequency, or hardware lock path override. Native build
profile `agx1` is the default; select `agx2` explicitly for its pinned binaries.

All manifest entries and every supplied Nmax are checked before lock acquisition.
Per-item Nmax overrides CLI Nmax, then defaults to 512; every supplied value must
still be an integer from 1 through 512. Prompt bytes remain strict UTF-8 with
whitespace and CRLF preserved. Actual token length must be 1024 through 8192.
CPU preflight validates the fixed snapshot name and pinned tokenizer/config/
generation-config hashes, then uses `tokenizers==0.19.1` and
`Tokenizer.from_file(tokenizer.json).encode(add_special_tokens=False)` with
padding/truncation disabled. It never imports `transformers.AutoTokenizer`.
The loaded model tokenizer must reproduce every frozen token sequence before
warmup or timed execution.

The output directory must be new and its parent must exist. Existing files,
directories, symlinks, and dangling symlinks reject without writing into them.
Inputs, resolved manifest and preflight evidence are saved before GPU work.
Knee dependencies are initialized on CPU before the hardware lock and outside
timed phases. An all-Nmax=1 session can skip those unused scientific dependencies.

The existing common lock `/run/lock/kvolt-p3-hardware.lock` is acquired
nonblockingly before Backend construction/loading, warmup, or GPU setting changes.
Busy/missing/replaced locks reject without waiting or selecting another path.
The explicit device profile checks separately prepared CPU/EMC/MAXN/fan settings;
standard run only changes/restores GPU governor/min/max. Power collection requires
an explicit power profile and starts before t0, then acquires a right bracket
after the final token. Exact worker JSONL and derived CSV traces are saved under
`telemetry/INDEX/`, with hashes and collection failures preserved.

`--warmup basic` is the default development warmup: actual frozen inputs, each
distinct length/active-Decode shape, all five groups when Decode is active,
one Prefill/pack/Decode per case, request-local cache release, then GPU restoration.
Natural first EOS skips pack/Decode. `--warmup none` explicitly disables this
work. Neither option constitutes complete JIT/residual-shape M4 qualification.

Stdout contains request-result JSONL followed by exactly one terminal
session-result record when output remains writable. Python and native loader
noise go to stderr. The terminal record is published after an owned hardware
lock is released. SIGINT/SIGTERM handlers only record the first signal; Runner
stops after the current forward, saves the prefix, restores settings and marks
later requests `not_run`. Original handlers are restored afterward. There is no
automatic retry or recovery guarantee for SIGKILL, power loss, or storage/stdout
failure.

Exit codes are 0 for completed required execution/measurement/cleanup, 1 for
technical loading/execution/measurement/storage/restoration failures, 2 for
argument/input/LUT/output-location rejection, 3 for hardware lock/preparation
failure, 130 for SIGINT and 143 for SIGTERM. Normal timing/energy measurements
remain valid when only later restoration fails. Failed collection never exposes
normal energy values; available diagnostics remain under `prefix`. Preflight
rejections also emit `session_result`, with unavailable artifact paths set to null.

`RunFactories` and `run(args, *, lock=None, factories=None, emit=None)` form an
internal CPU-test/qualification seam. A supplied lock must already be held and
remains caller-owned. These are Python interfaces, not user-selectable CLI bypasses.
