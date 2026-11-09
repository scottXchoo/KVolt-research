# Latest LUT feasibility and TPOT remedies

The tables and GPU observations below are historical zero-buffer results.
Current `latest_tpot_audit.py` applies the global `config.TPOT_BUFFER_RATIO`
(default `0.20`) to the unchanged raw FP16 Reference before feasibility checks,
and records raw Reference, ratio and effective target separately. Re-running
the commands therefore uses the current allowance. It does not reclassify the
archived measured runs or establish faster execution under the changed policy.

## Scope and reproducibility

This audit uses all **320 latest AGX1 Decode predictions**, the latest 64 Prefill
predictions and 75 raw quality scores. Targets remain the explicit sealed
**AGX2 FP16 reference** from `agx2-repeat5-20261103`; no AGX1 reference curve is
substituted. This device mismatch is deliberate for a development crossover
check. It does not validate Runtime prediction accuracy, actual SLO compliance,
quality eligibility or formal AGX2 qualification.

The compact original manifest, all three raw table SHA-256 values, the sealed
reference manifest and every sealed reference file hash are written into the
strict JSON report. Source bytes and targets remain unchanged. See
[`data/luts/README.md`](../data/luts/README.md) for the raw table contract.

```sh
python3 tools/latest_tpot_audit.py \
  --latest-dir data/luts/latest \
  --reference-lut-dir ../KVolt-support/KVolt-lut-source/data/luts/agx2-repeat5-20261103 \
  --output artifacts/latest-lut/feasibility.json
```

The default run executes the existing `PrefillController` and `DecodeController`
with the pinned `KneeAdapter` (`kneed==0.9.0`, `scipy==1.15.3`). It evaluates all
eight lengths, assumed phase elapsed times of **0 and 2,000,000 ns**, and
`Nmax=2,128,512`: 48 scenarios. These fixed clock values are assumptions, not
measured controller overhead. `--elapsed-ns` and `--nmax` can select other
explicit scenarios. First token ID 1 is a non-EOS policy input; no model runs.

On a CPU environment without the pins, the default command writes the global
arithmetic, marks policy execution `unavailable`, writes no policy scenarios,
and exits 2. Use `--arithmetic-only` to request only that arithmetic explicitly.
There is no synthetic Knee substitution. A full run in the existing pinned
AGX2 environment is required to review the actual policy selections.

## Global time feasibility before Knee restriction

A candidate is feasible when its unrounded binary64 TPOT is `<= R_d(L)`.
The following display rounds milliseconds only after classification. Minimum
and maximum range over all 40 `(G,f)` pairs. Margin is target minus minimum.

| L | AGX2 target ms | AGX1 min ms | AGX1 max ms | Min G / MHz | Feasible / 40 | Min margin ms |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1024 | 103.211482 | 117.562830 | 234.609487 | 128 / 1020 | 0 | -14.351349 |
| 2048 | 107.987008 | 117.611971 | 243.937634 | 128 / 1122 | 0 | -9.624963 |
| 3072 | 112.686189 | 117.567787 | 253.305022 | 128 / 1020 | 0 | -4.881598 |
| 4096 | 117.438934 | 118.036820 | 262.001997 | 128 / 1020 | 0 | -0.597886 |
| 5120 | 122.180925 | 118.046843 | 271.635151 | 128 / 1122 | 7 | 4.134082 |
| 6144 | 127.008534 | 120.098807 | 281.328684 | 128 / 1122 | 10 | 6.909727 |
| 7168 | 131.692505 | 122.588349 | 290.440144 | 128 / 1122 | 11 | 9.104156 |
| 8192 | 136.512458 | 125.091561 | 300.134204 | 128 / 1122 | 12 | 11.420897 |

Thus 1K–4K have no globally time-feasible latest prediction even with zero
elapsed controller time. Changing selection among the same predictions cannot
make those lengths feasible against these targets. This is a prediction-space
statement across two devices, not proof of an actual AGX2 SLO violation.

The JSON separately records the global 40-candidate count, time-feasible count
under each scenario budget, Knee-active count, Knee-active time-feasible count,
selection path, selected predicted feasibility, exact target/budget, selected
G/f, TPOT, raw energy, raw quality and budget/target margins. An empty
Knee-restricted feasible set triggers latency fallback over all 40 candidates.
That fallback can still select a globally feasible prediction; its path alone
does not prove infeasibility or an actual SLO miss.

## Output length and initial packing

The current Method uses `B_d = R_d - elapsed_s / (Nmax - 1)`. At an assumed
2 ms DC elapsed time, the deductions for `Nmax=2,128,512` are respectively
2 ms, about 0.015748 ms and about 0.003914 ms per Decode token. This budget
formula accounts for that assumed control time. It does not predict the
request's actual EOS position or re-amortize the LUT's startup cost.

The recovered original `artifacts/latest-lut/sources/original-decode.json`
records 960 source request observations. Of these, 912 produced 512 tokens.
The 1K observations include N=453,493,512; all 2K–8K observations have N=512.
The source therefore largely amortizes initial packing over D=511 Decode
forwards, with the stated 1K exceptions. The source campaign permits natural
EOS and a maximum of 512; it does not establish that every Runtime request
will produce 512 tokens.

The latest TPOT already includes its source callback and initial packing once.
Do not add the recorded source packing time again. A shorter new request can
have a larger startup contribution per token because its measured denominator
is actual `D=N-1`. An illustrative decomposition `TPOT = H/D + steady_mean`
explains this sensitivity, but is not a calibrated Runtime predictor. The
current LUT has no output-length axis and does not identify a request-specific
steady mean or new controller cost.

These boundaries follow the latest
[Offline Profiler: LUT](https://app.notion.com/p/411e0acd77f88094b9bcdf5272ea776f)
and [Method](https://app.notion.com/p/40be0acd77f88014a36beb5285250208).
The current Method deliberately skips DC and packing for first-token EOS or
`Nmax=1`, and forbids reselection during a request. The audit does not alter
those rules or force output length.

## Remedies and decision boundaries

| Candidate | Code evidence and next validation | Status |
| --- | --- | --- |
| Keep decision serialization outside the measured generation interval | `execution/runner.py` already defers decision materialization; commit `d709924` implements this. Use stage traces to check residual CPU work. | Already implemented; no new change here |
| Reduce initial packing launches with a batched 32-layer prototype | `backend/cache.py:pack_cache` loops over 32 layers and invokes unchanged K/V quantizers per layer. Measure launch, transpose/copy and kernel costs before choosing batching. Preserve native packed bytes, tails, validation and atomic commit. | Investigation only; no speedup claimed |
| Reduce repeated metadata validation and per-token allocation overhead | `backend/cache.py:validate_packed_cache` inspects all layers; `backend/decode.py:decode_one` validates before/after forward, allocates token/position tensors and reads `.item()`. Profile these spans before a prototype. Preserve fatal-error checks and cache contracts. | Investigation only; no checks removed |
| Capture graph-safe GPU sections | CUDA Graphs can reduce CPU launch overhead, but current KIVI concatenations change KV shapes/storage, and Decode reads `.item()` and synchronizes. Stable buffers/shapes plus capture boundaries need a backend redesign and correctness qualification. | Investigation only; no flag-only remedy |
| Use FP16 Decode when no K2V2 candidate can meet a supported target | The current DC has only K2V2 `(G,f)` candidates. FP16 requires a new candidate/path, latency/energy/quality evidence and transition semantics; unchanged FP16 reference targets alone are insufficient. | Method change requiring a decision; not implemented |
| Model startup separately and account for output length | A new model could distinguish initial packing/control from steady Decode and evaluate an assumed/predicted D. It requires source stage measurements, held-out error checks and a defined EOS uncertainty policy. | Method/LUT change requiring a decision; not implemented |

KIVI's reported gains depend on workload and batching; its memory and aggregate
throughput results do not establish faster single-request TPOT on this Orin
Runtime. See the original [KIVI paper](https://arxiv.org/abs/2402.02750).
PyTorch describes CUDA Graphs as reducing submission overhead and explains the
amortization of capture work in its
[official CUDA Graphs article](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).
The versioned [PyTorch 2.5.1 CUDA semantics](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst)
requires stable replay addresses and graph-safe shapes/control flow; CPU/GPU
synchronization such as `.item()` is prohibited inside capture. These are
constraints on a future prototype, not evidence that this backend is capturable
or that capture will close the measured TPOT gap.

No backend, online selection rule, target curve, quality score or measurement
boundary is changed by this audit. CPU feasibility and fixed-clock policy
execution must remain separate from the subsequent measured development run.
