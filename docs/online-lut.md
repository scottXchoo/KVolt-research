# Runtime Online LUT feedback

Development `kvolt run` enables an adaptive overlay by default. The selected,
verified offline bundle remains the immutable baseline. Completed requests feed
measured Prefill and Decode latency and, when valid collected power exists,
energy into the overlay. PC and DC use the same repository and the same revision
throughout one request. Feedback becomes visible only to the next request.

## Measurement and update policy

Prefill observes the complete `ttft_s = (t1-t0)/1e9` interval and its integrated
Prefill joules. Decode observes `mean_tpot_s = (tN-t1)/(N-1)/1e9` and integrated
Decode joules divided by `N-1`. These measurements already include controller,
frequency application/verification, and initial packing costs inside their
existing boundaries. Feedback never adds those costs a second time. Cleanup,
observation storage, and state updates occur after the measured intervals.

The update is `new = (1-alpha)*old + alpha*measured`, with `alpha=0.1` by default.
The first update starts from the offline prediction, including baseline length
interpolation. Only the selected Prefill `(actual L, f)` and Decode
`(actual L, G, f)` cells change. An observation at an intermediate length creates
an exact-length override; it does not modify neighboring offline knots or alter
predictions for another length. Unobserved lengths retain the baseline's length
interpolation. No frequency/group interpolation or length extrapolation is added.
Quality and frozen FP16 Reference targets remain unchanged.

A valid SLO miss remains an observation. Incomplete generation and invalid timing
do not update the predictor. A restoration-only failure after completed valid
measurements remains eligible. Missing or invalid energy does not overwrite
energy with a prediction or zero; independently valid latency can still update.
For `N=1`, Decode remains null and no Decode cell changes. State records sample
counts and actual output-count ranges. Decode averages still depend on actual
`N`, prompt,
and stopping behavior. This change does not add an N predictor or bucket, change
the existing Nmax-based controller budget, or guarantee improved SLO/energy.

## Files and state reuse

A new output directory receives:

- `online-lut.json`: adaptive state, written atomically after eligible feedback.
- `online-lut.initial.json`: full initial state, before hardware ownership.
- `online-lut.final.json`: full final state, including an aborted session.
- `requests/NNNNNN/online-observation.json`: durable original request result,
  saved before applying feedback.
- `requests/NNNNNN/result.json`: final request result with the used revision,
  observation identity, original evidence path, and update receipt.

Session provenance preserves the original baseline LUT metadata and records the
online policy, initial snapshot and state identity separately. Session metadata
also links the initial/final state exports. Each observation identity is a
session UUID plus request index; it contains no prompt text. JSONL request records
publish the feedback receipt only after saving the final result.

By default, each new output directory starts a fresh adaptive history. Reuse a
previous compatible state explicitly:

```sh
kvolt run --prompt-file prompt.txt --model-dir /models/FIXED_SNAPSHOT \
  --device-profile device.json --output-dir runs/second \
  --online-lut-state runs/first/online-lut.json
```

Use `--online-lut-alpha FLOAT` for a finite EMA weight in `(0,1]`. Reusing a state
requires matching alpha, baseline identity, model/tokenizer content, native build,
device/power profiles, collection mode, and runtime software/source identity.
Model/tokenizer artifact locations and session locations are excluded from the
environment context. Baseline provenance remains a strict identity, including
current-selection selector paths; moving that checkout/selector can therefore
reject reuse despite unchanged table bytes. Hardware profile sysfs paths and
measurement timing settings remain bound conditions. State updates use
an exclusive lock and compare-and-swap; a competing writer causes an explicit
failure rather than silently overwriting a newer history. The full deduplication
list of observation IDs grows with the number of eligible observations; long
running histories therefore increase state size.

Use `--no-online-lut` to reproduce frozen-source behavior. It conflicts with
explicit state/alpha options. Formal `--mode performance` remains frozen by
default, and rejects explicit online state/alpha before hardware access. Existing
M4 qualification does not qualify this adaptive policy. Direct SessionRunner
performance calls with an adaptive repository are also rejected before hardware.

An observation/state write failure preserves valid measurements, records a
technical error and aborts subsequent requests. The runtime never substitutes a
prediction or zero for a missing measurement. Durable original evidence permits
inspection even when feedback or later result storage fails. No new GPU,
accuracy, SLO or formal performance qualification is claimed by CPU tests.
