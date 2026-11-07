# Scoped M4 qualification evidence

`load_evidence(report_path, binding)` validates a report and returns an immutable
`QualificationGrant`. The caller supplies the current trusted
`qualification_binding(...)` result. That producer checks clean identified
runtime source, measured AGX2 LUTs, native profile, physical identity, boot and
collected-power prerequisites. The loader compares the entire binding as
canonical JSON, including source, kernel/model/tokenizer bytes, LUT manifest,
device and collector profiles. A changed condition requires new linked evidence.

The grant means all required recorded correctness checks passed for that exact
binding and explicit coverage. SHA-256 links establish provenance and integrity;
they are not signatures or proof of honesty by an untrusted report author.
Retain and review the audit tooling and original outputs. The loader never runs
the recorded commands. It does not alter original LUT `runtime_eligible=false`
or `runtime_validation_required=true` flags. SLO violations and measured energy
remain evaluation observations, not technical check failures or approval gates.

## Report format

All objects below have exactly the documented keys. JSON must be strict UTF-8,
finite, and free of duplicate keys at every depth. Report and summary JSON files
are limited to 16 MiB each. Placeholder values in the example are not evidence.

```json
{
  "schema_version": "kvolt-runtime-qualification-v1",
  "binding": {"COPY": "the complete current qualification_binding result"},
  "coverage": {"1024": 1, "4096": 128, "8192": 512},
  "checks": {
    "cpu_policy": {"status": "passed", "artifact": "checks/cpu_policy.json"},
    "data": {"status": "passed", "artifact": "checks/data.json"},
    "backend_exact": {"status": "passed", "artifact": "checks/backend_exact.json"},
    "timing_power": {"status": "passed", "artifact": "checks/timing_power.json"},
    "device_control": {"status": "passed", "artifact": "checks/device_control.json"},
    "failure_recovery": {"status": "passed", "artifact": "checks/failure_recovery.json"},
    "reproducibility": {"status": "passed", "artifact": "checks/reproducibility.json"},
    "warmup": {"status": "passed", "artifact": "checks/warmup.json"}
  },
  "artifacts": {
    "checks/cpu_policy.json": "SHA256_OF_EXACT_FILE_BYTES",
    "checks/data.json": "SHA256_OF_EXACT_FILE_BYTES",
    "checks/backend_exact.json": "SHA256_OF_EXACT_FILE_BYTES",
    "checks/timing_power.json": "SHA256_OF_EXACT_FILE_BYTES",
    "checks/device_control.json": "SHA256_OF_EXACT_FILE_BYTES",
    "checks/failure_recovery.json": "SHA256_OF_EXACT_FILE_BYTES",
    "checks/reproducibility.json": "SHA256_OF_EXACT_FILE_BYTES",
    "checks/warmup.json": "SHA256_OF_EXACT_FILE_BYTES",
    "raw/audit.json": "SHA256_OF_EXACT_FILE_BYTES",
    "raw/audit.stdout": "SHA256_OF_EXACT_FILE_BYTES"
  }
}
```

`checks` must contain every name in `REQUIRED_CHECKS`, once each. Every receipt
must record `status="passed"` and its own distinct hashed summary file.
`coverage` must be nonempty. Keys are canonical decimal strings for integer input
lengths 1024 through 8192; values are integer maximum output caps 1 through 512.
Booleans, floats and numeric strings are invalid caps. Coverage does not imply
interpolation to unlisted lengths. A request needs its exact length listed and
an output cap at most the listed maximum.

Every SHA-256 value is 64 lowercase hexadecimal characters. Paths are canonical
POSIX relative paths within the report directory: no absolute paths, empty path
components, `.`, `..`, backslashes or NUL characters. All report ancestors and
all artifact path components must be free of symlinks. Files must be nonempty
regular files; FIFOs, directories and missing files fail validation. On macOS,
use `/private/tmp/...` rather than the `/tmp` symlink when staging a report.

## Per-check machine summary

Each summary file has exactly this structure:

```json
{
  "schema_version": "kvolt-qualification-check-v1",
  "check": "cpu_policy",
  "binding_sha256": "SHA256_OF_CANONICAL_BINDING_JSON",
  "coverage": {"1024": 1, "4096": 128, "8192": 512},
  "command": ["python3", "-m", "pytest", "tests/test_policy_prefill.py"],
  "exit_code": 0,
  "tests": {"run": 106, "passed": 3, "failed": 0, "skipped": 103},
  "artifacts": ["raw/audit.json", "raw/audit.stdout"]
}
```

`binding_sha256` is `value_sha256(binding)`, using UTF-8 canonical JSON:
`ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False`.
Summary coverage must exactly equal report coverage. `command` is a nonempty
array of nonempty strings recording the actual invocation; it is provenance.
`exit_code` must be integer zero. Counts must be nonnegative integers with
`run = passed + failed + skipped`, `passed > 0`, and `failed = 0`.

Only `cpu_policy`, `data` and `failure_recovery` permit nonzero `skipped` counts.
Their original counts must be retained; for example, a CPU suite can explicitly
skip GPU tests that separate backend/device checks execute. Backend, timing and
power, device control, reproducibility and warmup require zero skipped checks.
These are executed test/case counts, not fabricated qualification booleans.

Each summary must link at least one original nonempty machine result or command
output file. Retain the detailed inputs/reference/new outputs/diffs/environment
needed to assess its checks. Different summaries may share raw output files.
Summary files cannot serve as raw outputs, and a summary cannot repeat a raw
path. Every summary and raw output must appear in report `artifacts`; every
entry in `artifacts` must be referenced. The report cannot hash itself.

An audit that failed, lacks hardware access, or did not execute a required
check must preserve that fact in its audit outputs and must not construct a
passing qualification report. Existing M1/M2/M3 results do not become M4 checks
merely by adding filenames or `passed` booleans.
