"""CPU-only checks for development profiling wrapper integrity."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock

from tools.profile_tpot import Trace, source_commit, validate_inputs


class TraceTests(unittest.TestCase):
    def test_wrap_preserves_arguments_identity_and_completed_timing(self):
        trace = Trace(clock=iter((11, 37)).__next__)
        trace.request_id = "request-a"
        sentinel = object()
        function = mock.Mock(return_value=sentinel)
        result = trace.wrap("pack_initial", function)("cache", group=16)
        self.assertIs(result, sentinel)
        function.assert_called_once_with("cache", group=16)
        self.assertEqual(trace.rows, [{"component": "pack_initial", "request_id": "request-a", "start_ns": 11, "end_ns": 37, "status": "completed"}])

    def test_wrap_preserves_original_exception_and_failed_end_time(self):
        trace = Trace(clock=iter((11, 37)).__next__)
        error = RuntimeError("injected backend failure")
        function = mock.Mock(side_effect=error)
        with self.assertRaises(RuntimeError) as raised:
            trace.wrap("decode_one", function)("cache", 17)
        self.assertIs(raised.exception, error)
        self.assertEqual(trace.rows[0]["status"], "failed")
        self.assertEqual(trace.rows[0]["end_ns"], 37)

    def test_validators_patch_both_call_sites_and_restore_exact_references(self):
        original_cache = mock.Mock(return_value="cache-valid")
        original_decode = mock.Mock(return_value="decode-valid")
        modules = [SimpleNamespace(validate_packed_cache=original_cache), SimpleNamespace(validate_packed_cache=original_decode)]
        trace = Trace(clock=iter((1, 2, 3, 4)).__next__)
        with mock.patch("tools.profile_tpot.importlib.import_module", side_effect=modules) as importer:
            with trace.validators():
                self.assertIsNot(modules[0].validate_packed_cache, original_cache)
                self.assertIsNot(modules[1].validate_packed_cache, original_decode)
                self.assertEqual(modules[0].validate_packed_cache("c"), "cache-valid")
                self.assertEqual(modules[1].validate_packed_cache("d"), "decode-valid")
        self.assertEqual(importer.call_args_list, [mock.call("kvolt_runtime.backend.cache"), mock.call("kvolt_runtime.backend.decode")])
        self.assertIs(modules[0].validate_packed_cache, original_cache)
        self.assertIs(modules[1].validate_packed_cache, original_decode)
        self.assertEqual([r["status"] for r in trace.rows], ["completed", "completed"])

    def test_validator_failure_restores_both_modules_and_records_failure(self):
        error = ValueError("invalid packed metadata")
        original_cache = mock.Mock(side_effect=error)
        original_decode = mock.Mock()
        modules = [SimpleNamespace(validate_packed_cache=original_cache), SimpleNamespace(validate_packed_cache=original_decode)]
        trace = Trace(clock=iter((1, 2)).__next__)
        with mock.patch("tools.profile_tpot.importlib.import_module", side_effect=modules):
            with self.assertRaises(ValueError) as raised:
                with trace.validators():
                    modules[0].validate_packed_cache("bad")
        self.assertIs(raised.exception, error)
        self.assertIs(modules[0].validate_packed_cache, original_cache)
        self.assertIs(modules[1].validate_packed_cache, original_decode)
        self.assertEqual(trace.rows[0]["status"], "failed")
        self.assertEqual(trace.rows[0]["end_ns"], 2)

    def test_body_failure_restores_both_validators(self):
        originals = [object(), object()]
        modules = [SimpleNamespace(validate_packed_cache=original) for original in originals]
        with mock.patch("tools.profile_tpot.importlib.import_module", side_effect=modules):
            with self.assertRaisesRegex(RuntimeError, "body failure"):
                with Trace().validators():
                    raise RuntimeError("body failure")
        for module, original in zip(modules, originals):
            self.assertIs(module.validate_packed_cache, original)

    def test_backend_wrappers_keep_request_context_and_results(self):
        request = SimpleNamespace(request_id="fixture")
        sentinel = object()
        backend = SimpleNamespace(prefill_raw=mock.Mock(return_value=sentinel), pack_initial=mock.Mock(return_value=sentinel), decode_one=mock.Mock(return_value=sentinel))
        trace = Trace(clock=iter(range(6)).__next__)
        factory = mock.Mock(return_value=backend)
        wrapped = trace.backend(factory)("args", "clock")
        factory.assert_called_once_with("args", "clock")
        self.assertIs(wrapped.prefill_raw(request), sentinel)
        self.assertIs(wrapped.pack_initial("cache", 8), sentinel)
        self.assertIs(wrapped.decode_one("cache", 17), sentinel)
        self.assertEqual([r["component"] for r in trace.rows], ["prefill_raw", "pack_initial", "decode_one"])
        self.assertEqual([r["request_id"] for r in trace.rows], ["fixture"] * 3)


class SourceCommitTests(unittest.TestCase):
    def test_clean_source_returns_exact_commit(self):
        with mock.patch("tools.profile_tpot.subprocess.check_output", side_effect=["", "abc123\n"]) as run:
            self.assertEqual(source_commit(Path("/tmp/runtime")), "abc123")
        self.assertEqual(run.call_args_list[0], mock.call(["git", "-C", "/tmp/runtime", "status", "--porcelain", "--", "src", "tests", "tools", "pyproject.toml"], text=True))
        self.assertEqual(run.call_args_list[1], mock.call(["git", "-C", "/tmp/runtime", "rev-parse", "HEAD"], text=True))

    def test_dirty_source_fails_before_reading_commit(self):
        with mock.patch("tools.profile_tpot.subprocess.check_output", return_value=" M tools/profile_tpot.py\n") as run:
            with self.assertRaisesRegex(ValueError, "commit reviewed source"):
                source_commit(Path("/tmp/runtime"))
        self.assertEqual(run.call_count, 1)


class FrozenInputTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'src'), *sys.path]))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.plan = {'schema_version': 'kvolt-m4-plan-v1', 'formal_evaluation_run': False,
                     'root': str(self.root), 'frozen_files': {}}
        for field, relative in {'device_profile': 'device-profile.json', 'power_profile': 'power-profile.json',
                                'manifest': 'cohort/manifest.json', 'eos_manifest': 'eos-cohort/manifest.json'}.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}\n')
            self.plan[field] = str(path)
            self.plan['frozen_files'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.path = self.root / 'plan.json'
        self.seal()

    def seal(self):
        self.path.write_text(json.dumps(self.plan))
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_valid_reviewed_frozen_inputs_are_accepted(self):
        self.assertEqual(validate_inputs(self.path, self.digest), self.plan)

    def test_plan_tamper_fails_reviewed_digest(self):
        self.path.write_text(self.path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            validate_inputs(self.path, self.digest)

    def test_manifest_tamper_fails_even_with_unchanged_plan(self):
        Path(self.plan['manifest']).write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'frozen input changed'):
            validate_inputs(self.path, self.digest)

    def test_resealed_plan_cannot_redirect_frozen_manifest(self):
        self.plan['manifest'] = str(self.root / 'eos-cohort/manifest.json')
        self.seal()
        with self.assertRaisesRegex(ValueError, 'frozen input path mismatch'):
            validate_inputs(self.path, self.digest)


if __name__ == "__main__":
    unittest.main()
