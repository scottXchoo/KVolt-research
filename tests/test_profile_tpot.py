"""CPU-only checks for development profiling wrapper integrity."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
import hashlib
import json
import subprocess
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


class RuntimeIsolationTests(unittest.TestCase):
    """Exercise main in fresh processes against two independently importable trees."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.baseline, self.driver = self.root / 'baseline', self.root / 'driver'
        for root, commit in ((self.baseline, 'baseline-sha'), (self.driver, 'driver-sha')):
            files = {
                '__init__.py': f'COMMIT = {commit!r}\n',
                'execution/__init__.py': '', 'execution/runner.py': 'MARKER = "runner"\n',
                'backend/__init__.py': '', 'backend/model.py': 'MARKER = "model"\n',
                'backend/cache.py': 'def validate_packed_cache(*args): pass\n',
                'backend/decode.py': 'def validate_packed_cache(*args): pass\n',
                'qualification/__init__.py': '',
                'qualification/contracts.py': 'def file_sha256(path): return "plan-sha"\n',
                'cli.py': ('import argparse\n'
                           'def _parser():\n'
                           '    parser = argparse.ArgumentParser()\n'
                           '    parser.add_argument("--output-dir")\n'
                           '    parser.parse_args = lambda args: parser.parse_known_args(args)[0]\n'
                           '    return parser\n'),
                'device.py': ('from contextlib import nullcontext\n'
                              'class DeviceProfile:\n'
                              '    from_json = staticmethod(lambda path: None)\n'
                              'HardwareLock = nullcontext\n'),
                'run_cli.py': ('from dataclasses import dataclass\n'
                               'from pathlib import Path\n'
                               'import json, os\n'
                               'from . import COMMIT\n'
                               '@dataclass\n'
                               'class RunFactories:\n'
                               '    backend: object = None\n'
                               'def run(args, **kwargs):\n'
                               '    root = Path(args.output_dir)\n'
                               '    root.mkdir()\n'
                               '    runtime = {"source_commit": os.environ.get("TEST_SOURCE", COMMIT),\n'
                               '               "source_worktree_dirty": os.environ.get("TEST_DIRTY") == "1"}\n'
                               '    (root/"session.json").write_text(json.dumps({"provenance": {"runtime": runtime}}))\n'
                               '    return 0\n'),
            }
            for relative, content in files.items():
                path = root / 'src/kvolt_runtime' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
        helper = self.driver / 'tools/qualify_m4.py'
        helper.parent.mkdir(parents=True)
        helper.write_text(
            'from pathlib import Path\nimport sys, json\nfrom contextlib import contextmanager\n'
            'sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))\n'
            'import kvolt_runtime\n'
            '@contextmanager\n'
            'def hardware_scope(*args):\n'
            '    try: yield\n'
            '    finally: (Path(__file__).parent/"restored").write_text("restored")\n'
            'progress = lambda message: None\n'
            'def read_json(path): return json.loads(Path(path).read_text())\n'
            'def write(path, value):\n'
            '    path = Path(path)\n'
            '    path.parent.mkdir(parents=True, exist_ok=True)\n'
            '    path.write_text(json.dumps(value))\n')

    def invoke(self, *, preimport=False, mixed=False, source=None, dirty=False):
        import os
        profile = Path(__file__).resolve().parents[1] / 'tools/profile_tpot.py'
        output = self.root / 'output'
        script = '''
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("profile_under_test", sys.argv[1])
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
baseline, driver, output = map(Path, sys.argv[2:5])
if sys.argv[5] == "preimport":
    sys.path.insert(0, str(driver/"src"))
    import kvolt_runtime
elif sys.argv[5] == "mixed":
    sys.path.insert(0, str(baseline/"src"))
    import kvolt_runtime
    wrong = importlib.util.spec_from_file_location("kvolt_runtime.backend.model", driver/"src/kvolt_runtime/backend/model.py")
    module = importlib.util.module_from_spec(wrong)
    wrong.loader.exec_module(module)
    sys.modules[wrong.name] = module
profile.ROOT = driver
profile.os.geteuid = lambda: 0
profile.source_commit = lambda root: "baseline-sha" if root == baseline else "driver-sha"
profile.validate_inputs = lambda *args: dict.fromkeys(
    ("device_profile", "power_profile", "manifest", "model_dir", "lut_dir", "kivi_root"), "fixture") | {"source_commit": "plan-source"}
try:
    profile.main(["--plan", "fixture", "--plan-sha256", "plan-sha",
                  "--runtime-root", str(baseline), "--output-dir", str(output)])
except ValueError as error:
    print(str(error), file=sys.stderr)
    sys.exit(3)
'''
        environment = dict(os.environ)
        environment.pop('TEST_SOURCE', None)
        environment.pop('TEST_DIRTY', None)
        if source is not None:
            environment['TEST_SOURCE'] = source
        if dirty:
            environment['TEST_DIRTY'] = '1'
        result = subprocess.run([sys.executable, '-I', '-c', script, str(profile), str(self.baseline),
                                 str(self.driver), str(output), 'preimport' if preimport else ('mixed' if mixed else 'fresh')],
                                text=True, capture_output=True, env=environment)
        status_path = output / 'profile-status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else None
        return result, status

    def test_selected_runtime_survives_helper_prepend_in_fresh_process(self):
        result, status = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        session = json.loads((self.root / 'output/run-1/session.json').read_text())
        self.assertEqual(session['provenance']['runtime']['source_commit'], 'baseline-sha')
        self.assertEqual(status['status'], 'completed')
        self.assertTrue(status['sessions'][0]['source_verified'])
        for name in ('kvolt_runtime', 'kvolt_runtime.execution.runner',
                     'kvolt_runtime.backend.model', 'kvolt_runtime.cli'):
            self.assertIn(name, status['runtime_modules'])
        for path in status['runtime_modules'].values():
            self.assertTrue(Path(path).is_relative_to(self.baseline / 'src/kvolt_runtime'))

    def test_preimported_wrong_package_is_rejected_before_hardware(self):
        result, status = self.invoke(preimport=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn('runtime module outside selected checkout', result.stderr)
        self.assertIsNone(status)
        self.assertFalse((self.driver / 'tools/restored').exists())

    def test_preimported_wrong_submodule_is_rejected_with_correct_package(self):
        result, status = self.invoke(mixed=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn('runtime module outside selected checkout: kvolt_runtime.backend.model', result.stderr)
        self.assertIsNone(status)
        self.assertFalse((self.driver / 'tools/restored').exists())

    def test_success_exit_with_mismatched_session_source_is_failed(self):
        result, status = self.invoke(source='driver-sha')
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['sessions'][0]['exit_code'], 0)
        self.assertFalse(status['sessions'][0]['source_verified'])
        self.assertIn('session Runtime source mismatch', status['errors'][0]['message'])
        self.assertEqual((self.driver / 'tools/restored').read_text(), 'restored')

    def test_success_exit_with_dirty_session_source_is_failed(self):
        result, status = self.invoke(dirty=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(status['status'], 'failed')
        self.assertFalse(status['sessions'][0]['source_verified'])
        self.assertIn('session Runtime source mismatch', status['errors'][0]['message'])
        self.assertEqual((self.driver / 'tools/restored').read_text(), 'restored')

    def test_missing_or_unknown_session_source_is_rejected(self):
        from tools.profile_tpot import verify_session_source
        for session in ({}, {'provenance': {'runtime': {'source_commit': 'baseline-sha'}}},
                        {'provenance': {'runtime': {'source_commit': 'baseline-sha', 'source_worktree_dirty': None}}}):
            with self.subTest(session=session):
                with self.assertRaisesRegex(ValueError, 'session Runtime source mismatch'):
                    verify_session_source(session, 'baseline-sha')


class LutOverrideTests(unittest.TestCase):
    def test_default_retains_frozen_lut_without_reference_override(self):
        from tools.profile_tpot import lut_selection
        self.assertEqual(lut_selection({'lut_dir': '/frozen/lut'}), ('/frozen/lut', []))

    def test_compact_override_passes_explicit_reference_and_preserves_plan(self):
        from tools.profile_tpot import lut_selection
        plan = {'lut_dir': '/frozen/lut'}
        selected, arguments = lut_selection(plan, Path('data/luts/latest'), Path('/reference/lut'))
        self.assertEqual(selected, str(Path('data/luts/latest').resolve()))
        self.assertEqual(arguments, ['--reference-lut-dir', '/reference/lut'])
        self.assertEqual(plan, {'lut_dir': '/frozen/lut'})

    def test_reference_cannot_silently_change_frozen_lut(self):
        from tools.profile_tpot import lut_selection
        with self.assertRaisesRegex(ValueError, 'explicit --lut-dir'):
            lut_selection({'lut_dir': '/frozen/lut'}, reference_lut_dir=Path('/reference/lut'))

    def test_actual_prediction_and_target_metadata_must_match_and_remain_development(self):
        from tools.profile_tpot import verify_session_lut
        expected = {'source_device': 'agx1', 'files_sha256': {'decode.json': 'latest'},
                    'reference': {'source_device': 'agx2', 'sha256': 'target'}}
        session = {'mode': 'development', 'provenance': {
            'lut': expected, 'formal_performance_qualified': False}}
        verify_session_lut(session, expected)
        for changed in ({**expected, 'files_sha256': {'decode.json': 'old'}},
                        {**expected, 'reference': {'source_device': 'agx2', 'sha256': 'other'}}):
            with self.assertRaisesRegex(ValueError, 'LUT/reference'):
                verify_session_lut(session, changed)
        with self.assertRaisesRegex(ValueError, 'unqualified development'):
            verify_session_lut({**session, 'mode': 'performance'}, expected)
        with self.assertRaisesRegex(ValueError, 'unqualified development'):
            verify_session_lut({'mode': 'development', 'provenance': {
                'lut': expected, 'formal_performance_qualified': True}}, expected)


if __name__ == "__main__":
    unittest.main()
