"""AGX3 GSM8K environment/energy validation; spawn imports perform no work."""
import argparse
import fcntl
import gc
import glob
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SERIAL_SHA = 'c9ecb6270ed595be35f29138221737aa9d120d86538219e2208bf5e68e6f8a61'
MODEL = Path('/mnt/kvolt-data/kvolt/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659')
KIVI = Path('/mnt/kvolt-data/kvolt/third_party/KIVI')
LOCK = Path('/run/lock/kvolt-p3-hardware.lock')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def source_check(root, expected):
    """Bind the selected clean Git checkout bytes to the caller's exact commit."""
    root = Path(root).resolve()
    if len(expected) != 40 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('expected commit must be a full lowercase 40-character SHA')

    def git(*arguments):
        return subprocess.run(['git', '-C', str(root), *arguments],
            check=True, capture_output=True).stdout

    if Path(git('rev-parse', '--show-toplevel').decode().strip()).resolve() != root:
        raise ValueError('source root must be the Git checkout root')
    if git('rev-parse', 'HEAD').decode().strip() != expected:
        raise ValueError('Git HEAD differs from expected commit')
    selected = ('src', 'pyproject.toml', 'tools/agx3_gsm8k')
    if git('status', '--porcelain=v1', '-z', '--untracked-files=all', '--', *selected):
        raise ValueError('selected source/tool files are dirty or untracked')
    ignored = [Path(name.decode('utf-8')) for name in
        git('ls-files', '--others', '--ignored', '--exclude-standard', '-z', '--', *selected).split(b'\0') if name]
    if any(not ('__pycache__' in name.parts and name.suffix in ('.pyc', '.pyo')) for name in ignored):
        raise ValueError('selected ignored source/tool files are untracked')
    names = []
    for entry in git('ls-tree', '-r', '-z', expected, '--', *selected).split(b'\0'):
        if not entry:
            continue
        metadata, name = entry.split(b'\t', 1)
        mode, kind, _ = metadata.split()
        if mode not in (b'100644', b'100755') or kind != b'blob':
            raise ValueError('selected source must contain regular tracked files')
        names.append(name.decode('utf-8'))
    if not names:
        raise ValueError('expected commit has no selected source files')
    files = {}
    for name in names:
        path = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root):
            raise ValueError('unsafe source path: ' + name)
        if any(candidate.is_symlink() for candidate in (path, *path.parents) if candidate != root and root in candidate.parents):
            raise ValueError('source symlinks are not supported: ' + name)
        committed_sha = hashlib.sha256(git('show', expected + ':' + name)).hexdigest()
        if not path.is_file() or digest(path) != committed_sha:
            raise ValueError('working bytes differ from committed source: ' + name)
        files[name] = committed_sha
    if git('rev-parse', 'HEAD').decode().strip() != expected:
        raise ValueError('Git HEAD changed during source validation')
    if git('status', '--porcelain=v1', '-z', '--untracked-files=all', '--', *selected):
        raise ValueError('selected source changed during validation')
    return dict(source_sha=expected, source_root=str(root), files=files)


def controls():
    values = {}
    for pattern in (
        '/sys/devices/system/cpu/cpufreq/policy*/scaling_governor',
        '/sys/devices/system/cpu/cpufreq/policy*/scaling_min_freq',
        '/sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq',
        '/sys/class/devfreq/*gpu*/governor', '/sys/class/devfreq/*gpu*/min_freq',
        '/sys/class/devfreq/*gpu*/max_freq',
        '/sys/kernel/debug/bpmp/debug/clk/emc/min_rate',
        '/sys/kernel/debug/bpmp/debug/clk/emc/max_rate',
        '/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked',
        '/sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt',
    ):
        for name in sorted(glob.glob(pattern)):
            try:
                values[name] = Path(name).read_text().strip()
            except OSError as error:
                values[name] = dict(error=type(error).__name__)
    try:
        service = subprocess.run(['systemctl', 'is-active', 'nvfancontrol'],
            capture_output=True, text=True, timeout=5, check=False)
        values['service.nvfancontrol'] = dict(returncode=service.returncode,
            state=service.stdout.strip(), stderr=service.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        values['service.nvfancontrol'] = dict(error=type(error).__name__)
    return values


def append_json(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def positive_limit(value):
    number = int(value)
    if number != 10:
        raise argparse.ArgumentTypeError('this validation protocol requires --limit 10')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='new artifact base; attempt1 is created beneath it')
    parser.add_argument('--expected-commit', required=True, help='exact full 40-character Git commit')
    parser.add_argument('--limit', type=positive_limit, default=10)
    parser.add_argument('--model-name', type=Path, default=MODEL, help='pinned local model snapshot directory')
    parser.add_argument('--profile', type=Path, default=Path(__file__).with_name('profile.json'))
    args = parser.parse_args(argv)
    ROOT = args.source_root.resolve()
    BASE = args.output_dir.resolve()
    EXPECTED_SHA = args.expected_commit
    snapshot = args.model_name.resolve()
    profile = args.profile.resolve()
    # Do not modify interpreter search paths or create output before argument validation.
    if len(EXPECTED_SHA) != 40 or any(c not in '0123456789abcdef' for c in EXPECTED_SHA):
        parser.error('--expected-commit requires a full lowercase 40-character SHA')
    if Path(__file__).resolve() != ROOT/'tools/agx3_gsm8k/verify.py':
        parser.error('run verify.py from the selected source checkout')
    if BASE.is_relative_to(ROOT):
        parser.error('--output-dir must be outside the source checkout')
    profile_bytes = profile.read_bytes()
    # A failed attempt remains immutable. A second run requires a separate bundle.
    BASE.mkdir(parents=True, exist_ok=False)
    out = BASE / 'attempt1'
    out.mkdir(exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'src'))
    report = dict(schema='kvolt.gsm8k-environment-energy-agx3-validation.v1',
        source_sha=EXPECTED_SHA, ssh_target='agx32@100.94.58.28', target='AGX3',
        script_sha256=digest(__file__), profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        source_root=str(ROOT), model_snapshot=str(snapshot), status='starting', cases=[], energy=[],
        formal_qualification=False, full_test_benchmark=False,
        sensor_response_validated=False, wall_energy_claimed=False,
        thermal_throttle_qualification=False,
        energy_boundary='complete original kivi_llama.generate: tokenization, H2D, model.generate, continuation decode and CUDA completion; excludes model load, warmup, evidence copies, scoring and artifact I/O',
        instrumentation='GPU tensor references, generation flags and integer quantize/matmul counters inside model.generate hook; all evidence CPU copies, hashes, decode checks and I/O after meter.stop')
    state, captures = {}, []
    torch = helper = vendor = env = None
    original_load = original_generate = raw_quantize = raw_matmul = None
    env_entered = False
    started = time.monotonic()
    code = 0
    try:
        report['source_before'] = source_check(ROOT, EXPECTED_SHA)
        with (BASE/'source-manifest.json').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(report['source_before'], ensure_ascii=False, indent=2)+'\n')
        with (BASE/'profile.json').open('xb') as stream:
            stream.write(profile_bytes)
        serial = Path('/proc/device-tree/serial-number').read_bytes().rstrip(b'\0\n')
        report['device_serial_sha256'] = hashlib.sha256(serial).hexdigest()
        del serial
        assert report['device_serial_sha256'] == EXPECTED_SERIAL_SHA, 'wrong physical device'
        from kvolt_runtime.device.lock import HardwareLock
        with HardwareLock() as lock:
            report['lock_acquired_before_torch_import'] = 'torch' not in sys.modules
            assert report['lock_acquired_before_torch_import']
            report['controls_before'] = controls()
            report['host'] = dict(hostname=platform.node(), kernel=platform.release(),
                machine=platform.machine(), model=Path('/proc/device-tree/model').read_bytes().rstrip(b'\0').decode())
            print('LOCK_ACQUIRED', flush=True)
            try:
                from kvolt_runtime.backend.provenance import validate_sources, import_pinned_source
                report['provenance'] = validate_sources(KIVI, snapshot, native_profile='agx2')
                import torch
                assert torch.cuda.is_available()
                vendor = import_pinned_source(KIVI, report['provenance'])
                from experiments.quality.GSM8K import KIVI_Llama as runner
                import model.kivi_llama as helper
                import kvolt_runtime.backend.vendor.llama_kivi as helper_vendor
                from experiment_control import environment_from_profile
                from kvolt_runtime.telemetry import EnergyMeter, discover_power_profile
                assert helper is runner.kivi_llama and helper_vendor is vendor
                assert Path(helper.__file__).resolve() == ROOT/'src/model/kivi_llama.py'
                assert Path(runner.__file__).resolve() == ROOT/'src/experiments/quality/GSM8K/KIVI_Llama.py'
                report['environment_versions'] = dict(python=sys.version, executable=sys.executable,
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    device=torch.cuda.get_device_name(0), CUDA_HOME=os.environ.get('CUDA_HOME'),
                    LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH'),
                    packages={name: importlib.metadata.version(name) for name in
                              ('transformers', 'flash-attn', 'triton', 'accelerate')})
                records = runner.load_dataset()[:args.limit]
                power_profile = discover_power_profile()
                report['power_profile'] = power_profile.to_dict()
                original_load, original_generate = helper.load_model, helper.generate
                raw_quantize = vendor.triton_quantize_and_pack_along_last_dim
                raw_matmul = vendor.cuda_bmm_fA_qB_outer
                counts = dict(quantize=0, matmul=0)

                def quantize(*args, **kwargs):
                    counts['quantize'] += 1
                    return raw_quantize(*args, **kwargs)

                def matmul(*args, **kwargs):
                    counts['matmul'] += 1
                    return raw_matmul(*args, **kwargs)

                vendor.triton_quantize_and_pack_along_last_dim = quantize
                vendor.cuda_bmm_fA_qB_outer = matmul

                def materialize(record, prediction, warmup):
                    assert len(captures) == 1
                    captured = captures.pop()
                    tensor = captured.pop('output_tensor')
                    inputs = captured.pop('input_tensor')[0].detach().cpu().tolist()
                    full = tensor[0].detach().cpu().tolist()
                    assert tensor.shape[0] == 1 and full[:len(inputs)] == inputs
                    ids = full[len(inputs):]
                    eos = report['model_audit']['eos_token_ids']
                    eos = [eos] if isinstance(eos, int) else list(eos)
                    assert 1 <= len(ids) <= 512 and (ids[-1] in eos or len(ids) == 512)
                    assert captured['flags'] == dict(max_new_tokens=512, do_sample=False, num_beams=1, use_cache=True)
                    assert captured['inference_mode'] and not captured['supplied_past_key_values']
                    assert captured['num_return_sequences'] == 1 and captured['return_dict_in_generate'] is False
                    assert captured['quantize_calls'] >= 0 and captured['matmul_calls'] >= 0
                    # A first-token EOS legitimately omits the decode matmul path.
                    if len(ids) > 1:
                        assert captured['quantize_calls'] > 0 and captured['matmul_calls'] > 0
                    tokenizer = state['tokenizer']
                    assert tokenizer(record['prompt'], return_tensors='pt')['input_ids'][0].tolist() == inputs
                    continuation = tokenizer.decode(ids, skip_special_tokens=True)
                    assert continuation == prediction
                    result = dict(index=len(report['cases']), id=record['id'], warmup=warmup,
                        prompt_sha256=hashlib.sha256(record['prompt'].encode()).hexdigest(),
                        input_token_ids=inputs, input_length=len(inputs),
                        input_ids_sha256=hashlib.sha256(json.dumps(inputs,separators=(',', ':')).encode()).hexdigest(),
                        output_token_ids=ids, output_length=len(ids), full_sequence_token_ids=full,
                        continuation=continuation, eos_token_ids=eos,
                        stop='eos' if ids[-1] in eos else 'max_new_tokens', **captured)
                    report['cases'].append(result)
                    return result

                def load_model(name, **settings):
                    nonlocal env, env_entered
                    load_started = time.monotonic()
                    model, tokenizer = original_load(name, **settings)
                    state.update(model=model, tokenizer=tokenizer)
                    model.generation_config.num_return_sequences = 1
                    model.generation_config.return_dict_in_generate = False
                    audit = dict(parameter_tensors=sum(1 for _ in model.parameters()),
                        all_fp16=all(p.dtype == torch.float16 for p in model.parameters()),
                        all_cuda0=all(p.device == torch.device('cuda:0') for p in model.parameters()),
                        all_eval=all(not m.training for m in model.modules()), settings=settings,
                        layer_count=len(model.model.layers),
                        attention_classes=sorted({type(l.self_attn).__name__ for l in model.model.layers}),
                        eos_token_ids=model.generation_config.eos_token_id,
                        tokenizer_eos_token_id=tokenizer.eos_token_id)
                    report['model_audit'] = audit
                    report['model_load_seconds'] = time.monotonic()-load_started
                    assert all(audit[k] for k in ('all_fp16', 'all_cuda0', 'all_eval'))
                    assert audit['layer_count'] == 32 and settings == runner.KV_SETTINGS
                    assert audit['attention_classes'] == ['LlamaFlashAttention_KIVI']
                    assert all((a.k_bits,a.v_bits,a.group_size,a.residual_length)==(2,2,32,128)
                               for a in (l.self_attn for l in model.model.layers))
                    raw_generate = model.generate
                    state['raw_generate'] = raw_generate

                    def capture(*args, **kwargs):
                        before = dict(counts)
                        output = raw_generate(*args, **kwargs)
                        captures.append(dict(input_tensor=kwargs['input_ids'], output_tensor=output,
                            flags={k: kwargs.get(k) for k in ('max_new_tokens','do_sample','num_beams','use_cache')},
                            inference_mode=torch.is_inference_mode_enabled(),
                            supplied_past_key_values='past_key_values' in kwargs,
                            num_return_sequences=model.generation_config.num_return_sequences,
                            return_dict_in_generate=model.generation_config.return_dict_in_generate,
                            quantize_calls=counts['quantize']-before['quantize'],
                            matmul_calls=counts['matmul']-before['matmul']))
                        return output

                    model.generate = capture

                    def warmup():
                        warmup_started = time.monotonic()
                        prediction = original_generate(model, tokenizer, records[0]['prompt'], max_new_tokens=512)
                        torch.cuda.synchronize(model.device)
                        result = materialize(records[0], prediction, True)
                        result['generation_seconds_including_completion_sync_and_evidence'] = time.monotonic()-warmup_started
                        append_json(out/'warmup.jsonl', result)

                    env = environment_from_profile(BASE/'profile.json', output_dir=out/'environment',
                        warmup=warmup, synchronize=lambda: torch.cuda.synchronize(model.device),
                        lock=lock, borrow_lock=True)
                    env.report['metadata'].update(source_sha=EXPECTED_SHA,
                        profile_sha256=report['profile_sha256'], experiment='gsm8k-agx3-first-N')
                    env.__enter__()
                    env_entered = True
                    print('MODEL_LOADED_ENVIRONMENT_READY', flush=True)
                    return model, tokenizer

                def measured_generate(model, tokenizer, prompt, **settings):
                    env.checkpoint()
                    ordinal = len(report['energy'])
                    record = records[ordinal]
                    assert prompt == record['prompt'] and settings == {'max_new_tokens':512}
                    meter = EnergyMeter(power_profile, synchronize=lambda: torch.cuda.synchronize(model.device))
                    relative = f'energy/{ordinal:06d}'
                    try:
                        with meter:
                            prediction = original_generate(model, tokenizer, prompt, **settings)
                    except BaseException as original:
                        try:
                            meter.save(out/relative)
                            append_json(out/'energy-failures.jsonl', dict(id=record['id'], energy=meter.result,
                                error=dict(type=type(original).__name__, message=str(original))))
                        except BaseException as secondary:
                            report.setdefault('diagnostic_errors', []).append(str(secondary))
                        raise
                    meter.save(out/relative)
                    result = dict(id=record['id'], ordinal=ordinal, energy=meter.result,
                                  energy_artifacts_dir=relative)
                    report['energy'].append(result)
                    append_json(out/'energy-results.jsonl', result)
                    materialize(record, prediction, False)
                    assert meter.result['valid'] and meter.result['completed'], 'invalid energy; raw artifacts preserved'
                    env.checkpoint()
                    print('REQUEST_COMPLETE', ordinal+1, record['id'], flush=True)
                    return prediction

                helper.load_model, helper.generate = load_model, measured_generate
                try:
                    report['runner_summary'] = runner.run(out/'results', limit=args.limit, model_name=str(snapshot))
                finally:
                    if env_entered:
                        active_exception = sys.exc_info()
                        env_entered = False
                        try:
                            env.__exit__(*active_exception)
                        finally:
                            report['environment_report'] = env.report
                assert report['runner_summary']['evaluated'] == len(report['energy']) == args.limit
                assert len(report['cases']) == args.limit+1
                assert all(item['energy']['valid'] for item in report['energy'])
                assert any(case['quantize_calls'] > 0 and case['matmul_calls'] > 0
                           for case in report['cases'] if not case['warmup'])
                assert env.report['preparation']['status'] == 'succeeded'
                assert env.report['preparation']['warmup_completed'] == 1
                assert env.report['workload']['status'] == env.report['restoration']['status'] == 'succeeded'
                assert env.report['environment']['status'] == 'satisfied'
                assert env.report['environment']['violation_count'] == 0
                assert not env.report['monitor']['errors']
                report['status'] = 'passed'
            finally:
                primary_exception = sys.exc_info()[1]
                if helper is not None and original_load is not None:
                    helper.load_model, helper.generate = original_load, original_generate
                if vendor is not None and raw_quantize is not None:
                    vendor.triton_quantize_and_pack_along_last_dim = raw_quantize
                    vendor.cuda_bmm_fA_qB_outer = raw_matmul
                if 'model' in state:
                    state['model'].generate = state.get('raw_generate', state['model'].generate)
                if env is not None:
                    report['environment_report'] = env.report
                    env.warmup = env.synchronize = None
                captures.clear()
                state.clear()
                gc.collect()
                cleanup_errors = []
                try:
                    if torch is not None:
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated()
                        clear = getattr(torch._C, '_cuda_clearCublasWorkspaces', None)
                        if clear is not None:
                            clear()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        report['cleanup'] = dict(allocated_before_workspace_clear=before,
                            workspace_clear_available=clear is not None,
                            allocated_after=torch.cuda.memory_allocated(), reserved_after=torch.cuda.memory_reserved())
                        assert report['cleanup']['allocated_after'] == report['cleanup']['reserved_after'] == 0
                except BaseException as error:
                    cleanup_errors.append(dict(stage='cuda', type=type(error).__name__, message=str(error)))
                try:
                    report['remaining_children'] = [p.pid for p in multiprocessing.active_children()]
                    report['controls_after'] = controls()
                    assert not report['remaining_children']
                    assert report['controls_before'] == report['controls_after']
                    lock.assert_held()
                except BaseException as error:
                    cleanup_errors.append(dict(stage='controls_children_lock', type=type(error).__name__, message=str(error)))
                report['cleanup_errors'] = cleanup_errors
                if cleanup_errors and primary_exception is None:
                    raise RuntimeError('cleanup/post-control checks failed; inspect cleanup_errors')
    except BaseException as error:
        report['status'] = 'blocked' if 'hardware lock busy' in str(error) else 'failed'
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        code = 75 if report['status'] == 'blocked' else 1
    finally:
        try:
            report['source_after'] = source_check(ROOT, EXPECTED_SHA)
            assert report['source_before'] == report['source_after']
            assert digest(BASE/'profile.json') == report['profile_sha256']
            with LOCK.open('rb') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    report['post_lock_available'] = True
                    fcntl.flock(stream, fcntl.LOCK_UN)
                except BlockingIOError:
                    report['post_lock_available'] = False
                    if report['status'] != 'blocked':
                        raise
        except BaseException as error:
            report['postcheck_error'] = dict(type=type(error).__name__, message=str(error))
            report['status'], code = 'failed', 1
        report['wall_seconds'] = time.monotonic()-started
        with (out/'gpu-validation.json').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print('RESULT', report['status'], flush=True)
        if report.get('error'):
            print(report['error']['traceback'], flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
