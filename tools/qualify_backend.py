#!/usr/bin/env python3
"""M1 correctness only: exclusive GPU lock precedes GPU imports/model load."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@contextmanager
def hardware_lock(path):
    with Path(path).open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('GPU hardware lock busy; no GPU imports or model load performed') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def device_report(torch):
    """Read stable fields; NVIDIA device-properties repr may contain bad UTF-8."""
    properties=torch.cuda.get_device_properties(0)
    return {'name':properties.name,'major':properties.major,'minor':properties.minor,
            'total_memory':properties.total_memory,
            'multi_processor_count':properties.multi_processor_count,
            'torch_cuda':torch.version.cuda}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['smoke','full','boundaries'],required=True)
    parser.add_argument('--inputs',type=Path,default=Path('artifacts/m1/cohort/inputs.json'))
    parser.add_argument('--manifest',type=Path,default=Path('artifacts/m1/cohort/manifest.json'))
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--kivi-root',type=Path,default=Path('/home/agx3/kvolt/third_party/KIVI'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-budget-mib',type=int,default=3072)
    parser.add_argument('--lock',type=Path,default=Path('/run/lock/kvolt-p3-hardware.lock'))
    args=parser.parse_args(argv)
    # CPU preflight runs before the lock. No torch/CUDA module is imported here.
    from kvolt_runtime.config import RuntimeConfig
    from kvolt_runtime.qualification.compare import load_cohort,build_cases,compare_generation,compare_cache,bounded_chunks
    from kvolt_runtime.qualification.reference import generate_reference,generate_candidate
    rows,cohort=load_cohort(args.inputs,args.manifest)
    cases=build_cases(rows,args.suite)
    if args.cache_budget_mib<1 or args.cache_budget_mib>4096:
        raise RuntimeError('cache disk budget must be 1..4096 MiB')
    chunks=bounded_chunks(cases,args.cache_budget_mib*1024**2)
    repeat_keys=[]
    repeat_conditions=set()
    for case in cases:
        condition=(case.request.length,case.group,case.request.nmax)
        if condition not in repeat_conditions:
            repeat_conditions.add(condition)
            repeat_keys.append(case.key)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('output must be empty; automatic resume/retry is intentionally unsupported')
    root=Path(__file__).resolve().parents[1]
    source_paths=sorted((root/'src').rglob('*.py'))+[Path(__file__).resolve()]
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    plan={'schema':'kvolt.m1.qualification.v1','suite':args.suite,'cohort':cohort,
          'runtime_config':asdict(RuntimeConfig()),'source_sha256':sources,
          'runtime_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'runtime_status':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),
          'cases':[{'key':c.key,'group':c.group,'request_id':c.request.request_id,'length':c.request.length,
                    'nmax':c.request.nmax,'token_sha256':c.request.token_sha256,'prompt_sha256':c.request.prompt_sha256} for c in cases],
          'generation_policy':{'greedy':True,'natural_eos':[128001,128008,128009],'performance_or_energy_claims':False},
          'initial_cache_proof':'exact CPU tensor bytes; reference tensors retained on disk until pair passes',
          'batches':[[c.key for c in chunk] for chunk in chunks],
          'cache_budget_mib':args.cache_budget_mib,'native_repeat_cases':repeat_keys,
          'first_eos_gpu':'not covered unless a natural first-EOS case is supplied by an approved frozen plan'}
    with hardware_lock(args.lock):
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/'plan.json',plan)
        # Imports that can lead to torch are deliberately below exclusive lock.
        from kvolt_runtime.backend.model import load_model,InferenceBackend
        free=shutil.disk_usage(args.output).free
        if free < (args.cache_budget_mib+1024)*1024**2:
            raise RuntimeError('insufficient free disk for bounded cache proof and 1 GiB reserve')
        def assert_sources_unchanged():
            if any(hashlib.sha256((root/path).read_bytes()).hexdigest()!=sha for path,sha in sources.items()):
                raise RuntimeError('runtime source changed during qualification; stop without retry')
        completed=0
        for batch_index,chunk in enumerate(chunks):
            prefix=f'batch-{batch_index:03d}'
            assert_sources_unchanged()
            loaded=load_model(args.snapshot,RuntimeConfig(),kivi_root=args.kivi_root,raw_prefill=False)
            torch=loaded.torch
            write_json(args.output/(prefix+'.native-provenance.json'),loaded.provenance)
            write_json(args.output/'device.json',device_report(torch))
            reference={}
            def save_initial(key,cache):
                # Both clone and contiguous are intentional: native tails may
                # be views into full raw KV, including CPU source storage.
                cpu=tuple(tuple(t.detach().contiguous().cpu().clone() if hasattr(t,'detach') else t for t in row) for row in cache)
                torch.save(cpu,args.output/(key+'.cache.pt'))
                fields=[]
                for layer,row in enumerate(cpu):
                    for field,tensor in enumerate(row):
                        if hasattr(tensor,'shape'):
                            fields.append({'layer':layer,'field':field,'shape':list(tensor.shape),'dtype':str(tensor.dtype),
                                           'bytes':tensor.numel()*tensor.element_size(),
                                           'sha256':hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()})
                write_json(args.output/(key+'.cache-proof.json'),{'fields':fields,'logical_bytes':sum(f['bytes'] for f in fields)})
            try:
                for index,case in enumerate(chunk):
                    print(f'{prefix} native {index+1}/{len(chunk)} {case.key}',flush=True)
                    record=generate_reference(loaded,case.request,case.group,initial_cache=lambda cache,key=case.key:save_initial(key,cache))
                    reference[case.key]=record
                    write_json(args.output/(case.key+'.native.json'),dict(asdict(record),batch=prefix))
                # Frozen representative repeats occur once per L/G/cap in
                # the suite, independent of storage batch boundaries.
                for case in chunk:
                    if case.key not in repeat_keys: continue
                    repeat=generate_reference(loaded,case.request,case.group)
                    write_json(args.output/(case.key+'.repeat.json'),dict(asdict(repeat),batch=prefix))
                    difference=compare_generation(reference[case.key],repeat)
                    if difference:
                        write_json(args.output/'first-mismatch.json',{'phase':'native_repeat','case':case.key,'batch':prefix,'difference':difference})
                        raise RuntimeError('native baseline is not repeatable; qualification stopped')
            finally:
                loaded=None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            backend=InferenceBackend(kivi_root=args.kivi_root)
            try:
                assert_sources_unchanged()
                backend.load_session(args.snapshot,RuntimeConfig())
                write_json(args.output/(prefix+'.candidate-provenance.json'),backend.provenance)
                for index,case in enumerate(chunk):
                    print(f'{prefix} candidate {index+1}/{len(chunk)} {case.key}',flush=True)
                    def check_initial(cache):
                        expected=torch.load(args.output/(case.key+'.cache.pt'),map_location='cpu',weights_only=True)
                        difference=compare_cache(expected,cache,torch)
                        if difference:
                            save_initial(case.key+'.candidate-mismatch',cache)
                            write_json(args.output/'first-mismatch.json',{'phase':'initial_cache','case':case.key,'batch':prefix,'difference':difference})
                            raise RuntimeError('initial cache mismatch; qualification stopped')
                    def check_first(token_id):
                        expected=reference[case.key].token_ids[0]
                        if expected!=token_id:
                            write_json(args.output/(case.key+'.candidate.json'),{'token_ids':[token_id],'aborted_at_first_token':True,'batch':prefix})
                            write_json(args.output/'first-mismatch.json',{'phase':'generation','case':case.key,'batch':prefix,
                                'difference':{'kind':'token','index':0,'expected':expected,'actual':token_id}})
                            raise RuntimeError('first token mismatch; qualification stopped before packing')
                    actual=generate_candidate(backend,case.request,case.group,initial_cache=check_initial,first_token=check_first)
                    write_json(args.output/(case.key+'.candidate.json'),dict(asdict(actual),batch=prefix))
                    difference=compare_generation(reference[case.key],actual)
                    if difference:
                        write_json(args.output/'first-mismatch.json',{'phase':'generation','case':case.key,'batch':prefix,'difference':difference})
                        raise RuntimeError('token/stop/cache length mismatch; qualification stopped')
                    proof=args.output/(case.key+'.cache.pt')
                    if proof.exists(): proof.unlink()
                    completed+=1
                    write_json(args.output/'progress.json',{'passed_pairs':completed,'total_pairs':len(cases),'batch':prefix})
            finally:
                backend=None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        assert_sources_unchanged()
        write_json(args.output/'summary.json',{'passed_pairs':len(cases),'all_tokens_and_stop_reasons_exact':True,
            'batches':len(chunks),'first_eos_gpu_qualified':False,'performance_or_energy_claims':False})
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
