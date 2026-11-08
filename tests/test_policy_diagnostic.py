"""CPU diagnostics: synthetic assumptions, real policy selection and safe output."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kvolt_runtime import cli
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.types import KneeResult, PolicyDependencyError

ROOT = Path(__file__).resolve().parents[1]


def arguments(*extra):
    return ['diagnose', '--length', '1536', '--mock', 'baseline',
            '--pc-elapsed-ns', '1000000', '--dc-elapsed-ns', '2000000', *extra]


def run_cli(*extra, module=True):
    command = ([sys.executable, '-m', 'kvolt_runtime'] if module else
               [sys.executable, str(ROOT / 'tools/diagnose_policy.py')])
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')}
    return subprocess.run([*command, *extra], env=env, text=True, capture_output=True)


class NoKnee:
    def detect(self, quality):
        return KneeResult('not_found', reason='no_knee')


@pytest.fixture
def fake_detector(monkeypatch):
    monkeypatch.setattr(cli, '_new_knee_adapter', lambda: NoKnee())


def report(capsys):
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert '\n' not in captured.out.rstrip('\n')
    return result


def test_full_trace_and_independent_argmin(fake_detector, capsys):
    assert cli.main(arguments()) == 0
    data = report(capsys)
    assert data['schema_version'] == 'kvolt.policy_diagnostic.v1'
    assert data['record_type'] == 'policy_diagnostic'
    assert data['synthetic_assumptions'] and data['predictions_only']
    assert data['measurements'] is None
    assert data['source']['kind'] == 'synthetic'
    pc, dc = data['prefill'], data['decode']
    assert len(pc['candidates']) == 8 and len(dc['candidates']) == 40
    assert len(dc['quality']) == 5
    assert pc['elapsed_ns'] == 1000000 and dc['elapsed_ns'] == 2000000
    assert pc['prefilter_ns'] == pc['tp_ns'] == 1000000
    assert pc['budget_s'] == pc['target_s'] - .001
    assert dc['budget_s'] == dc['target_s'] - .002 / 511
    prefills = [c['prediction'] for c in pc['candidates'] if c['meets_budget']]
    assert pc['selected']['frequency_mhz'] == 510
    assert dc['selected']['group'] == 128 and dc['selected']['frequency_mhz'] == 408
    assert pc['selected'] == min(prefills, key=lambda p: (p['energy_j'], p['ttft_s'], p['frequency_mhz']))
    rejected = [c for c in pc['candidates'] if not c['meets_budget']]
    assert rejected and all(c['prediction']['energy_j'] is None for c in rejected)
    eligible = [c for c in dc['candidates'] if c['eligible']]
    expected = min(eligible, key=lambda c: (c['prediction']['energy_j_per_token'], -c['quality'], c['prediction']['tpot_s'], c['prediction']['group'], c['prediction']['frequency_mhz']))
    assert dc['selected'] == expected['prediction']
    assert sum(c['selected'] for c in dc['candidates']) == 1
    assert data['assumptions']['first_token_id'] == 17
    assert data['assumptions']['times_measured'] is False
    assert data['runtime']['source_sha256']['cli.py']


@pytest.mark.parametrize('flags,reason', [(['--max-new-tokens','1'], 'nmax_one'),
                                        (['--first-token-id','128001'], 'first_token_eos'),
                                        (['--max-new-tokens','1','--first-token-id','128009'], 'first_token_eos')])
def test_skip_without_detector_quality_decode_or_clock(monkeypatch, capsys, flags, reason):
    events = []
    class Spy(MockLUTRepository):
        def preflight(self, length):
            events.append('preflight'); return super().preflight(length)
        def prefill(self, length):
            pytest.fail('online full-details Prefill lookup')
        def prefill_lookup(self, length, *, latency_only=False):
            assert latency_only is True
            events.append('prefill_lookup'); return super().prefill_lookup(length, latency_only=True)
        def prefill_latencies(self, length):
            pytest.fail('online separate Prefill latency lookup')
        def prefill_candidates(self, length, frequencies):
            events.append('prefill_candidates'); return super().prefill_candidates(length, frequencies)
        def prefill_target(self, length):
            pytest.fail('online separate Prefill target lookup')
        def quality(self, length):
            pytest.fail('quality on skip')
        def decode(self, length):
            pytest.fail('decode on skip')
        def decode_lookup(self, length):
            pytest.fail('combined decode lookup on skip')
        def decode_target(self, length):
            pytest.fail('decode target on skip')
    monkeypatch.setattr(cli, '_build_repository', lambda args: Spy())
    monkeypatch.setattr(cli, '_new_knee_adapter', lambda: pytest.fail('detector initialization on skip'))
    assert cli.main(arguments(*flags)) == 0
    data = report(capsys)
    assert events == ['preflight','prefill_lookup','prefill_candidates']
    dc = data['decode']
    assert dc['status'] == 'skipped' and dc['skip_reason'] == reason
    for key in ('selected','target_s','budget_s','elapsed_ns','td_ns','knee','predicted_feasible'):
        assert dc[key] is None
    assert dc['candidates'] == [] and dc['quality'] == []
    assert not data['assumptions']['dc_elapsed_applied']


def test_preflight_failure_prevents_online_queries(monkeypatch, capsys):
    class Bad(MockLUTRepository):
        def preflight(self, length):
            raise ValueError('unsupported fixture')
        def prefill(self, length):
            pytest.fail('query after failed preflight')
        def prefill_lookup(self, length, *, latency_only=False):
            pytest.fail('combined lookup after failed preflight')
        def prefill_target(self, length):
            pytest.fail('target query after failed preflight')
        def prefill_latencies(self, length):
            pytest.fail('latency query after failed preflight')
        def prefill_candidates(self, length, frequencies):
            pytest.fail('detail query after failed preflight')
    monkeypatch.setattr(cli, '_build_repository', lambda args: Bad())
    assert cli.main(arguments()) == 2
    assert report(capsys)['error']['stage'] == 'preflight'


@pytest.mark.parametrize('name,value', [('--length','1023'),('--length','8193'),
    ('--max-new-tokens','0'),('--max-new-tokens','513'),('--pc-elapsed-ns','-1'),
    ('--dc-elapsed-ns','-1'),('--first-token-id','128256'),('--first-token-id','-1')])
def test_invalid_semantics_rejected_before_lut(monkeypatch,capsys,name,value):
    monkeypatch.setattr(cli, '_build_repository', lambda args: pytest.fail('LUT queried for invalid input'))
    assert cli.main(arguments(name,value)) == 2
    data=report(capsys)
    assert data['status']=='rejected' and data['error']['stage']=='input'
    assert data['measurements'] is None


def test_missing_active_dc_time_before_lut(monkeypatch,capsys):
    monkeypatch.setattr(cli, '_build_repository', lambda args: pytest.fail('LUT before timing validation'))
    assert cli.main(['diagnose','--length','1024','--mock','baseline','--pc-elapsed-ns','0']) == 2
    assert report(capsys)['error']['stage']=='input'


def test_dependency_failure_is_not_recovered(monkeypatch,capsys):
    def unavailable():
        raise PolicyDependencyError('missing actual pinned package')
    monkeypatch.setattr(cli,'_new_knee_adapter',unavailable)
    assert cli.main(arguments()) == 2
    data=report(capsys)
    assert data['status']=='error' and data['error']['stage']=='dependency'
    assert 'decode' not in data
    assert 'error_recovered' not in json.dumps(data)


def test_exclusive_output_preserves_attempt(fake_detector,capsys,tmp_path):
    path=tmp_path/'report.json'
    assert cli.main(arguments('--output',str(path))) == 0
    data=report(capsys)
    assert json.loads(path.read_text())==data
    before=path.read_bytes()
    assert cli.main(arguments('--output',str(path))) == 2
    assert report(capsys)['error']['stage']=='output'
    assert path.read_bytes()==before


def test_output_missing_parent_rejected(capsys,tmp_path):
    assert cli.main(arguments('--output',str(tmp_path/'absent'/'report.json'))) == 2
    assert report(capsys)['error']['stage']=='output'


def test_missing_measured_bundle_is_not_mock(capsys,tmp_path):
    argv=['diagnose','--length','1024','--lut-bundle',str(tmp_path/'absent'),
          '--max-new-tokens','1','--pc-elapsed-ns','0']
    assert cli.main(argv)==2
    data=report(capsys)
    assert data['status']=='rejected' and data['error']['stage']=='lut'
    assert 'source' not in data


@pytest.mark.parametrize('extra', [[],['run'],['diagnose'],
    ['diagnose','--length','a'],arguments('--mock','unknown'),
    arguments('--lut-bundle','anything'),arguments('--pc-elapsed-ns','1.5')])
def test_syntax_is_argparse_exit_two(extra):
    p=run_cli(*extra)
    assert p.returncode==2 and 'usage:' in p.stderr
    if extra == ['run']:
        result=json.loads(p.stdout)
        assert result['record_type']=='session_result' and result['exit_code']==2
        assert result['session_path'] is None
    else:
        assert p.stdout==''


def test_help_and_import_have_no_scientific_or_gpu_imports():
    code='''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','scipy','kneed','torch','transformers','tokenizers'}:
            raise RuntimeError('forbidden import '+fullname)
sys.meta_path.insert(0,Guard())
from kvolt_runtime.cli import main
raise SystemExit(main(['diagnose','--help']))
'''
    p=subprocess.run([sys.executable,'-c',code],env={**os.environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert 'assumed' in p.stdout.lower()


def test_direct_wrapper_skip_works_without_package_install():
    p=run_cli('--length','1024','--mock','baseline','--pc-elapsed-ns','0','--max-new-tokens','1',module=False)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['decode']['status']=='skipped'


@pytest.mark.parametrize('scenario',['baseline','flat_quality','nonmonotonic','exact_ties'])
def test_actual_cli_mock_scenarios(scenario):
    p=run_cli(*arguments('--mock',scenario))
    assert p.returncode==0,p.stderr+'\n'+p.stdout
    data=json.loads(p.stdout)
    assert data['source']['kind']=='synthetic'
    assert data['source']['metadata']['scenario']==scenario
    assert data['runtime']['loaded_versions']=={'numpy':'1.26.4','scipy':'1.15.3','kneed':'0.9.0'}
    assert len(data['decode']['candidates'])==40
    if scenario in ('flat_quality','exact_ties'):
        assert data['decode']['knee']['reason']=='flat_quality'
    if scenario=='exact_ties':
        assert data['prefill']['selected']['frequency_mhz']==408
        assert data['decode']['selected']['group']==8
        assert data['decode']['selected']['frequency_mhz']==408


def test_actual_cli_measured_bundle():
    bundle=Path(os.environ.get('KVOLT_LUT_BUNDLE',ROOT.parent/'KVolt-lut-source/data/luts/agx2-repeat5-20261103'))
    if not bundle.exists() and 'KVOLT_LUT_BUNDLE' not in os.environ:
        pytest.skip('published optional LUT data unavailable')
    argv=['diagnose','--length','1536','--lut-bundle',str(bundle),'--pc-elapsed-ns','1000000','--dc-elapsed-ns','2000000']
    p=run_cli(*argv)
    assert p.returncode==0,p.stderr+'\n'+p.stdout
    data=json.loads(p.stdout)
    assert data['source']['kind']=='measured'
    assert len(data['source']['files_sha256'])==10
    assert data['source']['metadata']['quality']['runtime_eligible'] is False
    assert data['source']['metadata']['quality']['runtime_validation_required'] is True
    assert 'entries' not in data['source']['metadata']['quality']
    assert data['measurements'] is None and data['synthetic_assumptions']
    assert len(data['prefill']['candidates'])==8 and len(data['decode']['candidates'])==40


def test_online_order_and_assumed_clock_boundaries(monkeypatch,capsys):
    events=[]
    class Spy(MockLUTRepository):
        def preflight(self,L):
            events.append('preflight'); return super().preflight(L)
        def prefill(self,L):
            pytest.fail('online full-details Prefill lookup')
        def prefill_lookup(self,L,*,latency_only=False):
            assert latency_only is True
            events.append('prefill_lookup'); return super().prefill_lookup(L,latency_only=True)
        def prefill_latencies(self,L):
            pytest.fail('online separate Prefill latency lookup')
        def prefill_candidates(self,L,frequencies):
            events.append('prefill_candidates'); return super().prefill_candidates(L,frequencies)
        def prefill_target(self,L):
            pytest.fail('online separate Prefill target lookup')
        def quality(self,L):
            events.append('quality'); return super().quality(L)
        def decode_lookup(self,L):
            events.append('decode_lookup'); return super().decode_lookup(L)
        def decode(self,L):
            pytest.fail('online separate Decode lookup')
        def decode_target(self,L):
            pytest.fail('online separate Decode target lookup')
    class Detector:
        def detect(self,quality):
            events.append('detect'); return KneeResult('not_found',reason='no_knee')
    monkeypatch.setattr(cli,'_build_repository',lambda args: Spy())
    monkeypatch.setattr(cli,'_new_knee_adapter',lambda: Detector())
    assert cli.main(arguments())==0
    data=report(capsys)
    assert events==['preflight','prefill_lookup','prefill_candidates',
                    'quality','detect','decode_lookup']
    assert data['prefill']['t0_ns']==0 and data['prefill']['tp_ns']==1000000
    assert data['decode']['t1_ns']==0 and data['decode']['td_ns']==2000000


def test_mislabeled_source_rejected(monkeypatch,capsys):
    from dataclasses import replace
    repo=MockLUTRepository()
    repo.provenance=replace(repo.provenance,kind='measured')
    monkeypatch.setattr(cli,'_build_repository',lambda args: repo)
    assert cli.main(arguments())==2
    assert 'provenance kind' in report(capsys)['error']['reason']


def test_pins_and_console_entrypoint_metadata():
    # Python 3.10 has no tomllib; packaging strings are also verified there.
    text=(ROOT/'pyproject.toml').read_text()
    assert 'dependencies = []' in text
    assert 'requires-python = ">=3.10"' in text
    assert '[project.optional-dependencies]' in text
    for package,version in [('numpy','1.26.4'),('scipy','1.15.3'),('kneed','0.9.0')]:
        assert f'"{package}=={version}"' in text
    assert 'kvolt = "kvolt_runtime.cli:main"' in text


@pytest.mark.parametrize('flags',[['--max-new-tokens','1'],['--first-token-id','128008']])
def test_skip_can_omit_dc_time(capsys,flags):
    argv=['diagnose','--length','1024','--mock','baseline','--pc-elapsed-ns','0',*flags]
    assert cli.main(argv)==0
    data=report(capsys)
    assert data['assumptions']['dc_elapsed_ns'] is None
    assert data['decode']['elapsed_ns'] is None


def test_negative_budgets_keep_full_trace(fake_detector,capsys):
    assert cli.main(arguments('--pc-elapsed-ns','1000000000000','--dc-elapsed-ns','1000000000000'))==0
    data=report(capsys)
    for phase in ('prefill','decode'):
        decision=data[phase]
        assert decision['budget_s']<0
        assert decision['selection_path']=='latency_fallback'
        assert decision['predicted_feasible'] is False
        assert not any(c['meets_budget'] for c in decision['candidates'])
        selected=[c for c in decision['candidates'] if c['selected']]
        assert len(selected)==1 and 'latency_fallback' in selected[0]['reasons']
        assert decision['selected']['frequency_mhz']==1122
    assert data['decode']['selected']['group']==128


def test_rejected_attempt_saved_without_measurements(capsys,tmp_path):
    path=tmp_path/'rejected.json'
    assert cli.main(arguments('--length','9000','--output',str(path)))==2
    data=report(capsys)
    assert json.loads(path.read_text())==data
    assert data['measurements'] is None and data['status']=='rejected'
    assert data['runtime']['python']
    assert 'prefill' not in data and 'decode' not in data
