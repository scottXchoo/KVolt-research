import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from experiment_control import environment_from_profile
from experiment_control.__main__ import main
from experiment_control.profile import load_profile


ROOT = Path(__file__).resolve().parents[1]


def profile_file(tmp_path, **kwargs):
    data = {'schema':'experiment-control.profile.v1', **kwargs}
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(data))
    return path


def test_empty_profile_is_detachable_and_records_policy(tmp_path):
    path = profile_file(tmp_path)
    env = environment_from_profile(path, output_dir=tmp_path/'out')
    assert env.run(lambda: 42) == 42
    assert env.report['metadata']['profile']['schema'] == 'experiment-control.profile.v1'


def test_profile_build_has_no_mutation(tmp_path):
    root = tmp_path/'cpu'
    root.mkdir()
    for name, value in [('scaling_min_freq','10'),('scaling_max_freq','30'),('scaling_governor','g'),('scaling_cur_freq','20')]:
        (root/name).write_text(value)
    lock_path = tmp_path/'lock'
    lock_path.touch()
    path = profile_file(tmp_path,lock_path=str(lock_path),controls=[{
        'kind':'clock','name':'cpu','path':str(root),'minimum':20,'maximum':20,
        'min_file':'scaling_min_freq','max_file':'scaling_max_freq',
        'governor_file':'scaling_governor','current_file':'scaling_cur_freq'}])
    env = environment_from_profile(path,output_dir=tmp_path/'out')
    assert (root/'scaling_min_freq').read_text() == '10'
    with env:
        assert (root/'scaling_min_freq').read_text() == '20'
    assert (root/'scaling_min_freq').read_text() == '10'
    assert (root/'scaling_max_freq').read_text() == '30'


@pytest.mark.parametrize('data', [
    {'schema':'unknown'}, {'schema':'experiment-control.profile.v1','typo':1},
    {'schema':'experiment-control.profile.v1','session':{'sample_interval':1}},
    {'schema':'experiment-control.profile.v1','controls':[{'kind':'file','path':'/a','value':1}]},
    {'schema':'experiment-control.profile.v1','controls':[{'kind':'clock','name':'x','path':'/sys/class/thermal/thermal_zone0','minimum':1,'maximum':1}]},
    {'schema':'experiment-control.profile.v1','controls':[{'kind':'clock','name':'x','path':'/x','minimum':1,'maximum':1,'min_file':'trip_point_0_temp'}]},
    {'schema':'experiment-control.profile.v1','controls':[{'kind':'fan','pwm_path':'/proc/sys/vm/swappiness','pwm':1}]},
    {'schema':'experiment-control.profile.v1','controls':[{'kind':'emc','rate_path':'/tmp/wrong','lock_path':'/tmp/mrq_rate_locked','halt_path':'/tmp/bwmgr_halt','rate_hz':1}]},
    {'schema':'experiment-control.profile.v1','rules':[{'probe':'missing','metric':'a'}]},
    {'schema':'experiment-control.profile.v1','probes':[{'kind':'linux','typo':1}]},
    {'schema':'experiment-control.profile.v1','session':{'warmup_runs':1}},
])
def test_profile_rejects_invalid_or_unsafe_options(tmp_path,data):
    with pytest.raises((ValueError,TypeError)):
        environment_from_profile(data,output_dir=tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('text',['{"schema":"x","schema":"y"}','{"schema":"x","n":NaN}'])
def test_json_rejects_duplicates_nonfinite(tmp_path,text):
    path=tmp_path/'bad.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        load_profile(path)


def test_cli_foreground_success_and_failure(tmp_path):
    path = profile_file(tmp_path)
    assert main(['--profile',str(path),'--output-dir',str(tmp_path/'ok'),'--',sys.executable,'-c','print("ok")']) == 0
    assert main(['--profile',str(path),'--output-dir',str(tmp_path/'bad'),'--',sys.executable,'-c','raise SystemExit(7)']) == 7
    result=json.loads((tmp_path/'bad'/'summary.json').read_text())
    assert result['workload']['status'] == 'failed'
    assert result['workload']['error']['type'] == 'CalledProcessError'


def test_cli_warmup_is_separate_and_failure_stops_workload(tmp_path):
    path = profile_file(tmp_path,session={'warmup_runs':2})
    warmup=json.dumps([sys.executable,'-c','pass'])
    assert main(['--profile',str(path),'--output-dir',str(tmp_path/'out'),'--warmup-command-json',warmup,
                 '--',sys.executable,'-c','pass']) == 0
    result=json.loads((tmp_path/'out'/'summary.json').read_text())
    assert result['preparation']['warmup_completed'] == 2
    fail=json.dumps([sys.executable,'-c','raise SystemExit(9)'])
    assert main(['--profile',str(path),'--output-dir',str(tmp_path/'fail'),'--warmup-command-json',fail,
                 '--',sys.executable,'-c','pass']) == 9
    result=json.loads((tmp_path/'fail'/'summary.json').read_text())
    assert result['workload']['status'] == 'not_started'


def test_cli_signal_cleanup_and_signal_handler_restoration(tmp_path):
    path = profile_file(tmp_path)
    marker=tmp_path/'started'
    command=[sys.executable,'-m','experiment_control','--profile',str(path),'--output-dir',str(tmp_path/'out'),'--',
             sys.executable,'-c',f'from pathlib import Path; import time; Path({str(marker)!r}).touch(); time.sleep(30)']
    process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')})
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:
            time.sleep(.02)
        assert marker.exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=8) == 128+signal.SIGTERM
        result=json.loads((tmp_path/'out'/'summary.json').read_text())
        assert result['workload']['status'] == 'failed'
        assert result['workload']['error']['signum'] == signal.SIGTERM
        assert result['restoration']['status'] == 'succeeded'
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    before=signal.getsignal(signal.SIGTERM)
    assert main(['--profile',str(path),'--output-dir',str(tmp_path/'another'),'--',sys.executable,'-c','pass']) == 0
    assert signal.getsignal(signal.SIGTERM) == before


def test_profile_required_linux_metric_fails_without_sources(tmp_path):
    path=profile_file(tmp_path,probes=[{'kind':'linux','sys_root':str(tmp_path/'sys'),'proc_root':str(tmp_path/'proc')}],
                      rules=[{'probe':'linux','metric':'temperature.GPU-therm.c','maximum':60}])
    env=environment_from_profile(path,output_dir=tmp_path/'out')
    with pytest.raises(RuntimeError):
        env.run(lambda: pytest.fail('missing required sensor cannot pass'))
