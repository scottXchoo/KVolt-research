import json
import time
import pytest
from kvolt_runtime.errors import ContractError
from kvolt_runtime.telemetry import (RAILS, INAPowerCollector, PowerProfile, RailPaths,
                                     read_sample, samples_csv)


def profile(tmp_path, **kwargs):
    rails={}
    for index,rail in enumerate(RAILS):
        folder=tmp_path/str(index); folder.mkdir()
        for name,value in [('label',rail),('current','-25'),('voltage','12000')]:
            (folder/name).write_text(value)
        rails[rail]=dict(label_path=str(folder/'label'),current_path=str(folder/'current'),
                         voltage_path=str(folder/'voltage'))
    return PowerProfile(rails=rails,interval_ns=5_000_000,max_gap_ns=100_000_000,
        ready_timeout_ns=2_000_000_000,coverage_timeout_ns=200_000_000,
        join_timeout_ns=1_000_000_000,**kwargs)


def test_spawn_real_file_sensor_readiness_coverage_and_idempotent_stop(tmp_path):
    config=profile(tmp_path)
    collector=INAPowerCollector(config)
    assert collector.start()['ready']
    end=time.monotonic_ns()
    result=collector.finish(end)
    assert result['status']=='stopped' and result['start_method']=='spawn'
    assert collector.stop() is result
    assert not collector._process.is_alive()
    for rail in RAILS:
        rows=[row for row in result['samples'] if row['rail']==rail]
        assert rows[0]['timestamp_ns'] < end < rows[-1]['timestamp_ns']
        assert all(row['power_w']==-.3 for row in rows)
    assert samples_csv(result['samples']).startswith('rail,timestamp_ns,')
    assert len(result['trace_sha256'])==64


def test_read_window_fake_clock_keeps_signed_values_and_error(tmp_path):
    config=profile(tmp_path)
    ticks=iter([10,14,20,26])
    row=read_sample(RAILS[0],config.rails[RAILS[0]],clock=lambda:next(ticks))
    assert (row['read_start_ns'],row['timestamp_ns'],row['read_end_ns'])==(10,12,14)
    assert row['power_w']==-.3
    bad=read_sample('wrong-label',config.rails[RAILS[0]],clock=lambda:next(ticks))
    assert bad['error'] and bad['power_w'] is None


def test_readiness_error_preserves_raw_sensor_failure_and_process_stops(tmp_path):
    config=profile(tmp_path)
    from dataclasses import replace
    config=replace(config,ready_timeout_ns=150_000_000)
    __import__('pathlib').Path(config.rails[RAILS[0]].current_path).write_text('bad')
    collector=INAPowerCollector(config)
    assert not collector.start()['ready']
    receipt=collector.stop()
    assert receipt['errors'] and receipt['status']=='error'
    assert not collector._process.is_alive()
    assert any(row['error'] for row in receipt['samples'])


def test_prestart_stop_and_json_validation(tmp_path):
    config=profile(tmp_path)
    collector=INAPowerCollector(config)
    assert collector.stop()['status']=='not_started'
    with pytest.raises(ContractError):
        collector.start()
    path=tmp_path/'profile.json'; path.write_text(json.dumps(config.to_dict()))
    assert PowerProfile.from_json(path).to_dict()==config.to_dict()
    path.write_text('{"interval_ns":1,"interval_ns":2}')
    with pytest.raises(ContractError):
        PowerProfile.from_json(path)


def test_profile_is_immutable_and_serializes_fresh_copies(tmp_path):
    config=profile(tmp_path)
    before=config.to_dict()
    with pytest.raises(TypeError):
        config.rails[RAILS[0]]=config.rails[RAILS[1]]
    with pytest.raises(TypeError):
        del config.rails[RAILS[0]]
    original=config.to_dict()
    original['rails'][RAILS[0]]['current_path']='/changed'
    assert config.to_dict()==before
    import pickle
    restored=pickle.loads(pickle.dumps(config))
    assert restored.to_dict()==before
    with pytest.raises(TypeError):
        restored.rails[RAILS[0]]=restored.rails[RAILS[1]]


@pytest.mark.parametrize('field', ['interval_ns','max_gap_ns','ready_timeout_ns',
    'coverage_timeout_ns','join_timeout_ns','poll_interval_ns'])
def test_profile_rejects_wait_above_twenty_four_hours(tmp_path, field):
    value=profile(tmp_path).to_dict()
    value[field]=86_400_000_000_001
    with pytest.raises(ContractError):
        PowerProfile(**value)


@pytest.mark.parametrize('field', ['name','evidence','label_path','current_path','voltage_path'])
def test_profile_rejects_lone_surrogates_early(tmp_path, field):
    value=profile(tmp_path).to_dict()
    if field in ('name','evidence'):
        value[field]='invalid\ud800'
    else:
        value['rails'][RAILS[0]][field]='/invalid\ud800'
    with pytest.raises(ContractError):
        PowerProfile(**value)
