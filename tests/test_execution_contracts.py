import signal
from pathlib import Path

import pytest

from kvolt_runtime.contracts import PromptInput, RequestSpec
from kvolt_runtime.errors import ContractError
from kvolt_runtime.execution.contracts import (
    ExecutionError, PreparedRequest, SessionOptions, StopController, error_record,
)


def test_prepared_metadata_is_an_immutable_snapshot(tmp_path):
    path = tmp_path / 'prompt.txt'
    path.write_bytes(b'hello\r\n')
    prompt = PromptInput.read(path)
    request = PreparedRequest(RequestSpec('r', (1,), prompt.sha256), prompt, '{"b":[2],"a":1}')
    request.metadata['b'].append(3)
    assert request.metadata == {'a': 1, 'b': [2]}
    assert request.metadata_json == '{"a":1,"b":[2]}'
    assert prompt.text == 'hello\r\n'


@pytest.mark.parametrize('metadata', ['null', '[]', '{"x":NaN}', '{"x":Infinity}',
                                     '{"x":1,"x":2}', '{"x":"\\ud800"}', 'invalid'])
def test_bad_metadata_is_rejected(tmp_path, metadata):
    prompt = PromptInput(tmp_path, 'a', 'a' * 64)
    with pytest.raises(ContractError):
        PreparedRequest(RequestSpec('r', (1,), 'a' * 64), prompt, metadata)


def test_prompt_hash_mismatch_is_rejected(tmp_path):
    with pytest.raises(ContractError):
        PreparedRequest(RequestSpec('r', (1,), 'a' * 64), PromptInput(tmp_path, '', 'b' * 64))


def test_signal_only_sets_first_reason():
    stop = StopController()
    assert not stop.requested and stop.exit_code is None
    stop.request(signal.SIGINT.value)
    stop.request(signal.SIGTERM.value)
    assert stop.requested and stop.signal_number == 2 and stop.exit_code == 130
    terminated = StopController()
    terminated.request(15)
    assert terminated.exit_code == 143


@pytest.mark.parametrize('value', [True, 0, -1, 65, 2.0, '2'])
def test_bad_signal_rejected(value):
    with pytest.raises(ContractError):
        StopController().request(value)


def test_technical_record_preserves_receipt_and_cause():
    try:
        try:
            raise OSError('readback')
        except OSError as cause:
            raise ExecutionError('GPU_APPLY_FAILED', 'device', 'failed', receipt={'changed': True}) from cause
    except ExecutionError as error:
        row = error_record(error)
    assert row['code'] == 'GPU_APPLY_FAILED'
    assert row['receipt']['changed'] is True
    assert row['cause'] == {'type': 'OSError', 'message': 'readback'}


def test_options_keep_performance_power_requirement():
    with pytest.raises(ContractError):
        SessionOptions(Path('/model'), Path('/result'), mode='performance')
    assert SessionOptions(Path('/model'), Path('/result')).mode == 'development'
