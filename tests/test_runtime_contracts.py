import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kvolt_runtime.config import RuntimeConfig, resolve_nmax
from kvolt_runtime.contracts import (
    CacheFormat, CacheHandle, CacheMetadata, PromptInput, RequestSpec,
    StopReason, TokenResult, generation_stop, ns_to_seconds, mhz_to_hz, token_ids_sha256,
)
from kvolt_runtime.errors import CacheStateError, ContractError


@pytest.mark.parametrize('value', [0, 513, -1, True, 1.0, '2', None])
def test_nmax_rejects_invalid_values(value):
    with pytest.raises(ContractError):
        resolve_nmax(item=value)


def test_nmax_precedence_and_limits():
    assert resolve_nmax() == 512
    assert resolve_nmax(cli=128) == 128
    assert resolve_nmax(item=1, cli=128) == 1
    assert resolve_nmax(item=512) == 512


def test_nmax_does_not_hide_invalid_cli_behind_item_override():
    with pytest.raises(ContractError):
        resolve_nmax(item=128, cli=True)


def test_prompt_bytes_and_no_automatic_special_tokens(tmp_path):
    raw = b' <|begin_of_text|>hello\r\n\n '
    path = tmp_path / 'prompt.txt'
    path.write_bytes(raw)
    prompt = PromptInput.read(path)
    assert prompt.text == raw.decode('utf-8')
    assert prompt.sha256 == hashlib.sha256(raw).hexdigest()
    calls = []
    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return {'input_ids': [128000, 1, 2]}
    request = RequestSpec.from_prompt('r1', prompt, tokenizer, nmax=1)
    assert calls == [(prompt.text, {'add_special_tokens': False})]
    assert request.token_ids == (128000, 1, 2)
    assert request.length == 3
    assert request.token_sha256 == hashlib.sha256(b'[128000,1,2]').hexdigest()


def test_prompt_rejects_invalid_utf8(tmp_path):
    path = tmp_path / 'invalid.txt'
    path.write_bytes(b'\xff')
    with pytest.raises(ContractError, match='UTF-8'):
        PromptInput.read(path)


@pytest.mark.parametrize('ids', [(), (True,), (1.0,), (-1,), (128256,), [1]])
def test_request_rejects_invalid_token_contract(ids):
    with pytest.raises(ContractError):
        RequestSpec('r1', ids, 'a' * 64, 128)


def test_hash_is_canonical_not_python_repr():
    assert token_ids_sha256((1, 2, 3)) == hashlib.sha256(b'[1,2,3]').hexdigest()


def test_model_contract_is_fixed():
    config = RuntimeConfig()
    assert config.snapshot == '0e9e39f249a16976918f6564b8830bc894c89659'
    assert config.eos_token_ids == (128001, 128008, 128009)
    assert config.groups == (8, 16, 32, 64, 128)
    assert (config.dtype, config.key_bits, config.value_bits, config.residual_length) == ('float16', 2, 2, 128)
    with pytest.raises(ContractError):
        RuntimeConfig(dtype='bfloat16')


@pytest.mark.parametrize('override', [
    {'groups': (8.0, 16.0, 32.0, 64.0, 128.0)},
    {'eos_token_ids': (128001.0, 128008.0, 128009.0)},
])
def test_fixed_config_tuple_members_must_be_integers(override):
    with pytest.raises(ContractError):
        RuntimeConfig(**override)


def raw_meta(request_id='r1', length=129):
    return CacheMetadata(request_id, length, 'cuda:0', CacheFormat.RAW)


def packed_meta(length=129, group_size=8):
    return CacheMetadata('r1', length, 'cuda:0', CacheFormat.PACKED, group_size)


def test_cache_handoff_preserves_length_and_releases_raw_reference():
    raw, packed = object(), object()
    handle = CacheHandle(raw_meta(), raw)
    assert handle.require(CacheFormat.RAW) is raw
    handle.commit_packed(packed_meta(), packed)
    assert handle.require(CacheFormat.PACKED) is packed
    assert handle.metadata.length == 129
    with pytest.raises(CacheStateError):
        handle.require(CacheFormat.RAW)
    with pytest.raises(CacheStateError):
        handle.commit_packed(packed_meta(), object())
    handle.commit_decode(packed_meta(length=130), object())
    assert handle.metadata.length == 130
    handle.release()
    handle.release()  # cleanup is idempotent
    with pytest.raises(CacheStateError):
        handle.require(CacheFormat.PACKED)


@pytest.mark.parametrize('meta', [packed_meta(length=130), CacheMetadata('other', 129, 'cuda:0', CacheFormat.PACKED, 8)])
def test_failed_handoff_keeps_original_cache(meta):
    original = object()
    handle = CacheHandle(raw_meta(), original)
    with pytest.raises(CacheStateError):
        handle.commit_packed(meta, object())
    assert handle.require(CacheFormat.RAW) is original


@pytest.mark.parametrize('kind', ['missing_payload', 'wrong_device', 'raw_metadata'])
def test_incomplete_handoff_keeps_metadata_and_payload_together(kind):
    metadata, payload = raw_meta(), object()
    handle = CacheHandle(metadata, payload)
    replacement = {
        'missing_payload': (packed_meta(), None),
        'wrong_device': (CacheMetadata('r1', 129, 'cuda:1', CacheFormat.PACKED, 8), object()),
        'raw_metadata': (raw_meta(), object()),
    }[kind]
    with pytest.raises(CacheStateError):
        handle.commit_packed(*replacement)
    assert handle.metadata is metadata
    assert handle.require(CacheFormat.RAW) is payload


def test_decode_cannot_skip_a_position_or_change_group():
    handle = CacheHandle(raw_meta(), object())
    metadata, payload = packed_meta(), object()
    handle.commit_packed(metadata, payload)
    for meta in [packed_meta(length=131), packed_meta(length=130, group_size=16)]:
        with pytest.raises(CacheStateError):
            handle.commit_decode(meta, object())
        assert handle.metadata is metadata
        assert handle.require(CacheFormat.PACKED) is payload


def test_released_cache_rejects_every_live_operation():
    handle = CacheHandle(raw_meta(), object())
    handle.release()
    assert handle.format is CacheFormat.RELEASED
    for operation in [lambda: handle.metadata,
                      lambda: handle.require(CacheFormat.RAW),
                      lambda: handle.commit_packed(packed_meta(), object()),
                      lambda: handle.commit_decode(packed_meta(length=130), object())]:
        with pytest.raises(CacheStateError):
            operation()


def test_token_result_requires_live_cache():
    handle = CacheHandle(raw_meta(), object())
    result = TokenResult(128009, 0, handle)
    assert result.cache is handle
    handle.release()
    with pytest.raises(ContractError, match='live'):
        TokenResult(100, 1, handle)


@pytest.mark.parametrize('token, ready_ns', [(True, 0), (128256, 0), (1, True), (1, -1), (1, 1.0)])
def test_token_result_rejects_invalid_ids_and_timestamps(token, ready_ns):
    with pytest.raises(ContractError):
        TokenResult(token, ready_ns, CacheHandle(raw_meta(), object()))


def test_stop_contract_includes_first_token_and_eos():
    assert generation_stop((100,), 1) is StopReason.LENGTH
    assert generation_stop((128009,), 512) is StopReason.EOS
    assert generation_stop((128009,), 1) is StopReason.EOS
    assert generation_stop((100,), 2) is None


@pytest.mark.parametrize('eos', [128001, 128008, 128009])
def test_every_fixed_eos_stops_and_cannot_be_followed(eos):
    assert generation_stop((100, eos), 2) is StopReason.EOS
    with pytest.raises(ContractError, match='after EOS'):
        generation_stop((eos, 100), 2)


def test_generation_cannot_exceed_output_limit():
    with pytest.raises(ContractError, match='exceed Nmax'):
        generation_stop((100, 101), 1)


def test_units_are_explicit_and_integer_inputs_are_strict():
    assert ns_to_seconds(1_000_000_000) == 1.0
    assert mhz_to_hz(306) == 306_000_000
    for value in [True, 1.0, -1]:
        with pytest.raises(ContractError):
            ns_to_seconds(value)
    for value in [True, 1.0, 0, -1]:
        with pytest.raises(ContractError):
            mhz_to_hz(value)


def test_core_import_has_no_torch_dependency():
    result = subprocess.run([sys.executable, '-c',
        "import sys; import kvolt_runtime.contracts; import kvolt_runtime.backend.protocol; assert 'torch' not in sys.modules"],
        capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')})
    assert result.returncode == 0, result.stderr
