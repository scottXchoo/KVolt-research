"""Versioned TPOT allowance identity, separate from immutable Reference data."""
from ..config import MISSING, resolve_tpot_buffer_ratio
from ..errors import ContractError

TPOT_POLICY_VERSION = 'reference-buffer-v1'


def tpot_policy(buffer_ratio=MISSING):
    return {'version': TPOT_POLICY_VERSION,
            'ratio': resolve_tpot_buffer_ratio(buffer_ratio)}


def validate_tpot_policy(value):
    if (type(value) is not dict or set(value) != {'version', 'ratio'}
            or value['version'] != TPOT_POLICY_VERSION):
        raise ContractError('missing or invalid TPOT buffer policy identity')
    return tpot_policy(value['ratio'])
