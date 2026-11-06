"""Profiles have no timing defaults and make no M4 qualification claim."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

from ..config import require_int
from ..errors import ContractError


@dataclass(frozen=True)
class DeviceProfile:
    schema_version: int
    name: str
    qualification: str
    evidence: str
    gpu_path: Path
    identity_path: Path
    identity_expected: str
    boot_id_path: Path
    cpu_policy_paths: tuple[Path, ...]
    emc_rate_path: Path
    emc_lock_path: Path
    emc_cap_path: Path
    emc_max_rate_path: Path
    emc_halt_path: Path
    maxn_command: Path
    maxn_mode_id: int
    tpc_path: Path
    tpc_expected: str
    fan_path: Path
    fan_policy_path: Path
    systemctl_command: Path
    timeout_ns: int
    poll_ns: int
    consecutive_matches: int
    query_timeout_ns: int

    def __post_init__(self):
        require_int(self.schema_version, 'profile schema_version', 1, 1)
        for name in ('name', 'evidence', 'identity_expected', 'tpc_expected'):
            value = getattr(self, name)
            if type(value) is not str or not value.strip() or value != value.strip() or '\x00' in value:
                raise ContractError(f'profile {name} must be a nonempty stripped string')
            try:
                value.encode('utf-8')
            except UnicodeError as error:
                raise ContractError(f'profile {name} must be UTF-8') from error
        if self.qualification not in ('provisional', 'device_verified'):
            raise ContractError('profile qualification must be provisional or device_verified; neither grants M4 eligibility')
        if len(self.identity_expected) != 64 or any(c not in '0123456789abcdef' for c in self.identity_expected):
            raise ContractError('identity_expected must be the SHA-256 of the stripped physical serial')
        for name in ('timeout_ns', 'poll_ns', 'query_timeout_ns'):
            require_int(getattr(self, name), name, 1, 86_400_000_000_000)
        require_int(self.consecutive_matches, 'consecutive_matches', 1, 1_000_000)
        require_int(self.maxn_mode_id, 'maxn_mode_id', 0, 1024)
        if self.poll_ns > self.timeout_ns or (self.consecutive_matches - 1) * self.poll_ns > self.timeout_ns:
            raise ContractError('verification matches must fit within explicit timeout')
        if type(self.cpu_policy_paths) is not tuple or len(self.cpu_policy_paths) != 3 or len(set(self.cpu_policy_paths)) != 3:
            raise ContractError('exactly three distinct CPU policies are required')
        paths = [getattr(self, field.name) for field in fields(self) if field.name.endswith('_path')]
        paths += [self.maxn_command, self.systemctl_command, *self.cpu_policy_paths]
        if any(not isinstance(path, Path) or not path.is_absolute() or '..' in path.parts for path in paths):
            raise ContractError('profile paths must be absolute, without parent traversal')
        if self.maxn_command not in (Path('/usr/sbin/nvpmodel'), Path('/usr/bin/nvpmodel')):
            raise ContractError('MAXN query must use an explicit system nvpmodel executable')
        if self.systemctl_command not in (Path('/usr/bin/systemctl'), Path('/bin/systemctl')):
            raise ContractError('fan service query must use an explicit system systemctl executable')

    @classmethod
    def from_dict(cls, data):
        names = {field.name for field in fields(cls)}
        if type(data) is not dict or set(data) != names:
            raise ContractError('device profile must specify exactly all documented fields')
        converted = dict(data)
        try:
            for name in names:
                if name.endswith('_path') or name in ('maxn_command', 'systemctl_command'):
                    if type(converted[name]) is not str:
                        raise TypeError(name)
                    converted[name] = Path(converted[name])
            if type(data['cpu_policy_paths']) is not list or any(type(p) is not str for p in data['cpu_policy_paths']):
                raise TypeError('cpu_policy_paths')
            converted['cpu_policy_paths'] = tuple(Path(path) for path in data['cpu_policy_paths'])
            return cls(**converted)
        except (TypeError, ValueError) as error:
            if isinstance(error, ContractError):
                raise
            raise ContractError('invalid device profile') from error

    @classmethod
    def from_json(cls, path):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ContractError(f'duplicate profile field: {key}')
                result[key] = value
            return result
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (OSError, ValueError, UnicodeError) as error:
            if isinstance(error, ContractError):
                raise
            raise ContractError('cannot read finite UTF-8 device profile JSON') from error
        return cls.from_dict(data)

    def as_dict(self):
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif key == 'cpu_policy_paths':
                result[key] = [str(path) for path in value]
        return result
