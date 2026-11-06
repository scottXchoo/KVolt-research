"""Read-only, spawn-isolated INA3221 collection with bounded lifecycle waits."""
import csv
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import tempfile
import time

from ..errors import ContractError
from .metrics import RAILS, SCOPE

MAX_PROFILE_NS = 86_400_000_000_000


def _positive_ns(value, name):
    if type(value) is not int or not 0 < value <= MAX_PROFILE_NS:
        raise ContractError(name+' must be integer ns in [1, 86400000000000]')


def _utf8(value, name):
    if type(value) is not str or not value:
        raise ContractError(name+' must be a nonempty UTF-8 string')
    try:
        value.encode('utf-8', errors='strict')
    except UnicodeError as error:
        raise ContractError(name+' must be a valid UTF-8 string') from error


def _unique(pairs):
    result={}
    for k,v in pairs:
        if k in result:
            raise ContractError('duplicate power profile key: '+k)
        result[k]=v
    return result


@dataclass(frozen=True)
class RailPaths:
    label_path: str
    current_path: str
    voltage_path: str

    def __post_init__(self):
        for field in ('label_path','current_path','voltage_path'):
            value=getattr(self,field)
            _utf8(value,field)
            if '\x00' in value or not Path(value).is_absolute():
                raise ContractError(field+' must be an absolute path')


@dataclass(frozen=True, eq=False)
class _FrozenRails(Mapping):
    """Tuple-backed mapping: immutable values, no mutable backing dictionary.

    Module-level dataclass remains pickleable with multiprocessing spawn.
    """
    _items: tuple

    def __getitem__(self,key):
        for label,value in self._items:
            if label == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (label for label,_ in self._items)

    def __len__(self):
        return len(self._items)


@dataclass(frozen=True)
class PowerProfile:
    rails: Mapping
    interval_ns: int
    max_gap_ns: int
    ready_timeout_ns: int
    coverage_timeout_ns: int
    join_timeout_ns: int
    poll_interval_ns: int = 1_000_000
    name: str = 'explicit-development'
    evidence: str = 'Provisional development profile; M4 qualification required'

    def __post_init__(self):
        if not isinstance(self.rails,Mapping) or set(self.rails) != set(RAILS):
            raise ContractError('power profile requires exactly '+', '.join(RAILS))
        rails={}
        for label in RAILS:
            value=self.rails[label]
            try:
                rails[label]=value if isinstance(value,RailPaths) else RailPaths(**value)
            except (TypeError, ValueError) as error:
                raise ContractError('invalid rail paths: '+label) from error
        paths=[v for rail in rails.values() for v in asdict(rail).values()]
        if len(set(paths)) != len(paths):
            raise ContractError('rail paths must be independent and unique')
        for name in ('interval_ns','max_gap_ns','ready_timeout_ns','coverage_timeout_ns',
                     'join_timeout_ns','poll_interval_ns'):
            _positive_ns(getattr(self,name),name)
        if self.max_gap_ns < self.interval_ns:
            raise ContractError('max_gap_ns must be at least interval_ns')
        _utf8(self.name,'name')
        _utf8(self.evidence,'evidence')
        # Neither caller-owned dictionaries nor the public mapping can mutate
        # the validated profile later used by the spawned worker.
        object.__setattr__(self,'rails',_FrozenRails(tuple(rails.items())))

    @classmethod
    def from_json(cls, path):
        try:
            value=json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=_unique,
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if type(value) is not dict:
                raise ValueError('profile must be an object')
            return cls(**value)
        except (OSError,TypeError,ValueError) as error:
            raise ContractError('invalid power profile: '+str(error)) from error

    def to_dict(self):
        result={field.name:getattr(self,field.name) for field in fields(self)
                if field.name != 'rails'}
        result['rails']={label:asdict(paths) for label,paths in self.rails.items()}
        return result


def read_sample(rail, paths, *, clock=time.monotonic_ns):
    """One signed mV/mA read window; preserve raw values and read failures."""
    start=clock(); current=voltage=power=None; error=None
    try:
        label=Path(paths.label_path).read_text().strip()
        if label != rail:
            raise ValueError('rail label mismatch: '+label)
        voltage=int(Path(paths.voltage_path).read_text().strip())
        current=int(Path(paths.current_path).read_text().strip())
        power=voltage*current/1e6
        if not math.isfinite(power):
            raise ValueError('nonfinite sensor power')
    except (OSError,ValueError,OverflowError) as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
    end=clock()
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        error={'type':'ClockError','message':'invalid monotonic read window'}
    return dict(rail=rail,timestamp_ns=(start+end)//2,read_start_ns=start,read_end_ns=end,
                current_ma=current,voltage_mv=voltage,power_w=power,error=error)


def _worker(profile, trace_path, stop_event, ready_event, update_event, latest, skipped, clock):
    """No parent GIL, CUDA inheritance, or per-sample parent IPC/file writes."""
    try:
        deadline=time.monotonic_ns()
        with open(trace_path,'w',encoding='utf-8',buffering=1) as stream:
            while not stop_event.is_set():
                for index,rail in enumerate(RAILS):
                    row=read_sample(rail,profile.rails[rail],clock=clock)
                    stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
                    if row['error'] is None:
                        latest[index]=row['timestamp_ns']
                if all(t >= 0 for t in latest):
                    ready_event.set()
                update_event.set()
                deadline += profile.interval_ns
                now=time.monotonic_ns()
                if now > deadline:
                    missed=(now-deadline)//profile.interval_ns+1
                    skipped.value += missed
                    deadline += missed*profile.interval_ns
                stop_event.wait(max(0,deadline-time.monotonic_ns())/1e9)
    except BaseException as error:
        # Durable error survives worker exit; parent reads only after joining.
        with open(trace_path+'.error','w',encoding='utf-8') as stream:
            json.dump({'code':'COLLECTOR_WORKER_ERROR','type':type(error).__name__,
                       'message':str(error)},stream)
        update_event.set()


def samples_csv(samples):
    """Serialize after finish; hash this exact UTF-8 payload when storing CSV."""
    output=io.StringIO(newline='')
    fields=('rail','timestamp_ns','read_start_ns','read_end_ns','current_ma','voltage_mv','power_w','error')
    writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n')
    writer.writeheader()
    for row in samples:
        value={k:row.get(k) for k in fields}
        if value['error'] is not None:
            value['error']=json.dumps(value['error'],ensure_ascii=False,sort_keys=True,allow_nan=False)
        writer.writerow(value)
    return output.getvalue()


class INAPowerCollector:
    """Single-attempt collector. start precedes t0; finish follows final token.

    All waits use real host monotonic deadlines even with an injected test clock.
    Start failures return ``ready=False`` and errors; runner must reject readiness.
    Stop never raises for sensor/worker failures and is safe before start/repeatedly.
    """
    def __init__(self, profile, *, clock=time.monotonic_ns):
        if not isinstance(profile,PowerProfile):
            raise ContractError('collector requires PowerProfile')
        self.profile=profile
        self.clock=clock
        self._context=multiprocessing.get_context('spawn')
        self._stop_event=self._context.Event()
        self._ready_event=self._context.Event()
        self._update_event=self._context.Event()
        self._latest=self._context.Array('q',[-1]*len(RAILS))
        self._skipped=self._context.Value('q',0)
        self._process=None
        self._temp=None
        self._trace_path=None
        self._started=False
        self._ready=False
        self._errors=[]
        self._receipt=None
        self._end_ns=None

    def _error(self,code,message):
        self._errors.append({'code':code,'stage':'power','message':message})

    def start(self):
        if self._started or self._receipt is not None:
            raise ContractError('power collector cannot start twice or after stop')
        self._started=True
        self._temp=tempfile.TemporaryDirectory(prefix='kvolt-power-')
        self._trace_path=str(Path(self._temp.name)/'raw.jsonl')
        self._process=self._context.Process(target=_worker,name='kvolt-ina3221',daemon=True,
            args=(self.profile,self._trace_path,self._stop_event,self._ready_event,
                  self._update_event,self._latest,self._skipped,self.clock))
        started=time.monotonic_ns()
        try:
            self._process.start()
            deadline=started+self.profile.ready_timeout_ns
            while time.monotonic_ns() < deadline and self._process.is_alive():
                if self._ready_event.wait(min(self.profile.poll_interval_ns,deadline-time.monotonic_ns())/1e9):
                    now=self.clock()
                    self._ready=all(0 <= now-t <= self.profile.max_gap_ns for t in self._latest)
                    if self._ready:
                        break
            if not self._ready:
                self._error('COLLECTOR_NOT_READY','No fresh valid sample on every rail before readiness deadline')
                self.stop()
        except Exception as error:
            self._error('COLLECTOR_START_ERROR',str(error))
            self.stop()
        return dict(ready=self._ready,status='ready' if self._ready else 'error',
                    errors=list(self._errors),profile=self.profile.to_dict(),
                    clock='time.monotonic_ns' if self.clock is time.monotonic_ns else 'injected-test-clock',
                    start_method='spawn',ready_ns=self.clock() if self._ready else None)

    def finish(self,end_ns):
        if self._receipt is not None:
            return self._receipt
        if type(end_ns) is not int or end_ns < 0:
            self._error('COLLECTOR_INVALID_END','end_ns must be nonnegative integer ns')
            return self.stop()
        self._end_ns=end_ns
        deadline=time.monotonic_ns()+self.profile.coverage_timeout_ns
        covered=False
        if self._ready:
            while time.monotonic_ns() < deadline:
                if all(t > end_ns for t in self._latest):
                    covered=True; break
                if not self._process.is_alive():
                    break
                self._update_event.clear()
                self._update_event.wait(max(0,min(self.profile.poll_interval_ns,deadline-time.monotonic_ns()))/1e9)
        if not covered:
            self._error('COLLECTOR_RIGHT_COVERAGE','No valid sample strictly after end_ns on every rail')
        return self.stop()

    def stop(self):
        if self._receipt is not None:
            return self._receipt
        self._stop_event.set()
        process=self._process
        if process is not None and process.pid is not None:
            process.join(self.profile.join_timeout_ns/1e9)
            if process.is_alive():
                self._error('COLLECTOR_JOIN_TIMEOUT','Collector required termination; trace may be incomplete')
                process.terminate(); process.join(self.profile.join_timeout_ns/1e9)
            if process.is_alive():
                process.kill(); process.join(self.profile.join_timeout_ns/1e9)
            if process.is_alive():
                self._error('COLLECTOR_STOP_FAILED','Collector survived bounded termination')
            elif process.exitcode != 0:
                self._error('COLLECTOR_EXIT_ERROR',str(process.exitcode))
        samples=[]; raw=b''
        if self._trace_path is not None:
            path=Path(self._trace_path)
            try:
                raw=path.read_bytes()
                for index,line in enumerate(raw.splitlines()):
                    try:
                        row=json.loads(line)
                        samples.append(row)
                        if row.get('error'):
                            self._error('COLLECTOR_READ_ERROR',json.dumps({'row':index,**row['error']}))
                    except (ValueError,TypeError) as error:
                        self._error('COLLECTOR_TRACE_ERROR',str(error))
                error_path=Path(self._trace_path+'.error')
                if error_path.exists():
                    self._errors.append(json.loads(error_path.read_text()))
            except OSError as error:
                self._error('COLLECTOR_TRACE_ERROR',str(error))
        self._receipt=dict(status='error' if self._errors else ('stopped' if self._started else 'not_started'),
            ready=self._ready,samples=samples,errors=list(self._errors),profile=self.profile.to_dict(),
            scope=SCOPE,clock='time.monotonic_ns' if self.clock is time.monotonic_ns else 'injected-test-clock',
            timestamp_unit='ns',power_unit='W',start_method='spawn',
            sensor_response_validated=False,skipped_deadlines=self._skipped.value,
            end_ns=self._end_ns,trace_sha256=hashlib.sha256(raw).hexdigest(),
            trace_hash_format='worker JSONL bytes',raw_trace_jsonl=raw.decode('utf-8',errors='replace'))
        if self._temp is not None and (process is None or not process.is_alive()):
            self._temp.cleanup()
        return self._receipt
