# 탈부착 실험 환경 제어

`src/experiment_control/`은 추론 프레임워크·KVolt Runtime에 의존하지 않는
Python 3.10+ 표준 라이브러리 패키지다. 설정·센서·정책을 조합해 실험 코드를
감싸고, scope를 제거하면 기존 코드로 돌아간다. CUDA·torch를 import하거나
실험의 에너지·TTFT·TPOT 측정 경계를 바꾸지 않는다.

## 연결과 제거

설치된 checkout에서는 `pip install -e .`로 공개 API와 `experiment-control`
명령을 설치할 수 있다. 설치 없이 사용할 때는 `PYTHONPATH=src`를 지정한다.

```python
from experiment_control import environment_from_profile

env = environment_from_profile(
    "my-environment.json",
    output_dir="artifacts/my-experiment/environment",  # 새 디렉터리
    warmup=warmup_model,
    synchronize=synchronize_device,
)
with env:
    result = experiment()  # 기존 함수 그대로
    env.checkpoint()      # 선택: raise 정책의 협력적 중단 지점
print(env.report)
```

`env.run(experiment, *args, **kwargs)`도 지원한다. 반복 호출용 데코레이터는
매 호출마다 새 scope와 새 output directory를 만드는 factory를 받는다.

```python
from experiment_control import controlled

@controlled(make_fresh_environment)
def experiment():
    ...
```

제거할 때 `with env` 또는 데코레이터를 빼고 `experiment()`를 직접 호출한다.
실험 함수·모델·데이터셋·기존 CLI를 수정할 필요가 없다. KVolt에 종속되지 않는
`contracts.py`/`guard.py`가 공통 계층이고, Linux 제어·센서는 교체 가능한 adapter다.
사용자 정의 Control은 `name/capture/apply/restore`, Probe는 `name/sample()->Reading`
계약만 구현하면 된다.

프로파일 factory는 목표 설정을 `report.metadata.profile`에 자동 보존한다. 직접
`Environment(controls=..., probes=..., rules=..., metadata=...)`를 구성할 때는
실험 ID·제어 목표·코드 버전 등 재현에 필요한 정보를 metadata에 명시한다.

외부 프로그램도 소스 수정 없이 감쌀 수 있다.

```sh
PYTHONPATH=src python3 -m experiment_control \
  --profile examples/experiment-control/observe-linux.json \
  --output-dir artifacts/example-environment \
  -- python3 my_experiment.py
```

명령은 shell 없이 argv로 실행하고 foreground 종료를 기다린다. 실패 exit code를
보존하고 SIGINT/SIGTERM 및 정책 중단 시 생성한 process group을 종료·회수한 뒤
설정을 복원한다. CLI만 signal handler를 임시 소유하며 종료 후 원래 handler를
복원한다. Python API는 전역 signal handler를 바꾸지 않는다.

## 일곱 영역의 책임

| 영역 | 제어·준비 | 관찰·검사 |
| --- | --- | --- |
| 온도·냉각 | 선택적 fan PWM·팬 서비스 제어, warmup 이후 조건 대기 | thermal type 기반 온도, fan RPM, 상·하한, 시작 안정 구간의 최대 온도차 |
| 스로틀링 | 보호 기능을 유지하고 시작 조건·위반 정책 적용 | 명시한 thermal/OC/저전압 counter 또는 flag. 수집 불가하면 unavailable |
| 클럭 | CPU/GPU min/max·governor, EMC rate/lock/halt 가역 설정 | 보고 클럭·설정·source, 고정값 적용 확인 선택 |
| 전력 모드 | 기대 nvpmodel ID/이름 준비 검사 | 시작·종료 모드 대조. 자동 모드 변경·TPC 수정·재부팅 없음 |
| 간섭 | 공통 독점 lock, 선택적 process-leader CPU affinity, 시작 부하 gate | CPU 부하·load·process count·PSI·GPU/EMC explicit utilization |
| 메모리·swap | 선택적 swappiness, 메모리 여유/사용량 조건 | RAM·swap 사용, pswpin/pswpout 누적값과 실행 중 증가·감소 |
| 준비 상태 | 명시 횟수 warmup callback, 동기화, 열·부하 안정화 | 완료 횟수·준비 실패·관측 시각·timeout |

온도·부하·메모리 임계값은 실험 프로토콜에서 정한다. 패키지는 연구용 숫자를
임의로 확정하지 않는다. `observe-linux.json`은 관찰만 하며 적격성을 판정하지
않는다. `cpu-fixed.json`은 **MAXN 확인과 policy0 한 개**의 설정/복원 사용 예다.
전체 AGX2 CPU·EMC·fan 준비 또는 기존 Runtime qualification을 대신하지 않는다.

## JSON 프로파일과 단위

schema는 `experiment-control.profile.v1`이다. 최상위는 `controls`, `probes`,
`rules`, `session`, 선택 `lock_path`다. 알 수 없는 필드·중복 JSON 키·NaN/Inf는
거부한다. 파일 로딩과 객체 생성만으로 장비를 변경하지 않는다.

`controls[].kind`별 인자:

- `clock`: `name`, `path`, `minimum`, `maximum`, 선택 `governor`.
  GPU 기본 파일은 min_freq/max_freq/governor/cur_freq, 단위 Hz다.
  CPU는 min_file/max_file/governor_file/current_file을 scaling_* 파일로 지정하며
  **sysfs의 원 단위 kHz**를 사용한다. `verify_current=true`는 min=max에서만 허용하며
  timeout_s/poll_s/consecutive에 따라 적용 직후 보고값을 확인한다.
- `emc`: `rate_path`, `lock_path`, `halt_path`, `rate_hz`.
- `fan`: `pwm_path`, `pwm`(0..255), 선택 `service`, `systemctl`, `timeout_s`.
- `power_mode`: `expected_id`, 선택 `expected_name`, `command`, `timeout_s`.
- `swappiness`: `value`(0..200). swappiness=0도 swap 비활성화를 뜻하지 않는다.
- `affinity`: `cpus`(CPU ID 배열). process leader와 이후 그 설정을 상속하는 작업에
  적용한다. 이미 존재하는 다른 thread의 affinity는 그대로다.

`probes[].kind="linux"`에는 `name`, 선택 `sys_root`, `proc_root`, `gpu_path`
(devfreq 디렉터리), `emc_path`(rate 파일 또는 devfreq 디렉터리), `extra_metrics`를
지정한다. 센서 경로는 BSP마다 다를 수 있으므로 실제 보드에서 확인해야 한다.
주요 metric은 다음과 같다. 모든 수치 metric의 실제 파일 경로도 `source.<metric>`에 남긴다.

| Metric | 단위·의미 |
| --- | --- |
| `temperature.<type>.c` | °C. 중복 type은 .0/.1, literal dot는 escape된 alias |
| `clock.cpu.policy0.current_hz/min_hz/max_hz` | Hz. current는 cpuinfo_cur_freq 우선, scaling_cur_freq 대체 시 current_source 기록 |
| `clock.gpu.current_hz/min_hz/max_hz/governor` | Hz 또는 governor 문자열. driver 보고값 |
| `clock.emc.hz` | Hz |
| `fan.hwmonN.fan1.rpm`, `fan.hwmonN.pwm1.pwm` | RPM, PWM 값. 실제 존재 채널별 기록 |
| `cpu.utilization_pct`, `load.1m`, `process.count`, `cpu.online` | 시스템 CPU 사용률·load·프로세스 수·online 목록. 첫 CPU delta는 unavailable |
| `memory.total_bytes/available_bytes/swap_used_bytes` | byte |
| `vm.pswpin`, `vm.pswpout` | Linux 누적 swap page 수. byte로 표시하지 않음 |
| `pressure.memory.some.avg10` 등 | 지원되는 PSI 10초 평균 |
| `gpu.utilization_pct`, `emc.utilization_pct` | 장치별 source·scale을 extra_metrics로 지정해야 함 |

`extra_metrics`는 예를 들어 `{"throttle.oc_count": {"path": "/actual/oc_counter",
"kind": "int", "scale": 1}}` 형식이다. `kind`는 int/float/text, scale은 유한한
양수다. 이는 경로 형식 예시이며 해당 파일의 존재를 가정하지 않는다. 지정하지
않은 throttle source는 false 대신 unavailable이다. cooling-device 상태·클럭 하락만으로
thermal 또는 전기적 throttling의 원인을 단정하지 않는다.

각 rule은 `probe`, `metric`, 선택 `minimum`, `maximum`, `expected`,
`max_increase`, `max_span`, `required`(기본 true), `phases`(기본 ready/run)를 받는다.
`expected`는 값과 JSON scalar 타입이 모두 같아야 한다.

- 온도 허용 범위: minimum/maximum. 허용값은 파일럿 후 사용자가 지정한다.
- 열 안정화: max_span과 session.stable_for_s. 첫 관측 완료 후 최소 두 관측 사이의
  실제 경과 시간 동안 조건과 최대 변화폭을 확인한다. 단일 긴 파일 읽기는 안정화가 아니다.
- 유휴 부하 gate: phases=["ready"]로 설정한다. 실행 중 CPU 사용률에는 실험 자체도 포함된다.
- counter: max_increase는 workload 시작 직전 baseline 대비 증가 한도다. 직전 관측보다
  감소하는 reset도 위반이다. delta counter는 시작 전에 필수 가용성을 검사한다.
- required=false의 누락값은 해당 rule에서 건너뛰지만 센서 오류 기록에는 남긴다.

session에는 sample_interval_s(기본0.2), ready_timeout_s(기본0, 단일 준비 검사),
stable_for_s(기본0), warmup_runs(기본0), on_violation(record/raise, 기본record)를
지정한다. stable_for_s는 ready_timeout_s 이하여야 한다. 준비 실패는 항상 시작을
거부한다. record는 실행 중 위반을 보존하고, raise는 checkpoint/종료 시 예외로
보고한다. CLI는 polling 사이 checkpoint를 호출하여 foreground 작업을 중단한다.
실험 함수 내부로 비동기 예외를 주입하거나 요청 중 클럭을 재설정하지 않는다.

## Warmup·완료·잠금 계약

순서는 lock 확보 → 모든 원 상태 capture/파일 저장 → 명시 controls 적용·확인 →
warmup → synchronize → 열/부하 준비 gate → 시작 직전 baseline 재검사 → 관측하며
workload 실행 → synchronize → 최종 관측 → 역순 복원 → lock 해제다.
warmup_runs>0에는 warmup callback이 필요하다. callback은 모델·shape·커널 준비를
실험에 맞게 수행하고, 기존 KV 초기화도 호출자가 유지한다.

CLI의 `--warmup-command-json '["python3","warmup.py"]'`는 별도 프로세스다.
같은 모델 프로세스의 CUDA/JIT warmup을 대신하지 않는다. 그런 준비는 Python
callback이나 원래 프로그램 내부 warmup으로 수행한다.

컨트롤을 변경하려면 lock과 새 output_dir가 필수다. 기본 lock은 기존
`/run/lock/kvolt-p3-hardware.lock` inode이며 자동 생성·대기·대체하지 않는다.
별도 실험 시스템은 자신이 관리하는 기존 lock_path를 지정할 수 있다.
같은 lock을 따르는 다른 실험만 배제할 수 있으므로 비협력 작업의 간섭은 별도
센서·정책으로 확인한다. **kvolt run처럼 자체 lock을 잡는 CLI는 관찰 전용
프로파일로 감싸거나**, 동일 프로세스에서 이미 소유한 lock을 넘긴다.

```python
with existing_hardware_lock:
    env = environment_from_profile(profile, output_dir=output,
                                   lock=existing_hardware_lock, borrow_lock=True)
    with env:
        experiment()
```

borrowed lock은 해제하지 않는다. own lock은 nonblocking 획득 후 같은 PID만
해제하며 fork 자식은 부모 scope의 장비 복원이나 잠금 해제를 할 수 없다.
thermal trip/governor·critical shutdown·TPC·swapoff/swapon·cache drop·다른
프로세스 kill은 제공하지 않는다. 최대 팬 설정도 명시 opt-in이다.

비동기 GPU/future 작업은 synchronize callback에서 완료를 기다려야 한다.
CLI는 foreground 명령을 대상으로 하며 daemon/background/detached child는
지원하지 않는다. SIGKILL·전원 손실·프로세스 crash에는 자동 복원을 보장할 수
없다. 이때 snapshot을 보고 외부에서 복구·검증한 후 다음 실험을 시작해야 한다.

## 산출물·판정·검증 범위

- `snapshot.json`: 변경 전 원 상태. 모든 capture 성공 후 적용 전에 저장·fsync.
- `events.jsonl`: monotonic timestamp의 준비·warmup·센서값·오류·위반·복원 이력.
- `summary.json`: 준비, workload, 환경 rule 충족, monitor, 복원을 각각 판정.
  profile 설정, read 오류 수, 관측 수·최대 간격·읽기 시간도 기록한다.

output_dir는 새 디렉터리만 허용하며 실패 원자료를 덮어쓰지 않는다. 실행 중
위반은 최대1,000개 요약과 전체 발생 수를 보존하고, 상세 항목은 JSONL에 남긴다.
늦게 끝난 probe는 scope 종료 후 report에 값을 게시하지 못한다. 측정이 정상인데
복원만 실패하면 workload 성공과 restoration 실패를 따로 남기고 cleanup 오류를
반환한다. 원 workload 예외가 있으면 복원/기록 오류로 덮지 않는다.

센서 값은 주기적 software 관측이다. cur_freq를 읽어도 물리 클럭의 연속 유지나
순간 throttling 부재가 입증되지는 않는다. sampling 간격·오버헤드는 실제 보드에서
확인해야 한다. 원래의 요청 시간/에너지 결과를 환경 판정으로 자동 재분류하거나
성공한 실험만 재시도하지 않는다. 기존 LUT·Reference·SLO·formal qualification은
이 모듈을 연결했다고 승계되지 않는다.

구현 검증은 가상 sysfs/proc, 실제 로컬 foreground 프로세스·signal·fork와 기존
CPU 회귀 테스트를 사용한다. Linux/AGX에서 실제 팬·클럭·전력 모드·센서 경로 및
성능 오버헤드 검증은 별도다. 이 문서는 하드웨어 실측 성공을 주장하지 않는다.
