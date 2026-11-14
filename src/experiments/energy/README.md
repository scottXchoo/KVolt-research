# 실험 구간 에너지 계측

AGX Orin 64GB에서 임의 Python 작업의 에너지를 선택적으로 수집하는 스켈레톤이다.
모델·데이터셋·정책과 독립적이며 Python 3.10 이상과 표준 라이브러리를 사용한다.
[EnergyMeter](../../kvolt_runtime/telemetry/experiment.py)는 기존
[INA3221 collector](../../kvolt_runtime/telemetry/power.py)를 재사용한다.

## 실험에 붙이기

패키지를 설치했거나 `PYTHONPATH=src`인 환경에서 다음 구조를 복사한다.
`with` 안에는 측정하려는 작업만 넣는다. import·모델 로드·warmup을 측정할지는
실험 목적에 따라 명시적으로 정하고 모든 비교군에 같은 경계를 적용한다.

```python
from kvolt_runtime.telemetry import EnergyMeter


def run_experiment():
    # 측정할 작업을 여기에 넣는다.
    pass


def main():
    with EnergyMeter(enabled=True) as meter:
        run_experiment()
    meter.save('artifacts/my-experiment/energy')
    print(meter.result)


if __name__ == '__main__':
    main()
```

`enabled=False`로 바꾸면 같은 작업을 실행하면서 센서 탐색·collector·동기화 callback을
호출하지 않는다. 이때 `energy_j`, `average_power_w`, `duration_s`는 `None`,
`reason`은 `not_collected`다. 미계측을 0 J로 해석하지 않는다.
`multiprocessing`의 `spawn`을 사용하므로 실행 진입점의 `if __name__ == '__main__'`을 유지한다.

비동기 GPU 작업은 완료 동기화를 호출자가 제공해야 한다. PyTorch 실험이라면 기존
`import torch` 아래에서 `EnergyMeter(synchronize=torch.cuda.synchronize)`를 사용한다.
callback은 계측 시작 직전과 종료 경계 직전에 호출된다. 특정 device나 stream을 쓰면
해당 작업을 기다리는 callback을 전달한다. 기본 callback은 없으며 이 모듈은 Torch를
설치하거나 import하지 않는다. 동기화 비용이 관측에 영향을 줄 수 있으므로 설정을 기록한다.

수동 경계도 지원한다. 새 측정마다 새 meter를 만들고 예외가 가능한 작업에는 `with`를 권장한다.

```python
meter = EnergyMeter()
meter.start()
run_experiment()
result = meter.stop()  # 반복 stop은 같은 결과를 반환한다.
meter.save('artifacts/my-experiment/manual-energy')
```

## 실행 가능한 데모

저장소 루트에서 실행한다. 기본 데모는 sleep만 수행하며 추론·성능 개선 실험이 아니다.
계측을 끈 경로는 macOS 등 INA3221이 없는 호스트에서도 실행할 수 있다.

```sh
python3 src/experiments/energy/measure.py \
  --no-energy --duration-s 0.1 --output-dir /tmp/kvolt-energy-disabled
```

Jetson에서는 INA3221의 이름과 rail label로 경로를 읽기 전용 탐색한다.
`hwmon1` 같은 번호를 고정하지 않는다. 물리 채널 1~3만 사용하고 summation 채널은 제외한다.
중복·누락 rail, label 읽기 오류, current/voltage 파일 누락은 탐색에서 실패한다.
전류·전압 숫자와 읽기 권한은 시간 제한이 있는 collector 준비 단계에서 검증한다.

```sh
python3 src/experiments/energy/measure.py \
  --duration-s 1 --output-dir artifacts/energy/sleep-001
```

명시적 프로필은 `--power-profile /absolute/path/power-profile.json`로 지정한다.
형식은 [기존 계측 계약](../../../docs/runtime-measurement.md)의 `Explicit collector profile`을 따른다.
자동 탐색 기본값은 polling 20 ms, 허용 최대 gap 200 ms, 준비 5 s·우측 coverage 2 s·종료 2 s다.
기존 문서 예시의 최대 gap 100 ms와 구분하며, 이 값은 개발 설정이다. polling 주기가 센서의
응답 대역폭을 뜻하지 않는다. 기존 runtime 프로필이나 formal qualification을 바꾸지 않는다.
데모 종료 코드는 미계측 또는 유효 측정이면 0, 수집 결과가 무효이면 2다.
준비·동기화·파일 저장 예외는 Python 오류로 전달된다.

## 결과와 해석

`meter.result`는 `valid`, `reason`, `duration_s`, `energy_j`, `average_power_w`,
`rails`, `start_ns`, `end_ns`, `completed`, `scope`, `collector`, `errors`를 포함한다.
시각은 host `time.monotonic_ns`, 에너지는 J, 평균 전력은 W다.
평균 전력은 측정 구간 에너지를 구간 시간으로 나눈 값이다.
`meter.receipt`는 원시 samples와 worker JSONL을 포함하는 collector 영수증이다.

에너지는 `VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0` 독립 3레일의 합이다.
GPU 단독 에너지나 벽전력이 아니다. `VDD_IN` 같은 중복 공급 rail은 더하지 않는다.
mV·mA 원값으로 W를 계산하고 signed 값을 보존하며 idle subtraction이나 0 clipping을 하지 않는다.
각 rail을 별도로 경계 보간·사다리꼴 적분한 뒤 합산한다.
모든 rail에서 시작 전·종료 후 표본과 허용 gap이 필요하다. 오류 표본을 버리거나
외삽·0 대체로 결과를 채우지 않는다. 센서 응답·평균화 지연은 검증하지 않았으며
`sensor_response_validated=False`를 보존한다. 짧은 작업의 에너지를 정밀하게 분해할 수 있다는
보장은 없다.

`save()`는 종료한 meter에 대해 다음 파일을 생성한다. 같은 이름의 기존 파일은 덮어쓰지 않는다.

- `energy.json`: 구간 결과·collector 설정·오류와 저장 근거.
- `power-samples.csv`: rail별 전류·전압·전력·읽기 시각·오류.
- `power-trace.jsonl`: worker가 작성한 원시 trace.

원시 JSONL과 CSV는 서로 다른 바이트 표현이다. 각 파일의 SHA-256을 혼용하지 않는다.
모델·입력·seed·반복·장비 제어 조건 등 실험 출처는 호출하는 실험에서 함께 보존한다.
이 모듈은 GPU 주파수·CPU·팬·센서 설정을 변경하지 않는다.

센서·collector 실패는 `valid=False`, 에너지 `None`과 오류로 남는다. 작업이 예외로 끝나면
원래 예외를 보존하고 `reason=workload_failed`, `completed=False`, 에너지 `None`을 기록한다.
예외 실행의 원자료도 저장하려면 `meter`를 미리 만들고 호출자의 예외 처리에서 `.result`를
확인한 뒤 `.save()`를 호출한다. 시작·종료 동기화 실패도 정리 후 원래 예외를 전달한다.

CPU 검증은 분석 적분, fake sysfs와 실제 spawn lifecycle, disabled·예외·저장 경계를 확인한다.
이번 추가의 실제 AGX 센서·GPU 계측 정확도·계측 오버헤드는 검증하지 않았다.
기존 LUT·Reference·Online·PC/DC·정식 qualification을 이 API의 검증 결과로 승계하지 않는다.
센서 단위·label의 공식 근거는
[NVIDIA Jetson Linux r36.4.4 전력 관리 문서](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)다.
