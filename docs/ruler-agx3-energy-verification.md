# AGX3 RULER 생성·에너지 연결 검증 — 2026-11-14

AGX3 `agx32@100.94.58.28`에서 대표 RULER 4과제 40문항의 실제 KIVI 생성과
문항별 3레일 에너지 저장을 완료했다. 생성 40/40, 에너지 유효 40/40,
독립 원자료 재적분 감사가 통과했다. 전체 요청의 공식 substring recall 평균은
95.83%였다. **40문항 모두 각 과제의 출력 상한에 도달했으며, 자연 EOS 종료는 관측하지 않았다.**

실행 소스는 `24086f58d9a08752336459eb143c9cbb9647937d`다. 대표 묶음 40행을
모두 실행한 결과이며, 전체 RULER 13과제 평가나 동결된 정식 평가 결과를 뜻하지 않는다.
초기 SSH alias `agx3@100.122.216.93`는 다른 장치였으며, 사용자가 지정한 위 AGX3에서
별도 검증했다. 다른 장치의 실행 중 작업은 중단하거나 변경하지 않았다.

## 실제 결과

각 과제의 첫 선택 문항을 원래 출력 상한으로 한 번씩 warmup한 뒤 본 측정 문항을 실행했다.
warmup 4회와 본 측정 40회를 합쳐 실제 생성 호출은 44회였다. 아래 에너지·시간·점수 집계에서
warmup은 제외했다. `L`과 `N`은 실제 입력·출력 token ID에서 확인한 길이다.

| 과제 | 문항 | 평균 점수 | 평균 에너지 J | 평균 계측 시간 s | 실제 L 범위 | 실제 N | 상한 종료 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `niah_single_1` | 10 | 100.00% | 764.9356 | 19.5251 | 3363–3969 | 128 | 10 |
| `niah_multikey_2` | 10 | 100.00% | 757.3106 | 19.3024 | 3537–3963 | 128 | 10 |
| `vt` | 10 | 100.00% | 334.0938 | 7.4341 | 4049–4058 | 30 | 10 |
| `fwe` | 10 | 83.33% | 417.6861 | 9.8638 | 3688–4019 | 50 | 10 |

- 저장된 40문항 에너지 합: **22,740.260472858583 J**.
- 40개 계측 구간의 시간 합: **561.253140601 s**.
- 시간으로 가중한 평균 전력, `에너지 합 / 계측 시간 합`: **40.51694115868626 W**.
- launcher 전체 경과시간: **655.126539 s**. 소스·모델 준비, warmup, collector 준비·종료,
  저장·정리 등을 포함하므로 위 계측 시간 합과 구분한다.

두 시간의 차이 `93.8734 s`에는 pin·모델 파일 읽기/로딩, warmup 4회,
사전 tokenization·hash·context 검증, collector 준비·우측 표본 대기, I/O·정리가 포함된다.
이 차이를 단계별 경과시간으로 분해하여 측정하지는 않았다. 경계에서 각 작업을 제외한 계약과
warmup 완료 횟수는 검증했으며, 전체 경과시간으로 에너지 합을 나누어 작업 전체 평균 전력으로 해석하지 않는다.

독립 감사의 합은 `22,740.260472858587 J`였다. 서로 다른 합산 순서의 부동소수점 오차 범위에서
저장 결과와 일치했다. 과제별 출력 상한은 기존 `128/128/30/50`을 유지했다.
상한 도달을 숨기기 위한 stop 설정 변경은 하지 않았다.

## 계측 경계와 해석

RULER 실행기는 기존 `EnergyMeter`로 `model.kivi_llama.generate` 전체 호출을 감쌌다.
내부 tokenization, 입력 H2D 전송, `model.generate`, 새 continuation text decode가 포함된다.
시작 직전과 종료 경계 직전에 `torch.cuda.synchronize(model.device)`로 완료를 기다렸다.
모델 로딩, 사전 입력 tokenization·hash·context 검증, warmup, 채점, 결과 직렬화와 파일 저장,
collector 준비·우측 표본 대기·정리는 이 구간 밖이다.

전력 범위는 INA3221의 `VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0` 독립 3레일 합이다.
GPU 단독 에너지나 벽전력이 아니다. signed mV·mA를 그대로 사용하고 idle subtraction이나
음수의 0 clipping을 하지 않았다. 각 rail의 경계를 따로 선형 보간한 뒤 사다리꼴 적분하여 합했다.
개발 프로필은 polling 20 ms, 허용 최대 gap 200 ms다.
`sensor_response_validated=false`를 유지하며 센서 물리 응답·평균화 지연·벽전력 정확성은 검증하지 않았다.

검증 harness는 계측 안에서 입력·출력 GPU tensor 참조와 작은 generation flag,
quantize/matmul 정수 카운터만 보존했다. 모든 계측이 끝난 뒤 CPU token 복사·hash·decode 검산을
수행했다. 이 참조 수집·카운터의 계측 오버헤드는 관측값에 포함되며, 별도로 빼지 않았다.
실제 output ID와 full sequence를 보존하여 prefix 제외, 새 continuation, 실제 N,
greedy·beam1·inference mode, EOS/상한과 quantized 경로를 확인했다.

이 결과로 TTFT/TPOT, Prefill/Decode 분리 에너지, J/token, SLO, 에너지 개선 또는
기존 LUT·Reference·formal qualification을 주장하지 않는다. 실행 조건은 고정 G32 한 구성이다.

## 장치·소스·실행 조건

공개 장치 정보는 NVIDIA Jetson AGX Orin Developer Kit, aarch64,
kernel `6.8.12-1021-tegra`였다. 기존 CPU `schedutil`과 729600–2201600 kHz 범위,
GPU `nvhost_podgov`와 306000000–1300500000 Hz 범위가 실행 전후 동일했다.
CPU·GPU 주파수, EMC, 팬 설정을 쓰지 않았다. 이 조건은 주파수 고정 정식 평가 조건이 아니다.
관측 가능한 CPU/GPU 제어값만 전후 대조했으며, EMC debug 제어값은 snapshot에 포함되지 않아
EMC readback 검증을 완료했다고 해석하지 않는다.

기존 venv의 Python 3.10.12, Torch 2.5 NVIDIA 빌드, Transformers 4.43.1,
FlashAttention 2.7.4.post1, Triton 3.5.0, Accelerate 0.33.0을 사용했다.
기존 portable CUDA12.6 라이브러리를 명령환경에 명시했고 설치나 시스템 CUDA 변경은 하지 않았다.
`validate_sources(..., native_profile='agx2')`가 기존 model/source/native SHA pin과 실제 바이트 일치를
검증했다. 여기서 `agx2`는 이미 등록된 native binary pin 집합의 이름이며 물리 장치는 AGX3다.
다른 장치의 grant를 승계하지 않았다.

모델은 로컬 `Llama-3.1-8B-Instruct` snapshot
`0e9e39f249a16976918f6564b8830bc894c89659`였다. 291 parameter tensor 모두 FP16/cuda:0,
모든 module eval, 32개 `LlamaFlashAttention_KIVI` layer와 K2V2/G32/residual128을 확인했다.
plain-text prompt에 chat template이나 별도 정답 completion을 추가하지 않았다.
원본 RULER `cl100k_base` 길이와 실제 Llama 입력 길이는 별도로 보존했다.

기존 `/run/lock/kvolt-p3-hardware.lock` inode의 nonblocking 독점 lock을
torch import·모델 로드 전에 획득했다. 소스는 원격 기존 checkout을 수정하지 않고 커밋에서 추출한
격리 snapshot 157파일을 사용했다. 실행 전후 manifest의 byte 수·SHA와 파일 inventory가 일치했다.
정리 단계에서 test-only `_cuda_clearCublasWorkspaces()`로 8,519,680 bytes workspace를 해제한 뒤
CUDA allocated/reserved `0/0`, 잔여 collector child·검증 process `0`, 공통 lock 재획득 가능을 확인했다.
workspace 정리는 생산 모델 helper 변경이 아니다.

## 검증 근거와 재현

- E1 CPU 회귀 238개 통과. 실패·null·warmup 제외·저장 오류·기본 미계측·기존 telemetry 계약을 검증했다.
- 실제 `spawn` collector와 fake sysfs를 연결한 로컬 2요청 fixture에서 6 W를 확인했다.
  이 fixture는 GPU·물리 센서 실측이 아니다.
- Sol medium 사전검증 16개 통과: compile, spawn-safe import, source manifest,
  signed/ramp·서로 다른 rail 시각의 경계 적분, gap·오류·CSV/trace 변조 탐지, busy 종료 코드 75.
- AGX3 실제 실행 exit0. Sol medium 독립 감사도 exit0으로 40문항과 warmup 4회의 매핑,
  입력·출력 ID/EOS/상한, pin·해시, 공식 prediction 후처리를 독립 구현한 substring recall을 대조했다.
- 84,390개 원 표본의 JSONL/CSV 숫자 일치, mV×mA, 모든 read error 0,
  rail별 시작 전·종료 후 coverage와 단조 시각, 모든 gap을 확인했다.
  최대 gap은 22.547043 ms로 200 ms 이하였다. 프로젝트 적분기를 import하지 않는 stdlib 감사가
  rail별 경계 knot를 별도로 구성해 다시 적분하여 J/W/시간과 3레일 합을 검산했다.
  token decode 자체는 harness의 모델 tokenizer 결과와 runner continuation을 대조한 범위이며,
  stdlib 감사가 tokenizer를 독립 재구현한 것은 아니다.

원자료와 ignored 검증 스크립트는 `artifacts/ruler-energy-agx3-20261114/`에 보존했다.

- [실행·모델·실제 44호출 기록](../artifacts/ruler-energy-agx3-20261114/remote-evidence/attempt1/gpu-validation.json)
- [40행 결과·점수·계측 집계](../artifacts/ruler-energy-agx3-20261114/remote-evidence/attempt1/results/summary.json)
- [문항별 결과](../artifacts/ruler-energy-agx3-20261114/remote-evidence/attempt1/results/predictions.jsonl),
  [warmup](../artifacts/ruler-energy-agx3-20261114/remote-evidence/attempt1/results/warmup.jsonl)
- [독립 전체 감사](../artifacts/ruler-energy-agx3-20261114/independent-audit.json),
  [독립 감사 로그](../artifacts/ruler-energy-agx3-20261114/independent-audit.log),
  [독립 번들·집계 교차검증](../artifacts/ruler-energy-agx3-20261114/final-audit-review.json),
  [종료 후 확인](../artifacts/ruler-energy-agx3-20261114/remote-evidence/postflight.json)
- [사전검증 16개](../artifacts/ruler-energy-agx3-20261114/preflight-validation.json),
  [로컬 spawn fixture](../artifacts/ruler-energy-agx3-20261114/local-meter-integration.json)
- [소스 manifest](../artifacts/ruler-energy-agx3-20261114/source-manifest.json),
  [실행 스크립트](../artifacts/ruler-energy-agx3-20261114/verify-energy.py),
  [독립 감사 스크립트](../artifacts/ruler-energy-agx3-20261114/independent-audit.py)

원격 수집 묶음은 136파일·5,680,039 bytes이며 SHA-256은
`e5596a8d0c3a0809270a99624d584a794dbfe34c3bb68db7cb9e82b1be911cba`다.
내부 135개 원자료 index 항목과 실제 파일 byte 수·SHA를 대조했다.
소스 archive SHA-256은 `5f32cf757a60b140a9e73aae7f2f4e7b6f7a24e1a26f1ad7993dd12e06a23cb0`,
manifest SHA-256은 `bc86b3557dbe54b97909ccb2f47880974e1637a918b28098a548ae680303a365`다.

동일 조건의 runner 옵션은 다음과 같다. **runner 자체는 하드웨어 lock을 획득하지 않는다.**
아래 호출은 이 검증의 `verify-energy.py`처럼 기존 공통 lock을 외부에서 독점하고,
고정 소스·native/model pin·CUDA 환경을 확인한 실행자 안에서 수행해야 한다.

```sh
PYTHONPATH=src /mnt/kvolt-data/kvolt/.venv/bin/python -B \
  -m experiments.quality.RULER.KIVI_Llama \
  --model-name /mnt/kvolt-data/kvolt/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --measure-energy --warmup-per-task 1 \
  --output-dir /absolute/new/ruler-energy-run
```

실제 검증은 독점 lock 안에서 같은 옵션의 `runner.run(..., measure_energy=True,
warmup_per_task=1)`을 호출했으며, 생성 중 capture의 파일 I/O·CPU token 복사·검산을 피했다.
독립 감사 재현은 실제 장비나 GPU 재실행 없이 보존한 immutable source snapshot을 사용한다.

```sh
python3 artifacts/ruler-energy-agx3-20261114/independent-audit.py \
  --attempt artifacts/ruler-energy-agx3-20261114/remote-evidence/attempt1 \
  --source-root artifacts/ruler-energy-agx3-20261114/source-snapshot \
  --manifest artifacts/ruler-energy-agx3-20261114/source-manifest.json \
  --expected-sha 24086f58d9a08752336459eb143c9cbb9647937d \
  --output /absolute/new/independent-audit.json
```
