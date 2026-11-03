# AGX2 Eval Pilot

이 파일럿은 기존 FP16/KIVI 고정 캐시 경로의 실행 가능성, 품질, 시간과 센서 에너지를 확인한다. 새 KVolt PC/DC Runtime의 성능 평가가 아니다. 기존 KIVI의 최초 양자화·packing은 Prefill 내부에서 발생하며 TTFT에 한 번 포함된다.

## 동결한 범위

- AGX2, Llama-3.1-8B-Instruct FP16 가중치, batch 1.
- MQ·MV 각 5개 의미 계열, 과제마다 기본 3·고난도 2. 같은 10개 계열을 정확히 1,024·4,096·8,192토큰으로 구성한다.
- FP16 KV 및 K2V2 G=8/16/32/64/128, residual 128. 30입력 × 6설정 × 1회 = 정식 180요청.
- Greedy, 자연 EOS, 첫 토큰 포함 출력 상한 128. N=1의 Decode TPOT·에너지는 null.
- MAXN, CPU 2,201,600kHz 고정, EMC 3,199,000,000Hz 고정, GPU nvhost_podgov 306–1,122MHz, 팬 최대.
- 새 입력 seed 20261103. 파일럿 계열은 본 평가에서 제외한다. CORE·PELM 및 고정 주파수 실험은 포함하지 않는다.
- 입력 블록 순서와 블록 내 6설정 순서는 seed 20261103으로 동결한다. 저품질·SLO 위반은 결과에 남긴다.

## 실행과 중단

`python -m scripts.eval_pilot.device --inputs INPUTS --output OUTPUT --snapshot SNAPSHOT --user agx31`을 root로 실행한다. OUTPUT은 미리 agx31 소유로 생성한다. GPU 추론은 agx31로 실행하며, root는 장비 설정·모니터·원상 복원만 담당한다.

기본 실행은 가장 긴 입력 하나의 6설정 smoke를 통과한 후 정식 180요청으로 이어진다. Smoke 결과는 `smoke/`에 보관하며 정식 수에 합치지 않는다. 각 실제 입력·설정은 동일 생성 정책으로 warmup한 후 측정한다. Warmup/JIT/모델 로드 시간은 정식 요청 지연과 분리해 기록한다.

요청 시작 온도는 기존 프로토콜의 35–55°C를 검증한다. Warmup 전에는 상한만 확인한다. 최대 900초 대기 후에도 충족하지 못하면 임의 가열 없이 중단한다. 기술 오류·전력 적분 무효도 원본 결과를 보존하고 중단한다. 입력·소스·프로토콜이 같은 경우만 재개할 수 있으며 성공한 정식 요청은 재실행하지 않는다.

동일 장비에서 다른 실험과 동시에 실행하지 않는다. 백그라운드 실행은 `systemd-run`을 사용한다. 정상 종료·예외·SIGINT·SIGTERM 시 기존 HardwareController가 CPU/GPU/EMC·팬·전원 모드 복원을 시도하고 `device-session-*/restoration.json`에 기록한다. 강제 전원 차단이나 SIGKILL에는 복원이 보장되지 않는다.

## 에너지와 산출물

20ms INA3221 로그에서 LUT와 같은 독립 3개 레일 `VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0`을 적분한다. VDD_IN 같은 중복 aggregate는 더하지 않는다. 이는 벽전력이나 보드 전체 전력 측정값이 아니다. 레일별 시각으로 t0/t1/tN을 보간하며 외삽·idle 전력 차감은 하지 않는다. tN 뒤 표본을 확인한 후 로거를 중단한다.

- `cohort/`: 동결 입력·생성 manifest·계열 정의.
- `environment-preflight.json`: 기존 설치·모델·커널·센서 사전 점검.
- `diagnostics/`: 제어 조건을 충족하지 않은 기능 점검. 정식 시간·에너지 결과에서 제외.
- `manifest.json`, `sessions/`: 입력·소스·스케줄 해시와 실행별 장비 근거.
- `requests.jsonl`, `requests/`: append-only 시도 이력, 출력, 실제 토큰 수, 채점, 원시 전력과 적분 결과.
- `warmup/`, `lifecycle/`: 계측 외 준비 비용 및 JIT 근거.
- `device-session-*/`: 설정 전 스냅샷, root 모니터, worker 로그, 복원 근거.
- `analysis/summary.json`, `analysis/cells.csv`: 완료 여부와 길이·설정별 18셀 요약.

분석은 `python -m scripts.eval_pilot.analyze --output OUTPUT`으로 재생성한다. 180요청은 독립 표본 180개가 아니라, 짝지은 10개 의미 계열의 측정이다. 반복 측정이 없으므로 관측 최소·최대는 입력 차이를 포함하며 신뢰구간이나 실행 변동 추정치로 해석하지 않는다. 실제 출력 길이를 보존하여 요청 에너지와 Decode J/token을 함께 검토한다.
