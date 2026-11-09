# KVolt

Jetson AGX에서 Llama-3.1-8B-Instruct의 Prefill·Decode 설정을 선택하는 실험용 추론 런타임이다.
PC는 GPU 주파수를, DC는 K2V2 그룹 크기와 GPU 주파수를 선택한다.

## 먼저 찾을 곳

- **현재 LUT:** [고정 선택 파일](data/luts/current/selection.json) · [원본·단위·SHA 안내](data/luts/README.md)
- **개발 실행의 최신 기본값:** [Online LUT 피드백](docs/online-lut.md)
- **문서 탐색:** [주제별 목차](docs/README.md)
- **보조 도구:** [기능과 실행 조건](tools/README.md)
- **로컬 결과:** [최신 검증과 과거 기록](artifacts/README.md)
- **작업 규칙:** [AGENTS.md](AGENTS.md)

## 폴더 구조

```text
KVolt/
├── src/kvolt_runtime/   # CLI, 정책, Backend, 실행·장비·계측 코드
├── data/luts/          # 버전별 원본 LUT와 current 선택 파일
├── tests/             # CPU 회귀 검사, fixtures, 장비 의존 검사
├── docs/              # 정책·실행 계약과 검증 기록
├── tools/             # 진단·감사·장비 검증 보조 CLI
├── artifacts/         # 로컬 실험 결과·로그·보존 archive
├── pyproject.toml     # Python 패키지, CLI, 선택 의존성 설정
└── AGENTS.md          # 연구 근거·원자료 보존·작업 절차
```

로컬 workspace의 현재 개발·실행 저장소는 `KVolt/`다.
형제 [KVolt-support/](../KVolt-support/)에는 `KVolt-eval-pilot`,
`KVolt-lut-source`, `KVolt-validation-20261108`,
`KVolt-validation-current-20261109`를 원래 이름 그대로 모아 보존한다.
이 보조 checkout·동결 검증 기록은 현재 실행 소스를 대신하지 않는다.
CPU 테스트는 명시적 `KVOLT_LUT_BUNDLE`이 없을 때
`../KVolt-support/KVolt-lut-source/data/luts/agx2-repeat5-20261103`을 찾고,
기존 `../KVolt-lut-source/` 배치도 지원한다. 이 탐색은 테스트에만 적용된다.

## 기본 LUT와 실행 모드

`kvolt run`과 `kvolt diagnose`는 LUT 옵션과 `--mock`를 생략하면
`data/luts/current/selection.json`을 읽는다. 현재 예측은
[`agx2-runtime-repeat5`](data/luts/agx2-runtime-repeat5/), FP16 목표는
[`agx2-reference-repeat10`](data/luts/agx2-reference-repeat10/)이다.
source checkout에서는 작업 디렉터리와 무관하게 같은 선택을 읽고 SHA를 검증한다.
`latest/`는 이전 AGX1 재현용이며 현재 기본값이 아니다.
버전별 LUT·manifest·참조 원본은 그대로 보존한다.

TPOT 통과 상한은 **원 FP16 Reference × 1.20**이다. 전역 설정
[`TPOT_BUFFER_RATIO = 0.20`](src/kvolt_runtime/config.py)을 바꾸고 새 프로세스를
시작하면 모든 길이의 DC 예산·후보 선택과 실제 TPOT 판정에 같은 비율이 적용된다.
`0.0`은 원 기준, `0.10`은 +10%, `0.30`은 +30%다. TTFT와 원본 LUT/Reference는
보존한다. 버퍼를 늘리는 것은 허용 지연 변경이며 연산 속도 향상이 아니다.
소스/비율이 바뀌면 기존 qualification을 재사용할 수 없다.

개발 `run`은 Offline LUT를 초기값으로 하는 별도 Online 보정 표를 기본 사용한다.
정상 실측을 EMA(기본 α=0.1)로 반영하고 다음 요청부터 사용한다.
Quality·FP16 목표·원본 LUT는 바꾸지 않는다. `--no-online-lut`로 동결 실행을 선택한다.
정식 `--mode performance`는 동결 LUT와 일치하는 M4 qualification을 요구한다.
개발 Online 검증이 정식 적격성이나 SLO·에너지 개선을 보장하지 않는다.
[Online 계약](docs/online-lut.md)과 [qualification 조건](docs/runtime-qualification.md)을 확인한다.

## CPU에서 확인하기

저장소 루트에서 Python 3.10 이상으로 실행한다. 다음 명령은 모델·tokenizer·GPU를
사용하지 않으며, 현재 LUT로 PC를 예측하고 Nmax=1에 따라 DC를 생략한다.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/diagnose_policy.py \
  --length 1024 --max-new-tokens 1 --pc-elapsed-ns 0
```

DC까지 예측하려면 `python3 -m pip install '.[policy]'`로 고정된 선택 의존성을
설치한 뒤 실행한다. 결과는 가정한 경과 시간에 대한 예측이며 실측이 아니다.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/diagnose_policy.py \
  --length 1536 --max-new-tokens 512 \
  --pc-elapsed-ns 1000000 --dc-elapsed-ns 2000000
```

전체 옵션은 `python3 tools/diagnose_policy.py --help`로 확인한다.
실제 GPU 추론에는 고정 모델 snapshot, 준비된 장비·전력 프로필, KIVI 및 native 의존성이 필요하다.
[실행 CLI](docs/runtime-run-cli.md)와 [장비 계약](docs/runtime-device.md)을 함께 읽는다.
기존 문서의 명시 LUT 예제·개발 동결 설명보다 위의 current 선택 및 Online 기본값을 우선한다.

## 검증 기록 읽기

[2026-11-09 종합 보고](docs/full-validation-current.md)는 source `79dbe41`의
frozen M4 및 별도 개발 Online 검증 기록이다. 정상 TPOT 목표 미달 관측도 포함한다.
폴더 안내와 중복 파일 정리는 새로운 GPU 실측이나 정식 적격성을 추가하지 않는다.
`artifacts/`의 기존 결과는 로컬 보존 자료이며 새 clone에는 함께 내려오지 않는다.
