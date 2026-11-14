# 문서 목차

[프로젝트 안내](../README.md) · [LUT 원본·현재 선택](../data/luts/README.md) ·
[도구 안내](../tools/README.md) · [로컬 결과 안내](../artifacts/README.md)

최신 연구 설계는 [Method V10: Proposal v3.0 변경 정리](method-v10.md)다.
DC 중심 구조·Unknown→Accepted·비율 기반 G 선택 제안과 PDF 내부 충돌을 정리했다.
코드 반영은 수행하지 않았으며 아래 정책·실행 문서는 현재 구현과 과거 검증의 계약이다.

최신 기본 선택은 `data/luts/current/selection.json`, 개발 실행 기본값은 Online 보정이다.
TPOT 허용 상한은 원 Reference의 기본1.20배이며 `config.py`의 `TPOT_BUFFER_RATIO`로
변경한다. 원본 표·TTFT는 보존하고 소스/비율별 qualification을 구분한다.
먼저 LUT 안내와 [Online 계약](online-lut.md)을 읽는다. 기존 단계별 문서에는 이전
명시 LUT·동결 개발 예제가 남아 있으므로 해당 시점의 범위를 함께 확인한다.

## LUT와 제어 정책

| 문서 | 찾을 내용 |
| --- | --- |
| [policy-contracts.md](policy-contracts.md) | CPU 정책 인터페이스, 값·동점·오류 계약 |
| [policy-lut.md](policy-lut.md) | LUT 로더, 보간, 원본 출처와 버전별 adapter |
| [policy-prefill.md](policy-prefill.md) | PC 시간 예산, 에너지 선택과 지연 fallback |
| [policy-decode.md](policy-decode.md) | Kneedle, DC 후보 집합, 선택·복구 규칙 |
| [policy-diagnostics.md](policy-diagnostics.md) | CPU 진단의 가정, 출력과 오류 의미 |
| [online-lut.md](online-lut.md) | 개발 Online EMA, 상태 재사용과 동결 모드 |

## Backend와 실행

| 문서 | 찾을 내용 |
| --- | --- |
| [runtime-contracts.md](runtime-contracts.md) | 초기 Runtime·Backend 경계와 공통 계약 |
| [backend-prefill.md](backend-prefill.md) | 단일 상주 모델의 FP16 Prefill과 raw KV 인계 |
| [runtime-execution.md](runtime-execution.md) | 순차 실행, phase 전환과 책임 경계 |
| [runtime-input-output.md](runtime-input-output.md) | 입력 사전 검사, 세션·요청 결과 보존 |
| [runtime-run-cli.md](runtime-run-cli.md) | 실제 실행 옵션, 준비·출력·종료 코드 |
| [runtime-device.md](runtime-device.md) | 장비 소유, GPU 적용·검증·복원과 프로필 |
| [runtime-measurement.md](runtime-measurement.md) | t0/t1/tN, 메모리 진단과 INA3221 에너지 |
| [독립 실험 에너지 계측](../src/experiments/energy/README.md) | 선택적으로 붙이는 EnergyMeter·sleep 데모·원시 trace 저장 |
| [탈부착 실험 환경 제어](experiment-control.md) | 온도·스로틀링·클럭·전력 모드·간섭·swap·warmup의 독립 제어와 기록 |
| [KIVI LM-Eval 연결](lm-eval.md) | 고정 lm-eval 공식 GSM8K/RULER task·실제 stop·원 토큰·선택 에너지 계측 |

## 적격성·검증·TPOT 분석

| 문서 | 찾을 내용 |
| --- | --- |
| [backend-qualification.md](backend-qualification.md) | M1 correctness, AGX native build, 첫 EOS 검사 |
| [runtime-qualification.md](runtime-qualification.md) | M4 동결 계획, 검증 범위와 performance 진입 조건 |
| [qualification-evidence.md](qualification-evidence.md) | qualification report 형식, 출처 결속과 grant 검사 |
| [full-validation-current.md](full-validation-current.md) | source `79dbe41`의 2026-11-09 실측·독립 감사 기록 |
| [tpot-diagnostics.md](tpot-diagnostics.md) | 기록된 TPOT 분해, 개발 profiling과 해석 한계 |
| [latest-lut-feasibility.md](latest-lut-feasibility.md) | 이전 AGX1 compact LUT의 예측 가능성과 TPOT 분석 |

적격성 통과와 SLO 달성은 별개다. 결과를 재사용할 때 source·LUT·장비·모드 결속과
보고서의 미검증 범위를 확인한다. 최신 종합 보고도 명시된 source의 기록이다.
