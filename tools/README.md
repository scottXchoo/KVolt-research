# 보조 도구 안내

[프로젝트 안내](../README.md) · [문서 목차](../docs/README.md)

저장소 루트에서 `python3 tools/도구이름.py --help`로 실제 옵션을 확인한다.
아래 10개 도구는 코드와 함께 유지하며 결과는 입력 source·LUT·환경 조건과 함께 읽는다.

## CPU 진단·기존 기록 분석

| 도구 | 기능·실행 조건 |
| --- | --- |
| [diagnose_policy.py](diagnose_policy.py) | 현재 기본 LUT 또는 명시 source로 PC/DC 예측. 가정한 시간·첫 토큰을 사용하며 추론하지 않는다. 활성 DC는 고정 `policy` 의존성이 필요하다. |
| [diagnose_tpot.py](diagnose_tpot.py) | 기존 `session.json`의 시간 경계를 읽어 TPOT를 분해한다. 표준 라이브러리만 사용하며 장비·LUT를 변경하지 않는다. |
| [latest_tpot_audit.py](latest_tpot_audit.py) | 명시 compact 예측 LUT와 FP16 reference의 CPU 감사. PC/DC 실행에는 고정 `policy` 의존성이 필요하며 `--arithmetic-only`는 정책 실행을 생략한다. |

현재 기본 LUT로 PC만 확인하는 완전한 예제:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/diagnose_policy.py \
  --length 1024 --max-new-tokens 1 --pc-elapsed-ns 0
```

자세한 해석은 [CPU 진단](../docs/policy-diagnostics.md),
[TPOT 분석](../docs/tpot-diagnostics.md), [LUT 안내](../data/luts/README.md)를 따른다.

## 장비·모델이 필요한 개발 및 적격성 도구

| 도구 | 기능·실행 조건 |
| --- | --- |
| [probe_device.py](probe_device.py) | `discover`·`probe`는 장비 경로·준비 상태를 읽는다. 명시 `qualify`는 GPU 전환과 복원을 수행한다. 실제 장비 프로필·시스템 경로가 필요하다. |
| [qualify_backend.py](qualify_backend.py) | M1 Backend smoke·full·boundaries correctness 검사. GPU, 고정 모델 snapshot, KIVI/native 의존성과 기존 공통 장비 잠금이 필요하다. |
| [qualify_first_eos.py](qualify_first_eos.py) | CPU 입력 동결 후 제한된 자연 첫 토큰 EOS 진단을 실행한다. 실제 생성 단계는 GPU·모델이 필요하다. |
| [qualify_runtime.py](qualify_runtime.py) | M3 장비·전력·실행 smoke 검증. 명시 프로필과 장비 준비가 필요하며 M4 정식 승인이 아니다. |
| [qualify_m4.py](qualify_m4.py) | M4 계획 동결·실행·감사. `prepare`는 CPU 및 opt-in correctness 검사를 수행하고, `execute`는 명시 관리자 실행과 장비 제어가 필요하다. 정식 평가 자체를 실행하지 않는다. |
| [profile_tpot.py](profile_tpot.py) | 기존 동결 plan과 SHA를 받아 개발 지연 원인을 계측한다. GPU·실행 환경이 필요하며 profiling 오버헤드가 있어 정식 qualification 근거로 쓰지 않는다. |
| [probe_fixed_decode.py](probe_fixed_decode.py) | 정상 PC/DC 계산 뒤 명시 fixed Decode 개입을 비교하는 개발 실험. 동결 plan·LUT·reference·GPU가 필요하며 단일 관측으로 속도·품질 동등성을 입증하지 않는다. |

장비 도구의 실제 실행 전 [장비·잠금 계약](../docs/runtime-device.md),
[Backend 조건](../docs/backend-qualification.md), [M4 절차](../docs/runtime-qualification.md)를 확인한다.
입력 파일·동결 SHA·snapshot·프로필·출력 경로를 갖춘 문서의 해당 절차를 따른다.
기존 출력과 측정 자료를 덮어쓰거나 실패 결과를 삭제하지 않는다.
