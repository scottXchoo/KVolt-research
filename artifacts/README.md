# 로컬 결과와 보존 자료

[프로젝트 안내](../README.md) · [문서 목차](../docs/README.md) · [도구 안내](../tools/README.md)

이 폴더는 실험 결과·로그·archive·당시 source snapshot을 보존한다.
Git은 이 안내 파일만 추적하므로 **새 clone에는 아래 로컬 하위 폴더가 없을 수 있다.**
기존 파일·폴더 이름과 바이트를 유지하며, 날짜·이름만으로 최신 적격성을 판단하지 않는다.

## 현재 확인을 시작할 곳

- [full-validation-20261109/](full-validation-20261109/): source `79dbe41`의 frozen M4·별도 개발 Online 실측 및 독립 감사 자료.
  [종합 보고](../docs/full-validation-current.md)에서 결과와 개별 증거를 먼저 읽는다.
  TPOT 목표 미달 관측을 포함하며 Online의 정식 적격성이나 개선을 뜻하지 않는다.
- 실제 실행의 기본 LUT 원본은 이 폴더가 아닌 [data/luts/current/selection.json](../data/luts/current/selection.json)에서 선택한다.

## LUT 관련 이전 검증 기록

| 폴더 | 보존 내용 |
| --- | --- |
| [agx2-runtime-lut/](agx2-runtime-lut/) | AGX2 runtime LUT 연결, 출처·측정·독립 검토와 재현 기록 |
| [reference-repeat5/](reference-repeat5/) | 이전 FP16 repeat5 reference 연결·비교·검토 기록. 현재 repeat10 목표와 구분한다. |
| [latest-lut/](latest-lut/) | 이전 AGX1 compact LUT, profiling·fixed Decode 실험과 source snapshot |
| [combined-lut-check-2650c96/](combined-lut-check-2650c96/) | 당시 source `2650c96`의 결합 LUT·Backend 후속 검사 |

## 단계별 개발·검증 이력

| 폴더 | 보존 내용 |
| --- | --- |
| [m1/](m1/) | Backend·Prefill·Decode correctness, 동결 cohort와 첫 EOS 검사 |
| [m2/](m2/) | CPU 정책 통합 패키지, sealed LUT·검토 환경 의존성 |
| [m2-c1/](m2-c1/) | 정책 공통 계약의 컴파일·pytest·독립 검사 로그 |
| [m2-c2/](m2-c2/) | Prefill 정책·binary64 손상 검사와 독립 검증 |
| [m2-c3/](m2-c3/) | Decode 정책·정규화·대상 장비 검사와 SHA 대조 |
| [m2-c4/](m2-c4/) | CPU 진단 CLI의 독립 경계 검사·pytest·source SHA |
| [m3/](m3/) | Runtime 통합, cohort·전력 프로필·실행 결과와 보충 검증 |
| [m3-h1/](m3-h1/) | 장비 제어의 독립 검토 결과 |
| [m3-i1/](m3-i1/) | 입력·출력 구현 근거와 독립 검사 |
| [m3-t1/](m3-t1/) | 시간·전력 계측의 source 감사, 실패·수정·보충 검토 |
| [m4/](m4/) | 이전 M4 qualification과 TPOT profiling·독립 검토·archive |

과거 성공·실패·수정 이력은 각 source와 당시 입력 조건에 묶인 기록이다.
기존 archive 안의 복사본·캐시도 해당 증거의 일부로 보존한다.
새 실험 결과를 추가할 때에는 기존 디렉터리와 파일을 덮어쓰지 않는다.
