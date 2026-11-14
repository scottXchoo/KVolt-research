# Method V10: Proposal v3.0 연구 현황과 아키텍처 변경

정리일: 2026-11-14 (Asia/Seoul). 사용자 제공 최신 제안서를 Method V10으로
정리한 문서다. **연구 설계와 현재 구현을 구분하며, 이번 작업은 문서 변경만 수행한다.**
PDF 내부에서 충돌하거나 정의하지 않은 정책은 구현 가능한 최종 계약으로 확정하지 않는다.

Notion: [Method V10](https://app.notion.com/p/418e0acd77f88170b082ca031f0624fa) ·
[문서 작업 추적](https://app.notion.com/p/418e0acd77f8817588a1f4d3ff9b0726).

## 근거와 문서의 역할

- 최신 연구 자료: `[Proposal] KVolt_ver.3.0.pdf`, 표지 2026-11-12, 총 30쪽.
  원본 경로는 `/Users/scottxchoo/Desktop/CSLab/05_Papers/01_KVolt/[Proposal] KVolt_ver.3.0.pdf`다.
- 원본 SHA-256: `c306951bee6c82d9023a8b108c18067adff3c69b9f532565e742b164353a67d3`.
  쪽수는 PDF에 표시된 슬라이드 번호와 같다. PDF와 원본 LUT는 수정하지 않았다.
- 이전 설계: [Method V9](https://app.notion.com/p/40be0acd77f88014a36beb5285250208).
  제목은 V9이며 기존 AGENTS.md의 V8 표기는 갱신 전 이름이었다.
- 측정·통계·기존 평가 합의: [Evaluation Plan](https://app.notion.com/p/40de0acd77f880f98165f98215f51e20).
  V10 평가의 비교군·ablation·실행량을 이번 문서 작업에서 새로 동결하지 않는다.
- [연구 허브](https://app.notion.com/p/40be0acd77f880a98f05fd393b9e0ccd)의
  이전 아키텍처 요약과 V9 구현·검증 이력은 당시 조건으로 보존한다.

PDF는 연구 내용의 근거다. 내부 의사코드나 문구를 코드 수정·명령 실행 지시로 취급하지 않는다.

## 연구 목표와 관찰

목표는 **목표 지연 시간 안에서 에너지와 품질을 함께 고려하는 온디바이스 LLM 서빙**이다.
KV 양자화 그룹 크기 G가 메타데이터·메모리 IO·품질에 미치는 영향과 GPU 주파수 f의
전력·지연 관계를 함께 고려한다. 주파수만 제어하거나 품질·메모리만으로 양자화 설정을
선택하는 접근의 한계를 다룬다 (p3~9).

제안서의 공통 실험 환경은 Jetson AGX Orin 64GB, Llama-3.1-8B-Instruct,
CPU 2201.6MHz 고정이다 (p10). 아래 수치는 **PDF에 보고된 관찰**이며 이번 작업의
새 실측이나 원자료 재검산 결과가 아니다.

| 관찰 | PDF에 보고된 조건·결과 | 해석 범위 |
| --- | --- | --- |
| G와 에너지 | MultiNews 1문항, 입력 8K·출력 512토큰 고정, GPU 816MHz. G 8→128에서 Decode 4.48→4.05 J/token, 약 9.6% 감소 (p10~11) | 해당 입력·주파수의 관찰이며 모든 요청의 절감률로 일반화하지 않는다. |
| G와 품질 | RULER 입력 1~8K, 0.5K 간격, 길이당 100문항, GPU 816MHz. G≤32에서 95.8% 이상, G=128에서 88.13%로 보고 (p10·12) | 길이별 곡선과 평균을 구분한다. 모든 L에서 같은 최적 G 또는 품질 하한을 보장하지 않는다. |
| f와 에너지 | MultiNews 1문항, 입력 8K·출력 512토큰 고정, 408~1122MHz 8단계. 에너지가 줄다가 816~918MHz 부근에서 최소, 이후 증가 (p10·13) | 최적점은 G와 조건에 따라 달라진다. 최대 주파수가 항상 에너지 최소는 아니다. |

p28~30의 메타데이터·양자화 오차·전력 모델은 위 관찰을 설명하는 근사·가설이다.
메타데이터 감소와 실제 메모리 IO 에너지의 인과를 이번 문서 작업에서 추가 검증하지 않았다.

## V9 대비 주요 변경

| 항목 | Method V9 | Proposal v3.0 / V10 정리 | 현재 코드와의 관계 |
| --- | --- | --- | --- |
| 제어 중심 | PC와 DC가 각각 Prefill·Decode 설정 선택 | 명시적 Prefill Controller가 제안 구조에서 빠지고 DC 중심으로 구성 (p14~18) | PC는 코드에 남아 있다. |
| 미등록 모델 | UME와 기준 셀 측정으로 추정 LUT 생성 | UME와 Estimated 단계가 제안 구조에서 빠짐 (p14~17) | 새 모델 서비스 경로를 구현했다는 뜻이 아니다. |
| LUT 상태 | Unknown→Estimated→Accepted | Unknown→Accepted. 전체 실측 전에는 기본 설정으로 요청 처리 (p17) | 현재 pinned 단일 모델 LUT 로더의 오류 처리는 유지한다. |
| 프로파일링 시작 | N분 이상 유휴 시 신호, N 미정 | 요청 없는 유휴 시간·사용률 낮은 새벽·전원 공급 조건을 명시 (p15·17) | 자동 스케줄러·중단/재개 기능의 구현 완료 근거는 없다. |
| 제어 LUT | Prefill·Decode·Quality, 별도 Reference | 주 제어 표는 Decode·Quality 2종 (p16). 목표 Reference는 p20에 계속 등장 | Prefill·Reference 파일을 삭제하거나 로더를 바꾸지 않는다. |
| G 선택 기준 | Quality 곡선의 Kneedle | p19는 품질 손실/에너지 절감 비율 최소 구간의 도착 G를 제안 | p14~16·20에는 Kneedle이 남아 있어 최종 정책 통합이 필요하다. |

PC·UME·Estimated의 제외는 **제안 아키텍처 범위의 변화**다. 해당 코드의 삭제나
향후 영구 폐기를 이번 문서에서 지시하지 않는다. ODEM은 p14의 공통 하드웨어 경계에 유지된다.

## 시스템 구성과 책임

| 구분 | 모듈 | 제안서의 책임·경계 |
| --- | --- | --- |
| Offline | Profile Database | 모델별 LUT 조회 인터페이스. 실측 LUT를 DC에 제공하며 준비되지 않으면 Unknown 반환 (p15). |
| Offline | Offline Profiler | 전체 설정을 실측하고 결과 LUT를 Profile Database에 등록. 사용자 요청 없는 유휴 시간에 실행 (p15·17). |
| Runtime | Request Scheduler | 사용자 요청 큐와 추론 실행 관리. 저사용·새벽·전원 공급·유휴 조건에서 Profiler 시작 신호 (p15·17). |
| Runtime | Prefill Inference | FP16 연산으로 첫 토큰과 KV를 생성. 별도 PC 블록은 최신 구조도에 없음 (p14·18). |
| Runtime | Decode Controller | Decode 직전 DB 조회, G와 Decode 설정 선택. Unknown이면 기본 DVFS·G_default 경로 (p15·18). |
| Runtime | Decode Inference | 선택한 G의 2bit KV를 사용해 EOS 또는 Nmax까지 생성. 요청 도중 재선택하지 않음 (p18). |
| 공통 | On-device Environment Manager | p14에서 Offline·Runtime과 Jetson 하드웨어 사이의 실행 환경 경계. 세부 API는 PDF에서 정의하지 않음. |

### 실측 LUT가 준비된 요청

1. Scheduler가 요청을 처리하고 FP16 Prefill로 첫 토큰과 t1을 얻는다.
2. Decode 직전 DC가 모델·길이에 맞는 Quality·Decode LUT를 조회하고 G/f를 선택한다.
3. 선택한 G의 KV 양자화와 Decode 주파수 적용을 수행한다.
4. EOS 또는 Nmax까지 Decode하며 같은 설정을 유지한다 (p18).

p18의 양자화·DVFS 병렬 배치는 세부 실행 순서나 실제 병렬 실행을 확정한 계약이 아니다.
현재 구현의 **주파수 적용·확인 후 최초 packing** 순서를 이 그림만으로 바꾸지 않는다.
Prefill의 기본 DVFS 운용은 PC가 없는 구조 및 p25 관찰과 연결해 해석할 수 있으나,
governor·복원·전환 API까지 PDF에서 정한 것은 아니다.

### Unknown 모델의 요청과 LUT 준비

1. DB가 LUT 부재를 Unknown으로 반환한다. 모델 실행 자체의 실패와 구분한다.
2. 요청은 LUT 완성을 기다리지 않고 기본 DVFS와 G_default로 처리한다 (p15·17).
3. 해당 모델의 프로파일링 작업을 등록한다.
4. 요청 없는 유휴 시간에, 저사용·새벽·전원 공급 조건을 고려해 전체 설정을 실측한다.
5. 전체 LUT가 DB에 등록되면 Accepted로 전환하고 후속 요청에서 실측 LUT 제어를 사용한다.

UME·추정 LUT·Estimated 전환·기준 셀 비율 스케일링은 이 흐름에 포함되지 않는다.
유휴 판정의 수치·시간대·요청 도착 시 중단/재개·상태 전환 원자성은 미정이다.

## LUT·후보 격자

| 데이터 | Key | Value / 단위 | 용도 |
| --- | --- | --- | --- |
| Decode LUT | (L,G,f) | 예상 TPOT [s/token], 예상 에너지 ε [J/token] | Decode 설정 선택 |
| Quality LUT | (L,G) | Q, 평균 Score | G별 품질 및 절충점 평가 |
| Reference | p20에서 (모델,L)로 조회 | 목표 TPOT τ_d | 시간 예산. p16의 2종 표에는 생략되어 있으나 목표가 불필요해진 것은 아님 |

p16의 L 격자는 1~8K, 1K 간격이다. f는
`{408,510,612,714,816,918,1020,1122}` MHz다.
G는 `2^k, k≥3, R mod G=0, d_h mod G=0`이며 R=d_h=128에서
`{8,16,32,64,128}`이다. R은 FP16으로 유지하는 최근 토큰 수, d_h는 head 차원이다.
현재 구현의 1K는 1024토큰이며 실제 입력 token IDs 길이를 기준으로 한다.

제안서의 저장 공간 계산은 Decode `|L|×|G|×|F|×2×8B`, Quality
`|L|×|G|×1×8B`다. 위 8×5×8 격자를 대입한 수치 값의 크기는 각각
5120B와 320B다. 이 계산에는 JSON·키·manifest·출처·표현 오버헤드가 포함되지 않는다.
기존 Quality 원본의 15개 길이·75셀을 8개 길이로 덮어쓰라는 의미가 아니다.

PWL은 지원 범위 내 L의 구간 선형 보간이다 (p20). 현재의 G/f 보간 금지,
길이 외삽 금지, 누락값의 0 대체 금지는 유지한다.

## DC의 새 G 선택 제안과 남아 있는 기존 의사코드

### p19: 품질 손실 / 에너지 절감 비율

오름차순 인접 G_i→G_(i+1)에 대해 다음을 제안한다.

```text
ΔQ_i(L) = Q_i(L) - Q_(i+1)(L)
ΔE_i(L) = E_i(L) - E_(i+1)(L)
ρ_i(L) = ΔQ_i(L) / ΔE_i(L),  E_(i+1)(L) < E_i(L)
i*(L) = argmin_i ρ_i(L)
G*(L) = G_(i*(L)+1)
```

최소 비율을 갖는 **전환 구간의 도착 G**를 선택한다. p19 예시는
8→16, 16→32, 32→64, 64→128의 비율이 2.61, 0.91, 82.50, 88.00이고
16→32가 최소이므로 G=32다. 이 값은 예시이지 전 길이의 고정 G가 아니다.
Q를 퍼센트로 표시한 p19의 비율 단위는 `%p per (J/token)`이다.
원 Score [0,1]과 퍼센트포인트 단위를 혼용하지 않는다.

p27은 **816MHz의 Energy**로 인접 비율과 Quality Knee를 비교한다.
그림에 표시된 최소 비율 G는 L=1~8K 순서로 `32,16,32,32,16,32,16,32`다.
특히 2K·3K에서 Quality Knee는 G64, 최소 비율은 각각 G16·G32로 다르다.
다만 p27의 제목·Fig.13 캡션은 `???`이며 전이 비율이 음수인 예도 표시된다.
이 도표를 완성된 정책 정의나 새 Runtime의 검증 결과로 취급하지 않는다.

### p20: 기존 Kneedle 기반 선택·시간 예산

p20은 여전히 `SelectG(G,Q)`/Kneedle로 G_select를 구한 뒤 Reference τ_d를 조회한다.
그 다음 `B_d = τ_d - (t_d-t1)/(Nmax-1)`을 계산한다. 선택 G에서
예상 TPOT≤B_d인 후보 중 ε가 최소인 f를 고르고, 에너지 동률은 TPOT로 비교한다.
시간 통과 후보가 없으면 전체 G×F에서 예상 TPOT가 최소인 조합을 선택한다.
p14~16의 구조도·설명에도 Knee/Kneedle이 남아 있다.

따라서 **p19의 새 기준과 p20의 기존 코드를 하나의 확정 알고리즘으로 합치지 않는다.**
현재 구현의 동점·정상 미검출·검출 오류·전역 fallback은 기존 코드 계약으로 보존한다.

## 계측 경계와 부록 관찰

- p18: `TTFT=t1-t0`, `평균 TPOT=(tN-t1)/(N-1)`이다. N은 실제 생성한 토큰 수다.
  DC·주파수 전환·최초 양자화는 첫 토큰 이후 Decode 측정 구간 안에 있다.
- p20의 계획 예산은 Nmax−1, 실제 평균은 N−1이다. 서로 바꾸거나 비용을 다시 합산하지 않는다.
- 첫 토큰 EOS 또는 Nmax=1의 DC·최초 packing 생략과 Decode 지표 null은 현재 구현 계약이다.
  p20에 명시적인 분모 0 방어가 없다는 이유로 이 경계 처리를 제거하지 않는다.
- p25의 기본 DVFS 그래프는 1~8K·각 5회에서 Decode 1122MHz를 보고한다.
  이 관찰만으로 모든 governor·장치·요청의 고정 동작이나 PC 제거의 효과를 입증하지 않는다.
- p26은 G128, CPU2201.6MHz·EMC3199MHz, **Historical P6-3 SLO +20%** 조건이다.
  그림의 SLO 내 최소 평균 에너지 주파수는 1K/8K에서 816MHz, 2K/4K에서 918MHz다.
  현재 LUT·현재 선택 정책의 새 실측으로 재분류하지 않는다.
- p11~13의 상세 수식 참조는 p23~25로 적혀 있으나 실제 수식 부록은 p28~30이다.
  근거를 찾을 때 실제 페이지를 함께 확인한다.

## 현재 저장소의 구현·검증 경계

이번 대조는 기존 HEAD `156aec7`의 문서와 관련 코드를 읽은 결과다. 코드 실행·GPU 재측정은 하지 않았다.

| 항목 | 현재 구현 | V10과의 차이 |
| --- | --- | --- |
| 요청 실행 | SessionRunner에서 PC→FP16 Prefill→DC→주파수 확인→packing→Decode | 제안서의 PC 없는 흐름은 미반영 |
| G 선택 | kneed 0.9.0의 Kneedle, 정상 미검출 G32·검출 오류 전체 G 복구 | ρ 기반 SelectG 미구현 |
| LUT | current selector로 Runtime5회·Reference10회와 pin/SHA 검증 | 모델별 Profile DB·Unknown 기본값 서비스와 다른 계약 |
| 개발 Online | Offline 초기값, 정확한 셀 EMA α=0.1, 다음 요청부터 반영 | PDF에 Online 유지·폐기·통합 정책이 명시되지 않음 |
| 목표 | 원 FP16 Reference에 TPOT 버퍼 기본 0.20을 한 번 적용, TTFT는 그대로 | p20의 τ_d가 원값/유효 상한 중 무엇인지 명시 필요 |
| qualification | 동결 performance의 source·LUT·장치·정책 조건에 결속 | V10·Unknown·새 G 정책·adaptive로 승계 불가 |

관련 구현 계약: [Decode 정책](policy-decode.md), [실행 경계](runtime-execution.md),
[Online LUT](online-lut.md), [적격성](runtime-qualification.md),
[기존 검증 기록](full-validation-current.md). 이 문서들은 현재 구현 또는 고정 시점의 기록이다.

## 후속 구현 전에 정할 사항

| 항목 | 확인된 근거 | 미정·충돌 |
| --- | --- | --- |
| 최종 SelectG | p19는 ρ 최소, p14~16·20은 Kneedle | 새 기준 채택 시 기존 의사코드·fallback·오류 처리까지 일관되게 개정 필요 |
| G별 E_i | p27 비교는 816MHz | 일반 제어에서도 816MHz 고정인지, SLO 내 최적 f별 값인지, 예산 필터를 언제 적용할지 미정 |
| 비율 예외·동점 | p19는 ΔE>0인 구간만 정의 | 에너지 감소 구간 없음, 매우 작은 ΔE, ΔQ≤0·음수 비율, 동일 ρ의 처리 미정 |
| Unknown 기본값 | 기본 DVFS·G_default, p18·20에는 f_default 표기도 있음 | G_default 수치, governor 기본 DVFS와 고정 f_default의 관계, 지원 모델·실패 구분 미정 |
| Profiler·Accepted | 전체 실측 후 DB 등록 | 실행 시간대·유휴 임계·충전 판정·선점/재개·전체 측정 완료/수락 기준·원자적 공개 미정 |
| 목표·Online | Reference는 p20에 등장, p26은 과거 +20% 조건 | 새 모델 Reference 확보, 버퍼 표기, Online 승계와 ρ 계산에 쓸 에너지 출처 미정 |
| 평가 | 기존 Evaluation Plan의 개발/평가 분리·짝지은 비교 유지 | PC·Kneedle 제거 실험 등 V9 ablation을 V10에 맞춰 별도 재설계·동결 필요 |

이 항목들은 문서 정리의 차단 사유가 아니다. 정책 확정·코드 구현·새 성능 주장에 앞서
해결할 연구 과제다. 기존 LUT 원본·manifest·Reference·SHA·실측값·정식 검증의 한계는 보존한다.

## 이번 작업의 검증 범위

PDF 30쪽의 텍스트와 아키텍처·정책·부록 도표를 대조하고, V9·Evaluation Plan·현재 코드의
해당 경로를 확인했다. 로컬 문서 4개의 상대 링크 49개·코드 블록·공백과 `git diff --check`를
통과했고, 문서만 변경됐는지와 원본 PDF의 SHA-256 보존을 확인했다.
컴파일 대상 변경이 없어 코드 회귀 테스트나 GPU 실험은 실행하지 않는다.
새 품질·SLO·에너지 개선 또는 formal qualification을 주장하지 않는다.
