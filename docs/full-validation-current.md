# KVolt 최신 소스 전체 검증 — 2026-11-09

이 보고서의 실측은 TPOT 허용 버퍼0의 당시 정책이다. 후속 사용자 결정으로 추가한
기본20% 버퍼는 이 과거 실행·판정을 바꾸지 않는다. 현재 설정과 새 실행의 계약은
[Decode 정책](policy-decode.md)을 따른다. 새 소스/비율의 GPU 적격성을 승계하지 않는다.

이 보고서는 `79dbe41d1b5a562d0986e4c4dd80483432a5ba77`의 실제 실행과 최종 독립
감사를 확정한다. frozen M4 구현 적격성과 별도 개발 Online 기능 검증은 통과했다.
다만 core Decode21건 모두 TPOT 목표를 넘었으며, Online의 정식 적격성이나 성능·
에너지 개선을 입증한 결과는 아니다.

원본의 통합 기준 `194c93c41ae77bf7c5091b073b327b367067d6a1`에 검증된
qualification 수정 세 개와 M4의 명시적 frozen 실행을 통합했다. 로컬 독립 CPU
검증과 AGX2 CPU/GPU 검증은 통과했다. 수신 원자료에서 새 M4의 실제 performance8건,
두 전체 복원, 별도 Online11건 및 추가 복원, 새 프로세스 cold reload를 확인했다.
최신 grant는 새 source/plan의 실행으로 생성됐으며 이전 source `079aaabb...`의
적격성을 승계하지 않았다. 독립 M4 policy1307검사·raw24단계·Online847검사는 모두
오류/누락 없이 통과했다. 추가5434검사·helper97/97·실제 변조16/16거부도
통과했고, 원본 작업 branch는 사용자 커밋을 보존한 안전한 fast-forward로 검증 코드에
반영됐다. 원본에 반영한 검증 코드 커밋은 `79dbe41...`이다. 최종 독립 종합 보고는
[v6-final-independent-review.json](../artifacts/full-validation-20261109/v6-final-independent-review.json)이다.

## 범위와 연구 기준

대상은 고정 모델·LUT·소프트웨어·장치/boot·제어/전력 프로필에 결속한 Runtime
구현의 정확성 및 기록/복구 계약이다. 본 평가, RULER 품질, 통계적 SLO 달성,
에너지/속도 개선, 센서의 물리적 보정이나 wall power를 입증하는 작업은 아니다.

기준은 [Method](https://app.notion.com/p/40be0acd77f88014a36beb5285250208),
[Evaluation Plan](https://app.notion.com/p/40de0acd77f880f98165f98215f51e20),
최신 `AGENTS.md` 및 `docs/online-lut.md`다. PC-B1의 Reference → 한 번의 Bp 계산 →
필터 조회, DC-B1의 품질/Kneedle → Reference/Bd → 필터 Decode 조회, DC-M1의 정상
Knee 미검출 시 중앙 G=32를 적용한다. 오래된 full-lookup-first/all-G 정상 미검출
문단으로 되돌리지 않는다. 시간 예산을 충족하는 활성 후보가 없으면 전역 최저
지연의 정확 동점 후보로 fallback하고 원 에너지와 확정된 동점 순서를 보존한다.

[OL-1](https://app.notion.com/p/412e0acd77f8812ea01ac4bb21e416c4)과
[OL-2](https://app.notion.com/p/412e0acd77f881a1a501e158aecc8dab)의 Online 정책은 개발
모드의 별도 기능이다. 요청 시작 revision을 PC/DC가 함께 쓰고, 완료 후 저장한
실측으로 정확한 `(L,f)`/`(L,G,f)` 셀만 EMA α=0.1로 갱신한다. 다음 요청부터
적용하며 비격자 길이는 이웃 Offline knot에 전파하지 않는다. Quality/Reference는
갱신하지 않는다. N=1은 Decode 갱신을 하지 않고, 전력 미계측/무효이면 유효한
시간만 갱신한다. 유효한 SLO 위반과 복원만 실패한 완료 측정은 자동 제외하지 않는다.

## 소스 이력과 원본 보존

초기 검증은 원본 `c354fd2e...`의 사용자 변경 16 tracked + 4 untracked 파일을
격리 복사해 시작했다. 그중 scratch `reference.json`을 제외한 검토된 변경을
통합했고, source `079aaabb...`의 r4 M4는 실제 performance까지 통과했다.
검증 중 원본에는 별도 작업으로 `a99e5ec...`/`194c93c...`의 Online 기능이 추가됐다.
이 병행 변경을 덮어쓰지 않고, 최신 clean main을 새 worktree의 기준으로 삼았다.

| 단계 | 통합 커밋 | 내용 |
| --- | --- | --- |
| 최신 기준 | `194c93c41ae77bf7c5091b073b327b367067d6a1` | Offline 기반 Online overlay·영속 상태·요청 종료 feedback·개발 기본 활성화 |
| 검증 수정 1 | `0ad132c1bf146ec20afae006c5557c222c8df8c8` | 기존 `45d167a...`를 적용. current 공통 로더, sealed CPU fixture 분리, runtime/CPU provenance 동결 |
| 검증 수정 2 | `62514a13609f29b15ebafe2f58be85f6204c3e6f` | 기존 `97e0b3d...`를 적용. 기록된 nested Reference identity의 canonical session 감사 |
| 검증 수정 3 | `f3ffe2300f5ca95b328c10a12d5148b9cbde6670` | 기존 `079aaabb...`를 적용. Runner도 공통 canonical runtime/reference identity 사용 |
| M4 frozen 고정 | `79dbe41d1b5a562d0986e4c4dd80483432a5ba77` | 모든 M4 경로에 `--no-online-lut` 명시 및 최신 실제 CLI/Runner 회귀 |

Runner 충돌은 기존 adaptive performance 조기 거부를 유지하면서 canonical 비교를
함께 보존해 해결했다. `policy/online.py`, `policy/lut.py`, `run_cli.py`는 최신 원본과
바이트가 같다. 원본 `194c93c` 대비 production 변경은 qualification 감사, Runner
gate, M4 도구에 한정되며 Offline/Reference/Quality나 Online 선택·EMA 알고리즘을
수정하지 않았다. source manifest는 runtime 48파일, Python+pyproject 111파일이다.

통합 전 로컬 검증에서는 원본 `194c93c`의 clean 상태·사용자 소스, LUT8파일,
과거 r2/r3/r4 archive와 r4 grant SHA를 보존했다. 모든 실제 감사가 끝난 뒤 메인은
원본 `codex/m4-qualification`이 여전히 clean `194c93c`임을 다시 확인하고,
검증 코드 `79dbe41...`로 조건부 fast-forward했다. 사용자 `194c93c`는 조상으로
보존되며 원본 worktree는 clean이다. reset/stash/push는 하지 않았다.

보존된 사용자 변경과 승인된 통합 변경을 구분한다. 원본에도 의도대로 반영된
production 수정은 qualification 감사·Runner gate·M4 도구3파일이다. Online 정책,
Offline/Reference/Quality 원본8파일은 바뀌지 않았다. 반영 뒤 원본과 검증 worktree의
생산·도구·테스트111파일/runtime48파일 SHA가 모두 같다.
[코드 반영 영수증](../artifacts/full-validation-20261109/original-code-integration.json)과
[통합 전후 근거](../artifacts/full-validation-20261109/v6-integration/handoff.json)에
source/LUT/사용자 조상 보존 조건이 기록돼 있다.
최종 문서만 추가하는 후속 커밋이 생겨도 실제 실행 소스의 커밋·Python/tool/test
바이트를 그 문서 커밋으로 다시 명명하지 않는다. 검증 identity는 원자료의
`79dbe41...`와 고정 파일 SHA로 유지한다.

## V-1~V-6 상태

| 항목 | 기존 source079 결과 | 최신 source79dbe 결과 및 완료 조건 |
| --- | --- | --- |
| V-1 소스/입력/출처 동결 | 완료 | 완료. 소스·입력·SHA 동결, archive 수신/불변성 감사, 원본 clean 확인 후 안전한 코드 FF |
| V-2 CPU/독립 정책 검산 | 완료 | 완료. 로컬/AGX2 각각2342pass/103 GPU skip/81 subtests, 집중548pass·compile, policy1307·변조16/16거부 |
| V-3 실제 Backend/native | GPU103·greedy15쌍/반복·EOS40쌍 확인 | 완료. 최신 GPU103=prepare102+greedy1, greedy15쌍+native15반복·3360cache field 독립 검산 |
| V-4 frozen M4 | r4 qualified, report591928... | 완료. 정상8/반복8/EOS8/performance8, native16+반복, 두 신호/두 전체 복원·raw24단계, 새 grant24ed63... |
| V-5 독립 감사/정직한 범위 | 완료 | 완료. 4개 실제 감사 PASS, 301artifact·전력/경계/복구/환경·grant·greedy/변조·SLO 미달 보존 |
| V-6 최신 main+Online 통합 | 별도 대상 아님 | 완료. 최신 frozen M4 및 Online11건·state0→6→10·cold reload·추가 복원, 독립847검사·코드 반영 |

위 실행 결과는 수신 원자료와 독립 수식·파일 바이트·불변성 감사에 연결된다.
정적 준비/helper 검사는 실제 실행 검사와 별도로 표기한다. raw 감사의24단계는
정책/에너지/해시의 여러 assertion을 포함하며 CPU/GPU 테스트 건수를 대신하지 않는다.

## LUT·입력·환경 결속

| 대상 | 고정 identity |
| --- | --- |
| current selector | `f35aaf2955e8dd533d131949f03b2070102259ef39aa80af1efe49a0e44fc44e` |
| Runtime repeat5 manifest | `b421fcd10ecb57df7d6962aa9a5f48e264e90defc0c76a3544218d31685f783b` |
| Reference repeat10 manifest | `d332875ccce626dd99700da8c7a128bf4801577333e76ee40a987f4051311e40` |
| Reference 원본 310bytes | `359fd02c609738755c87847f27b19286f45a3a7cb8d6bef8acd7c369e1c30c2b` |
| pinned M1 입력 | `e9f0c05e0d21e13356f6a64d1176b03bcff613c708b0b0fd72228f711aadc20d` |
| 최신 M4 plan | `400593ecbab163a8d008c058dc075802eadedf3443782fe5dd8b9b78c2828c95` |
| 별도 Online plan | `8dcae8044db280ec5e94adc3983ad8f89e63e2726b009a67e81fcaf8773ab4ce` |

Reference는 길이별10회·총80요청의 TTFT와 요청 평균 TPOT의 독립 최댓값이다.
N=1은 TPOT 집계에서 제외하고 목표 배율은1.0이다. 길이만 보간하며 외삽하지 않는다.
원본 runtime accuracy/SLO 및 quality eligibility=false, Reference controller
integration=false를 바꾸지 않는다. source-validation 요약은 검증하지만 없어진
개별 component 원본 manifest를 이번에 다시 읽었다고 주장하지 않는다.

로컬은 `/private/tmp/kvolt-prefill-test-venv/bin/python`의 Python3.12.14,
NumPy1.26.4/SciPy1.15.3/kneed0.9.0/pytest8.4.2다. 전체 CPU 결과는
**2342 passed, 103 skipped, 81 subtests passed**, 18.30초, 오류·경고0이다.
JUnit testcase는2445개이고 outer tests2526에는 subtests81개가 포함된다.
skip은 cache101/Prefill1/greedy1의 GPU opt-in 사례뿐이며 데이터/CPU skip은 없다.
관련7모듈 집중548pass, compile도 exit0이다.

아래 재현 명령과 문서 끝의 보존 링크는 현재 로컬 `KVolt-support/` 배치에 맞췄다.
원 측정 수치·source identity·보존 artifact의 원본 경로 기록은 바꾸지 않았다.

```sh
env -u KVOLT_RUN_GPU -u KVOLT_RUN_B1_GPU PYTHONDONTWRITEBYTECODE=1 \
  KVOLT_LUT_BUNDLE=/Users/scottxchoo/Desktop/CSLab/07_Projects/KVolt-support/KVolt-lut-source/data/luts/agx2-repeat5-20261103 \
  /private/tmp/kvolt-prefill-test-venv/bin/python -m pytest -q --tb=short \
  --junitxml=artifacts/full-validation-20261109/v6-integration/tester-full.xml
```

AGX2 실행 인터프리터는 `/mnt/kvolt-data/kvolt/.venv/bin/python`이다.
`PYTHONPATH=/home/agx31/kvolt-policy-deps:/home/agx31/kvolt-runtime-test-deps`,
`LD_LIBRARY_PATH=/mnt/kvolt-data/kvolt/.local/cusparselt/lib:/usr/local/cuda/lib64`,
`PYTHONDONTWRITEBYTECODE=1`을 사용한다. 수신 binding은 Python3.10.12,
Linux5.15.199-tegra/aarch64/glibc2.35, torch2.5.0a0+872d972e41.nv24.8,
transformers4.43.1, accelerate0.33.0, flash-attn2.7.4.post1, triton3.5.0을 기록한다.
정책 의존성은 NumPy1.26.4/SciPy1.15.3/kneed0.9.0이다. native binary·모델·boot·
physical identity·장치/전력 프로필 전체는 `m4-plan-r1/qualification.json`의 binding에
결속된다. AGX2 CPU2342pass/103GPUskip/81subtests는 pytest 기준75.02초였고 외부
의존성 deprecation warning14개가 기록됐다. compile은 exit0이다.
prepare GPU102pass/0skip은102.73초, 별도 greedy1pass/0skip은230.33초로 총103pass다.
원격 경로·전체 명령은 `prepare-reviewed-latest.sh`, `execute-reviewed-latest.sh`에
보존한다. 공통 nonblocking lock은 `/run/lock/kvolt-p3-hardware.lock`이다.

## 최신 frozen M4 — 실측 및 독립 감사 통과

새 M4는 `m4-plan-r1`을 사용한다. 정상/max-cap8요청과 L1024 자연 first-EOS8문맥,
L1024/4096/8192의 Nmax≤512를 동결했다. M4의 development·repeat·EOS·두 signal
probe·performance 모두 `--no-online-lut`를 명시한다. 정상/성능 formal warmup과
signal probe의 warmup-none은 구분하며 Online state/feedback이 없어야 한다.

확인할 최종 증거는 다음과 같다: 8주파수 왕복32 receipt/no-op, 선택 주파수 검증 후
최초 packing, native16쌍과 반복·Runtime8 재현, SIGINT130/SIGTERM143의 저장 prefix와
후속 not_run, 두 전체 복원, 실제 performance8건, candidate가 소비한 SHA와 최종
qualification bytes의 동일성, 8 required checks와 모든 참조 artifact hash다.
장치·boot·source·kernel·runtime/ref pin·프로필·coverage 변경 및 이전 grant는 거부돼야 한다.

실제 `execution-status.json`은 qualified/errors0이며 required checks8개가 passed다.
report는301개 artifact를 참조한다. 정상/반복/EOS/performance는 각각8건 완료했고
생성·시간·전력이 모두 유효하다. M4의6세션 모두 `online_lut.enabled=false`,
`policy=frozen`, `disabled_reason=explicit_no_online_lut`를 기록한다. formal warmup은
15+15+40+15=85case이며 두 signal probe는 warmup-none이다.

두 신호는 각각 exit130/143, 완료2건·중단 prefix1건·후속 not_run5건을 남겼다.
따라서6세션48records는 core완료32 + signal완료4 + 중단2 + not_run10이다.
native-exact는16case+각 reference repeat, Runtime repeat는8쌍이다. 주파수32receipt와
두 전체 복원 감사는 passed/errors0/primary_error=null이다. performance proof는 실제
8요청·원 전력7995행·SLO 위반7건 보존을 기록한다. candidate를 소비한 report SHA와
최종 `qualification.json` SHA는 모두 `24ed63e8...`이며 독립 감사가 다시 대조했다.

core32건에서 **TTFT32/32 충족, TPOT0/21 충족(21건 미충족), N1 Decode null11건**이다.
performance8건만 보아도 TTFT8건 충족, Decode가 있는7건 모두 TPOT 미충족이다.
이 미달은 정상 완료·유효 측정이며 기술 오류로 제외하지 않았다. 실제 N 분포는
`1:11, 2:6, 46:6, 48:3, 50:6`으로 실제 최대 N=50이다. formal warmup의 최대 N도50이다.
Nmax512를 포함한 입력 동결은 실제512토큰 생성이나 그 미도달 shape 검증을 뜻하지 않는다.

### TPOT 미달의 관측과 예측 하한

`performance-findings.json`은 원 Decode40셀 전체의 최소 예측과 동결 FP16 목표를
대조한다. 아래는 초/Decode token이며, 실제 지연이나 품질 제약을 제거한 보장을
뜻하지 않는다.

| L | 전체40셀 최저 Offline TPOT | FP16 목표 | 제어 overhead 차감 전 예측 | 실제 선택과 해석 |
| --- | --- | --- | --- | --- |
| 1024 | 0.1187258241, G128/f1122 | 0.1033992056 | 최저값부터 목표 초과 | 예측 가능한 시간 통과 후보가 없어 latency fallback |
| 4096 | 0.1192061872, G128/f1020 | 0.1176141950 | 최저값부터 목표 초과 | 예측 가능한 시간 통과 후보가 없어 latency fallback |
| 8192 | 0.1249472416, G128/f1122 | 0.1367639872 | 전체 최소값은 목표 이하 | 실제 품질 활성 G32에서 f918/예측0.1345003065 선택. 실제는 목표 미충족 |

core Decode21건 중15건은 latency_fallback/predicted_infeasible이고6건은
energy_min/predicted_feasible이나, 실제21건 모두 TPOT 목표를 넘었다. L1024/4096은
제어 예산을 차감하기 전부터 원 표의 예측 하한이 목표보다 크다는 사실을 확인할
수 있다. 이것만으로 모든 실제 미달을 정책 구현 오류나 LUT라는 단일 원인으로
확정하거나 원 pin·목표를 바꿀 수는 없다.

같은 run-first의 L8192/G32/f918에서 실제 N2는 TPOT0.396634403초,
initial packing0.251696406초였다. N48은 TPOT0.140109195초이고 packing0.250334422초의
실제 Decode47토큰당 몫은 약0.005326264초다. 실제 N에 따라 최초 비용의 토큰당
몫이 달라지는 관측이다. packing은 이미 측정 Decode 안에 포함돼 있으므로 이를
TPOT에 다시 더하지 않으며, 이 두 사례를 원인 분해나 일반적인 효과 추정으로
해석하지 않는다.

이 한계는 새로 발견한 미분석 문제가 아니라 기존
[L2 분석](https://app.notion.com/p/412e0acd77f881c0a2f0c85a53a0bbb4)의
startup/steady 비용·실제 N 불확실성 문제가 최신 source/Reference10에서도 남아 있음을
확인한 결과다. startup/steady 분리와 출력 길이 불확실성을 반영한 모델, 명시적
FP16 fallback, fused quantize/pack은 기존 후속 검증·정책 결정 대상이며 이번에
구현하거나 자동 채택하지 않았다.
[L3 고정 실험](https://app.notion.com/p/412e0acd77f8815b9e14ca40fd2e794a)의 과거
AGX1 LUT 기반 G64/1122·N48 성공은 현재 G32·최신 Runtime의 효과로 승계하지 않는다.
[R5-L2](https://app.notion.com/p/412e0acd77f881c8b0f8cab20754b974)의 Reference5 시절
8K/N48·여유0.054708ms 단일 성공도 현재 결과나 안정적 달성 보장이 아니다.
이 기록을 근거로 목표·품질·LUT pin을 완화하지 않는다.

## 별도 개발 Online smoke — 실측 및 독립 기능 감사 통과

정식 M4와 분리한 `online-smoke-r1`은 A6요청, 같은 state를 재사용하는 B4요청,
별도 time-only state의1요청으로 총11건이다. L1024/1536/4096/8192를 포함하며
1536은 prepare의 실제 tokenizer로 길이·token SHA를 검증한 합성 입력이다.
Nmax1/2와 자연 종료를 사용하고 force-length나 EOS 억제는 없다.

원자료상 A6건의 사용 revision은0~5, final은6이다. B는 revision6으로 시작해6~9를
사용하고 final10으로 끝났다. A/B10건 모두 시간·전력이 유효하고, 별도 time-only1건은
시간만 유효하며 energy_not_collected=1이다. time-only의 별도 state는0→1이다.
A의 final export6은 B 이후 공유 state10과 구분해 보존했다. snapshot상 공유 state는
Prefill4셀·Decode8셀이고, N1의 Decode null과 actual N=1/2 통계를 보존한다.

native11case+각 반복은 실제 선택 G에서 통과했다. Online 추가 전체 복원도
passed/errors0/primary_error=null이다. checksum 손상, alpha 불일치, adaptive
performance의 세 negative는 각각 exit2이며 hardware guard 호출이0이다. 거부는
해당 오류 문구로 확인했고 일반 missing qualification 실패로 대신하지 않았다.
원 관측 → 정확한 셀의 EMA → 다음 요청 revision·비격자 exact-L·Quality/Reference
불변·시간/에너지 독립 갱신은 별도 감사가 독립 수식으로 재생해 통과했다.

Online에서도 A/B/time-only 전체11건의 **TTFT9/11 충족(2건 미충족), TPOT0/9
충족(9건 미충족), N1 Decode null2건**이다. TTFT 미충족은1536의 A/B 각1건이다. 이 정상 관측은 feedback에서
자동 제외되지 않았다. 이 소수 기능 smoke로 SLO·에너지 개선을 주장하지 않는다.

모든 복원 후 **새 프로세스**의 cold reload는 passed/fresh_process=true이며 revision10·
observation_count10을 읽었다. 실제 state 파일 SHA는 전후 모두
`0e4bea0eb3c76d761e7385a13280b68f98ee75e51b8f6b9c5661a30fa0495297`이다.
state payload의 semantic `state_sha256=24f8a396...`와 파일 바이트 SHA는 서로 다른
항목이다. 새 프로세스는 최신 grant24ed63...를 수용하고 과거079 grant591928...를
binding mismatch로 거부했다. hardware_lock_calls=0은 읽기 전용 helper의 기록이며
source call-path 검토도 별도로 보존했다. 독립 감사는 원 JSON으로 B final과 state를
대조하고 7길이(1024/1535/1536/1537/2048/4096/8192)의 Prefill56/Decode280예측까지
검산했다. 생산 `observe`나 state validator를 정답 oracle로 재호출하지 않았다.

Online11건의 성공은 feedback·영속·재사용 기능의 관측 범위다. adaptive formal
qualification, 개선 효과나 새 품질 적격성을 발급하지 않는다.

## 독립 감사와 재현 근거

| 감사 | 실제 결과 | 결과 JSON SHA-256 |
| --- | --- | --- |
| M4 policy | 1307검사, errors0. 실제6세션/48records·원 표/공식 Kneedle/예산·선택·frozen provenance 검산 | `5138440357d98d364599740c471065722a5ea3fd93fc3144bc13552d61f6c24d` |
| M4 raw/evidence | 24단계, failures0/missing0. 301artifact/30frozen/111source·48runtime·신호·복원·실제 grant 검산 | `c4ee438cb9baec42c0c5cf468c7795c533a4799bac28bd04981dabd3ed64e1dc` |
| Online/EMA/cold reload | 847검사, errors0. 11요청·원 관측/수식·상태 재사용·3negative·native·추가 복원·새 프로세스 검산 | `a77b601062ab131b9e65e80e87699f683069f45e34f490bf7d033a35d826880e` |
| 추가 독립 감사 | 5434검사, errors0. archive605파일·greedy15쌍/15반복·3360cache field·현재 TPOT finding 대조 | `f7a21d741e8195b2a55076328ace89609fe236c5f2856c0b7d975609b7c87429` |

M4는 원 전력26757행,101구간,504개 에너지/토큰당 지표를 독립 Fraction 및 Decimal60
적분으로 교차 검산했다. 기록값과의 최대 차이는 에너지 약2.56×10⁻¹³J,
토큰당 약4.29×10⁻¹⁵J/token이다. 이는 부동소수 누적 차이이며 센서 정확도 수치가 아니다.
38개 실제 memory 기록은 유효했고 미실행10건은 값을 만들지 않았다. 관측 RSS HWM
최대27,268,214,784bytes, CUDA max allocated23,514,013,696bytes/reserved25,105,006,592bytes를
각각 보존하며 합산하지 않는다.

실제 current grant 수용과 stale source/boot/runtime/reference/file SHA/kernel/프로필,
grant 범위 밖 길이2048·cap513, 과거 r5 grant·실제 Reference3/5 등15가지 거부를 확인했다.
Online은 원 전력4227행과144개 에너지 지표를 대조한 뒤 기록 측정으로 EMA를 재생했다.
처음 Online 감사에서 time-only 결과에 `power`가 있다고 가정한 KeyError가 있었으나
이는 감사기 schema 호환 오류였다. artifact 감사기만 고쳐 실패 기록을 보존하고
Online 감사를 다시 통과했으며 먼저 통과한 M4 감사 결과도 유지됐다.
production/source/raw는 바꾸지 않았다.

정확한 명령·source/plan 인자는 `independent-auditor-commands.json`, 결과는
`v6-m4-independent-policy-results.json`, `v6-m4-independent-raw-results.json`,
`v6-online-independent-results.json`, `v6-additional-independent-results.json`이다.
97/97개의 준비/helper 검사와16/16개의 실제 변조 거부도 별도 통과했다. 최초 감사기
오류의 log는 `auditor-tests/online-initial-failure-report.json`, `online-initial-failure.log`,
`online-exception-trace.log`로 남겼다. 최종 종합 보고 SHA는
`248baa9182521a0bd327027e50a732628f934922b1fcb2f6b157ded2cb462f5f`다.

greedy smoke는 L1024/4096/8192·5G·Nmax4의 범위다. 통과한 cache tensor 원본은
bounded harness가 삭제했으므로 사후 검토는 저장된 native digest/shape/bytes와
검토한 source·JUnit proof를 확인했다. 없는 tensor 파일을 다시 바이트 비교했다고
주장하지 않는다.

## 이전 실패와 역사적 성공

| 시도 | 보존 결과 | SHA-256 |
| --- | --- | --- |
| 기존 r2/source2dcf517 | 생성8건 후 감사의 nested reference KeyError. 첫 전체 복원 완료, grant 없음 | archive `9e58f6ee7c098d6040e10b03b87fa2e986f4af1a90025a7719c66ed62dbd8762` |
| 기존 r3/source97e0b3d | 정상24/native/두 신호·두 복원 후 Runner의 수동5field 비교로 마지막 gate 거부. performance8 not_run, candidate/final 삭제 | archive `4c529784be5717c9b8dfc7c6a294d3382922691268731142bc67ca6632aa9257` |
| 기존 r4/source079aaabb | 전체 M4와 독립 감사 통과. 최신 Online 소스를 포함하지 않는 역사적 적격성 | archive `8ae2a1208f21367bed334fac4cd0396da34b1b712d60d97db8bf2f9df7666238` |
| 기존 r4 grant | 원본 그대로 보존, 최신 source identity로는 binding mismatch 거부 | `5919280344502a4bf8bfc9c632352c76697110ac0a7dae08dc64e7f06db952c2` |

이 실패들은 추가 실행으로 덮어쓰지 않았다. canonical producer/session auditor/Runner를
일관되게 고쳤고, real current 및 explicit compact의 실제 binding → 유효 CPU fixture
report → real loader → CLI → real Runner 통합 회귀와 변조/이전 참조 거부를 추가했다.
CPU의 hardware 대체 결과를 GPU 적격성으로 표시하지 않았다.

## 최종 결과·무결성

| 산출물 | 상태/값 |
| --- | --- |
| 최신 M4 execution status / grant SHA / artifact 수 | qualified/errors0, `24ed63e89350328c14ae06736eb330612d14cd3c095b6326962306cd6fc4821b`, 301artifacts/8checks |
| 최신 M4+Online+GPU+cold-read archive SHA | `75aff90e939485a7342c345817aed06fe5dcb6e176063cc5c5beaab3fcef8cc8` |
| archive 크기/수신 inventory | 4,715,131bytes, 759members, 605regular files, 비압축23,469,351bytes. inventory604파일 일치 수신 검사 passed |
| M4 raw·policy 독립 감사 | policy1307검사/raw24단계 통과, 오류·누락0. 정확한 명령·SHA는 독립 감사 표 참조 |
| Online 독립 EMA/state/cold-read 감사 및 native·복원 | 847검사 통과, errors0. 추가 복원/native11×2/7길이 cold56+280 검산 |
| 최신 32 core M4의 TTFT/TPOT 위반·N1 null | TTFT0위반, TPOT21위반, N1 null11. 생성/시간/전력32건 유효 |
| 실제 최대 N 및 요청별 N 분포 | 실제/최대 warmup 모두50. `1:11, 2:6, 46:6, 48:3, 50:6` |
| 새 프로세스 grant/state | current24ed63... 수용, old079/591928... 거부. 공유 state revision10·파일 SHA 전후 동일 |
| source/raw/LUT/원본 반영 | raw/source 독립 감사 전후 불변. 원본194 clean 재확인 후79로 안전한 코드 FF, 사용자194 조상·LUT8 보존 |
| 추가 검증 | 추가5434검사, helper97/97, 실제변조16/16거부, greedy15쌍/15반복·3360cache field 통과 |
| 원본 branch | `codex/m4-qualification`이 검증 코드79에 FF됐고 clean. 문서만의 후속 커밋은 측정 source identity를 바꾸지 않음 |

별도 감사가 raw integer mV/mA·timestamp의 엄격한 외부 bracket, sample gap, clipping·
정규화와 Fraction/Decimal 독립 적분을 재검산했다. 실제 SLO 미달은 정상 관측으로
세며 기술 무효·interrupted prefix·not_run과 구분한다. Nmax≤512 coverage는 자연 EOS로
도달한 cache shape 범위이며 실제512토큰 생성이나 모든 입력으로 일반화하지 않는다.
메모리는 boundary snapshot/CUDA reset window이고 RSS HWM은 process lifetime peak다.
RAM/RSS/CUDA pool은 겹치므로 합산하지 않는다. hash는 기록 바이트의 출처 결속이며
신뢰할 수 없는 report 작성자를 인증하지 않는다.

최신 추적: [V-6 카드](https://app.notion.com/p/412e0acd77f88134849de135709cd872).
로컬 CPU 근거: `artifacts/full-validation-20261109/v6-integration/tester-results.json`
(SHA `933bfabea8e7df4f70eff309d33c72bdbef9f889af082420c68b047c658bdad8`),
`handoff.json` (SHA `c926dad08fa6c86199e722b8390f3f8d4a5dd5d6b94299ad4caca8d4073d8321`).
최종 상태는 [독립 종합 보고](../artifacts/full-validation-20261109/v6-final-independent-review.json)와
[원본 코드 반영 기록](../artifacts/full-validation-20261109/original-code-integration.json)에 연결된다.


## 주요 증거 위치

다음 상대 링크는 이 문서의 최종 위치 `docs/full-validation-current.md`와 원본에
복사한 `artifacts/full-validation-20261109/`를 기준으로 한다.

- [최종 독립 종합 보고](../artifacts/full-validation-20261109/v6-final-independent-review.json)
- [M4 qualification](../artifacts/full-validation-20261109/m4-plan-r1/qualification.json) · [M4 동결 plan](../artifacts/full-validation-20261109/m4-plan-r1/plan.json)
- [M4 policy 감사](../artifacts/full-validation-20261109/v6-m4-independent-policy-results.json) · [M4 raw 감사](../artifacts/full-validation-20261109/v6-m4-independent-raw-results.json)
- [Online 독립 감사](../artifacts/full-validation-20261109/v6-online-independent-results.json) · [추가 독립 감사](../artifacts/full-validation-20261109/v6-additional-independent-results.json)
- [원자료 archive](../artifacts/full-validation-20261109/kvolt-full-validation-20261108-latest-complete.tar.gz) · [수신 검증](../artifacts/full-validation-20261109/received-verification.json)
- [정확한 독립 감사 명령](../artifacts/full-validation-20261109/independent-auditor-commands.json) · [AGX2 준비 명령](../artifacts/full-validation-20261109/prepare-reviewed-latest.sh) · [검토된 실행 명령](../artifacts/full-validation-20261109/execute-reviewed-latest.sh)
- [새 프로세스 cold reload](../artifacts/full-validation-20261109/online-cold-reload.json) · [복원 후 grant 수용/거부](../artifacts/full-validation-20261109/post-restoration-acceptance.json)
- [현재 TPOT 하한·실측 finding](../artifacts/full-validation-20261109/performance-findings.json) · [로컬 CPU 명령·환경·보존](../artifacts/full-validation-20261109/v6-integration/tester-results.json)
- [원본 코드 FF 영수증](../artifacts/full-validation-20261109/original-code-integration.json)

이전2026-11-08 시도의 원자료는 별도 보존 worktree에 있다. 현재 원본에 없는 과거
경로를 상대 링크로 가정하지 않는다:
[이전 source079 보고서](/Users/scottxchoo/Desktop/CSLab/07_Projects/KVolt-support/KVolt-validation-20261108/docs/full-validation.md),
[실패 r2 archive](/Users/scottxchoo/Desktop/CSLab/07_Projects/KVolt-support/KVolt-validation-20261108/artifacts/full-validation-20261108/m4-failed-r2.tar.gz),
[실패 r3 archive](/Users/scottxchoo/Desktop/CSLab/07_Projects/KVolt-support/KVolt-validation-20261108/artifacts/full-validation-20261108/m4-failed-r3.tar.gz),
[이전 성공 r4 archive](/Users/scottxchoo/Desktop/CSLab/07_Projects/KVolt-support/KVolt-validation-20261108/artifacts/full-validation-20261108/m4-complete-r4.tar.gz).
