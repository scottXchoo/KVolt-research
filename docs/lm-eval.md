# KIVI와 LM Evaluation Harness 연결

`experiments.lm_eval`은 `lm_eval==0.4.12`의 공식 task 구성·후처리·채점을 사용하고,
추론만 저장소의 `model.kivi_llama`로 실행한다. 기본 모델은
`meta-llama/Llama-3.1-8B-Instruct`, FP16 가중치·K2V2·G32·residual128이다.
`KIVILM`은 `lm_eval.api.model.LM`을 상속하고 `kivi-llama`로 등록된다.
HF 기본 backend/HFLM으로 모델을 로딩하지 않는다. loglikelihood·rolling likelihood
과제는 지원하지 않으며 `NotImplementedError`로 실패한다.

이 경로는 기존 [GSM8K](../src/experiments/quality/GSM8K/README.md)와
[RULER](../src/experiments/quality/RULER/README.md)의 고정 로컬 dataset/protocol과 구분된다.
기존 CLI는 기본 동작을 유지하고 명시적인 `--lm-eval`에서만 새 runner를 쓴다.
문항·few-shot·필터·stop·생성 상한이 다르면 같은 평가가 아니다.
공식 데이터와 합성 자료 준비에는 네트워크·외부 cache가 필요할 수 있다.

## 설치와 실행

기존 CUDA·KIVI·PyTorch·Transformers 4.43.1 환경은 먼저 보존한다. 별도 환경에서
선택 의존성 `.[lm-eval]`을 설치한다. 이 extra는 다음 평가 패키지만 고정한다.

```text
lm_eval[ruler]==0.4.12
datasets==3.6.0
huggingface-hub==0.36.0
```

`lm_eval[hf]`, `lm_eval[all]`은 사용하지 않는다. HF extra가 요구하는 새 GPU 패키지로
기존 backend 환경을 교체하지 않는다. core/ruler 설치가 Torch·Transformers·Accelerate·
FlashAttention·Triton·KIVI native 커널의 설치나 호환성 검증을 대신하지 않는다.
실제 모델 로딩은 기존 `load_model`의 CUDA 경로를 유지한다.

저장소 루트에서 준비된 별도 환경으로 실행한다. 새 결과 디렉터리를 지정한다.

```sh
python3 -m pip install '.[lm-eval]'
PYTHONPATH=src python3 -m experiments.lm_eval \
  --tasks gsm8k --limit 3 --seed 1234 \
  --group-size 32 --max-new-tokens 512 --max-input-tokens 8192 \
  --output-dir artifacts/lm-eval/gsm8k-smoke

PYTHONPATH=src python3 -m experiments.lm_eval \
  --tasks niah_single_1 --ruler-lengths 4096 --limit 3 \
  --max-new-tokens 128 --max-input-tokens 8192 \
  --output-dir artifacts/lm-eval/ruler-smoke
```

기존 진입점의 명시적 선택 예시는 다음과 같다.

```sh
python3 src/experiments/quality/GSM8K/KIVI_Llama.py --lm-eval \
  --limit 3 --output-dir artifacts/lm-eval/gsm8k-legacy-entry
python3 src/experiments/quality/RULER/KIVI_Llama.py --lm-eval \
  --tasks niah_single_1 --ruler-lengths 4096 --limit 3 \
  --output-dir artifacts/lm-eval/ruler-legacy-entry
```

`--num-fewshot`을 생략하면 공식 task 설정을 쓴다. 기존 로컬 GSM8K train 첫 5문항과
공식 task의 few-shot 선택이 같다고 가정하지 않는다. `--limit` 실행은 부분집합 결과다.
자동 재시도·이어하기는 없고 기존 결과를 덮어쓰지 않는다. `--help`와 runner import는
선택 평가 의존성·Torch·Transformers·GPU 없이 동작한다. 실제 실행은 패키지 버전을
검사하며 다른 lm-eval 버전을 조용히 사용하지 않는다.

## 요청·생성 계약

요청은 harness가 넘긴 순서 그대로 batch size 1, rank 0/world size 1에서 실행한다.
응답 cache와 request cache를 끄며 같은 prompt가 반복돼도 실제 추론한다.
모델 내부 KV cache(`use_cache=True`)는 유지한다. 자동 재정렬·자동 batch·분산·
chat template는 사용하지 않는다. 기본 warmup은 0이다.

지원하는 task 생성 인자는 `until`(문자열 또는 문자열 목록), `max_gen_toks` 또는
`max_new_tokens`, `do_sample=False`, `num_beams=1`, `use_cache=True`,
`temperature=0.0`이다. 두 토큰 상한이 함께 있으면 같은 값이어야 한다.
최종 출력 상한은 **task 상한과 CLI 상한의 최솟값**이다. RULER의 과제별 30·128 등
출력 상한을 일괄 512로 늘리지 않는다. bool·실수 토큰 수·0·음수와 sampling·beam·
batch·truncation 등의 미지원 인자는 명확히 실패한다.

`until`은 공통 `generate_with_details`의 실제 stopping criterion에 전달된다.
새 continuation에서 stop 문자열을 처음 완성하면 추론을 끝낸다. prompt에 있는 stop과
prompt·continuation을 걸친 문자열은 종료 조건이 아니다. 채점용 `text`는 가장 먼저
나타난 stop 앞까지다. `raw_text`는 special token을 숨겨 decode한 원 continuation이다.
`output_token_ids`는 실제 stop/EOS 토큰까지 모두 보존한다. special token 문자열이 stop이면
표시용 raw text에 없어도 실제 종료 사유에 남긴다. 출력 문자열을 다시 tokenize하여 N을
추정하지 않는다. 자연 EOS·stop·상한 종료를 구분한다.

`max_input_tokens`는 실제 모델 tokenizer의 입력 ID 길이에 적용한다. 입력 상한 초과 또는
입력 길이+유효 출력 상한이 모델 context를 넘으면 실패한다. 입력을 자르거나 출력 예산을
조용히 줄이지 않는다. 실제 로딩 후 backend 클래스와 k_bits/v_bits/group_size/
residual_length가 요청한 KIVI 설정인지 검증한다. CPU tiny 모델 주입은
`preloaded_test_model`로 별도 기록하며 GPU KIVI 실행으로 표시하지 않는다.

## RULER tokenizer와 길이

공식 그룹 `ruler`와 `niah_single_1`, `niah_multikey_2`, `ruler_vt`, `ruler_fwe` 등의 task는
고정 0.4.12 패키지에서 탐색한다. `TaskManager` metadata에 모델 tokenizer 이름을
`tokenizer`·`pretrained`로, 요청 길이를 `max_seq_lengths`로 전달한다.
기본 요청 길이는 4096이며 공식 합성·scoring 규칙을 쓴다.
기존 bundled 40문항의 cl100k_base 길이와 동일한 평가라고 표시하지 않는다.

이 버전의 공식 metric 길이는 `4096, 8192, 16384, 32768, 65536, 131072`다.
RULER에 다른 길이를 지정하면 생성 전에 실패한다. 그룹 `ruler`의 공식 집계 metric은
4096만 정의돼 있어 그룹 실행은 길이 목록에 4096이 있어야 한다. 8192만 평가하려면
`--tasks niah_single_1 --ruler-lengths 8192`처럼 개별 task를 고른다.
여러 길이를 요청해도 그룹 집계 점수는 전 길이 평균이 아니며 과제별 길이 metric을 확인한다.

공식 RULER 길이는 과제별 합성 입력·예약 출력 조건을 포함할 수 있다.
모든 완성 prompt가 정확히 요청 길이의 입력 ID를 갖는다는 뜻은 아니다.
요청 길이·실제 input IDs/N·task 설정을 함께 확인한다. 실제 추론 직전의 별도
context 검사는 항상 적용하며 overflow 시 truncation으로 통과시키지 않는다.

## 계측과 결과

`--measure-energy`로 선택 계측하며 `--power-profile PATH`는 계측 활성화 시에만 쓴다.
미계측은 센서 접근·CUDA synchronization을 추가하지 않는다. 계측 시 tokenization·입력
H2D·실제 생성과 stop 검사·raw continuation decode 및 CUDA 완료가 구간에 들어간다.
모델 로드·사전 입력 검사·원 토큰 ID CPU 복사·채점·파일 저장은 제외한다.
collector 시작·오른쪽 bracket 확보·정리는 구간 밖이다. 공통 helper가 한 번만 계측하며
adapter는 동일 비용을 다시 더하지 않는다. Prefill/Decode·TTFT·TPOT·Decode J/token을
분리 산출하는 경계가 아니다.

기존 EnergyMeter의 독립 3레일 합·signed 값·경계 보간·bracket 검증을 유지한다.
전력 무효는 `valid=false`, `energy_j=null`로 보존하며 0으로 바꾸지 않는다.
생성과 채점이 정상인 전력 무효 요청도 품질 결과와 함께 남긴다.
센서 응답 검증과 전력 계약은 [runtime-measurement.md](runtime-measurement.md)를 따른다.

| 산출물 | 내용 |
| --- | --- |
| `run.json` | 별도 schema, 설정·seed·패키지/환경·task 구성/출처·완료 상태 |
| `results.json` | 모든 선택 과제가 끝난 뒤 lm-eval 집계 결과; sample은 분리 |
| `samples.jsonl` | harness 문항별 응답·정답·필터·채점 자료 |
| `records.jsonl` | 요청 순서별 prompt SHA-256, task/doc/repeat, 원 입력·출력 IDs와 N, raw/반환 text, 종료 사유, 요청/유효 생성 설정, 모델·attention 클래스/출처, 에너지 결속; 요청마다 flush |
| `energy/000000/energy.json` | 원 계측 결과; 숫자·무효 원인 보존 |
| `energy/000000/power-samples.csv` | 기존 `samples_csv`로 저장한 전력 표본 |
| `energy/000000/power-trace.jsonl` | 원 worker trace 바이트; receipt SHA와 결속 |

파일 경로에는 요청 ordinal만 사용하며 dataset ID를 쓰지 않는다. JSONL 원 trace hash와
CSV hash를 혼동하지 않는다. 생성 예외에 계측 결과/receipt가 있으면 같은 ordinal에
원자료와 실패 record를 저장한 뒤 원예외를 유지한다. 진단 저장도 실패하면 stderr와
지원 Python의 exception note에 남긴다. 실행 실패에는 완료 `results.json`을 만들지
않으며 이미 flush한 요청 자료를 보존한다.

## 검증 범위

실제 설치된 lm-eval 0.4.12에서 다운로드 없는 local `ConfigurableTask`와 tiny CPU HF
모델로 `simple_evaluate`·정답 채점·실제 stop에 따른 토큰 단축·원 N·요청 순서·trace
저장·실패 보존을 검사한다. 공식 GSM8K/RULER task discovery는 dataset 다운로드나
공식 benchmark 전체 실행 검증이 아니다.

개발 CPU의 Torch 2.7.1·Transformers 4.57.3 검사는 GPU 환경의 Transformers 4.43.1·
KIVI CUDA·native 커널·FlashAttention·실센서 호환성 검증을 대신하지 않는다.
새 경로의 실제 AGX 품질·시간·에너지·정식 통계 평가는 별도 실행 근거가 필요하다.
stop으로 출력량이 달라지므로 기존 512토큰 상한 실행과 단순 비교하여 속도·에너지 개선을
주장하지 않는다. Runtime·LUT·V10 정책이나 기존 formal qualification을 변경·승계하지 않는다.
