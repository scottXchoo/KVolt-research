# RULER KIVI Llama 실행기

`KIVI_Llama.py`는 NVIDIA/RULER legacy v1의 대표 4과제·40문항을 순서대로
생성하고 공식 substring recall로 채점한다. 기본 모델은
`meta-llama/Llama-3.1-8B-Instruct`이며 `model.kivi_llama.load_model/generate`를
재사용한다. 전체 13과제 RULER benchmark 결과로 해석하지 않는다.

FP16 가중치, K2V2·G32·residual128, greedy·beam1·한 출력·캐시 사용·자연 EOS를
고정한다. upstream vanilla completion의 `input + answer_prefix`를 그대로 전달한다.
chat template나 별도 BOS를 붙이지 않으며 tokenizer의 기본 special token 처리를 사용한다.
정답은 생성 함수에 전달하지 않고 신규 continuation만 채점한다.

## 데이터와 길이

포함 과제는 `niah_single_1`, `niah_multikey_2`, `vt`, `fwe`이며 각 10문항이다.
공식 revision `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`의 생성 스크립트를
`cl100k_base`, `max_seq_length=4096`, seed42 조건으로 실행한 대표 묶음이다.
세부 생성·출처·SHA-256 pin·라이선스·채점 규칙은 [dataset/README.md](dataset/README.md)를
따른다. 기존 평가 파일럿이나 LUT의 품질 표를 바꾸지 않는다.

`source_length`는 upstream `cl100k_base` 입력 길이에 과제별 출력 예약을 더한 값이다.
Llama의 실제 입력 토큰 수나 정확한 Llama 4K 길이를 뜻하지 않는다. 실행기는 각 문항에서
공용 생성 함수와 같은 `tokenizer(prompt, return_tensors="pt")` 호출로 실제 `input_tokens`와
token ID SHA-256을 기록한다. 실제 입력+출력 상한이 `model.config.max_position_embeddings`를
넘으면 자르지 않고 실패한다. 정확히 같은 경계는 허용한다.

공식 출력 상한은 두 NIAH 과제 128토큰, `vt` 30토큰, `fwe` 50토큰이다.
출력 길이를 고정하지 않으며 EOS가 먼저 나오면 종료한다.

## 실행

저장소 루트에서 다음 중 하나를 실행한다. `--task`를 생략하면 40문항 전체를,
지정하면 해당 과제의 10문항을 선택한다. `--limit N`은 선택 후 첫 N문항만 평가한다.
N은 선택한 문항 수 이하의 양의 정수다.

```sh
PYTHONPATH="/path/to/KIVI" python3 src/experiments/quality/RULER/KIVI_Llama.py --task niah_single_1 --limit 1 --output-dir artifacts/ruler/smoke-01
PYTHONPATH="src:/path/to/KIVI" python3 -m experiments.quality.RULER.KIVI_Llama --limit 3 --output-dir artifacts/ruler/smoke-02
PYTHONPATH="src:/path/to/KIVI" python3 -m experiments.quality.RULER.KIVI_Llama --output-dir artifacts/ruler/representative-01
```

`--model-name`은 모델 ID나 로컬 경로를 받는다. 직접 스크립트는 절대 경로로 실행할 수 있고,
모듈은 `PYTHONPATH`에 저장소 `src`의 절대 경로를 넣으면 다른 작업 디렉터리에서도 실행된다.
결과 상대 경로는 현재 작업 디렉터리 기준이다. 출력 경로를 생략하면
`artifacts/ruler/<UTC 실행 시각>/`을 만든다. 이미 존재하는 디렉터리는 덮어쓰지 않는다.

GPU 로딩 전에 인자·선택하지 않은 과제를 포함한 전체 데이터 pin·출력 경로 충돌을
검증한다. import와 `--help`는 GPU 패키지를 로드하지 않는다. 실제 생성에는 CUDA,
호환 PyTorch·Transformers·FlashAttention·Triton·Accelerate, KIVI quant 커널
(`kivi_gemv`)과 모델 접근 권한이 필요하다. `/path/to/KIVI`는 `quant.new_pack`·
`quant.matmul`을 import할 수 있는 호환 checkout이다. 기존 버전은
[provenance.py](../../../kvolt_runtime/backend/provenance.py)의 `VERSIONS`
(Transformers 4.43.1)를 참고한다. 최신 upstream 환경의 호환성은 가정하지 않으며,
실행기가 의존성을 설치하지 않는다.

## 결과와 검증 범위

- `predictions.jsonl`: `id`, `task`, 신규 `prediction`, 정답 `answers`, 원점수 `score`,
  원 `source_index`·`source_length`·길이 tokenizer 설명, 실제 tokenizer 모델명·`input_tokens`·
  `input_ids_sha256`, 과제별 `max_new_tokens`를 문항마다 기록하고 flush한다.
- `summary.json`: 모든 선택 문항이 성공한 뒤에만 생성한다. 평가 수·전체/선택 과제 문항 수·
  부분집합 여부·첫/마지막 ID·선택 과제·설정·dataset/protocol pin·선택 prompt SHA-256,
  원점수 평균과 과제별 평가 수/원점수 평균/백분율을 보존한다.

점수는 대소문자를 무시한 정답 substring 포함 비율이다. 여러 정답 중 일부만 포함하면
부분 점수를 받는다. `mean_score`는 문항별 0~1 점수의 micro 평균이고, `score_percent`는
이를 100배해 소수 둘째 자리로 반올림한 값이다. `per_task`는 같은 방식으로 과제별
집계하며 정답 문항 수로 대체하지 않는다. 공식 prediction 후처리는 채점에만 적용하고
원 continuation은 그대로 저장한다. 과제별 표본 수가 달라지는 선택에서는 micro 평균을
과제 평균의 macro 평균과 혼동하지 않는다. 40문항 전체 실행도 4과제 대표 묶음 결과다.

생성·토큰화·context 검증 오류는 실행을 중단하고 이미 flush한 JSONL을 남긴다.
실패 시 완료 summary가 없으며 자동 재시도·이어하기는 없다. 새 디렉터리에서 다시 실행한다.

macOS CPU fake 테스트는 흐름·기록·실패 처리를 검증한다. 이 실행기의 실제 CUDA 추론과
정확도는 이번 작업에서 측정하지 않았다. 모델/tokenizer revision까지 동결한 정식 평가
harness가 아니며, 시간·에너지·SLO를 측정하거나 개선을 주장하지 않는다.
Runtime·LUT·V10 정책·기존 formal qualification을 변경하거나 승계하지 않는다.
