# GSM8K KIVI Llama 실행 스켈레톤

공식 LM Evaluation Harness task는 명시적인 `--lm-eval`로 실행한다.
설치·실제 stop·원 토큰·계측 계약은 [LM-Eval 안내](../../../../docs/lm-eval.md)를 따른다.
아래는 기본 경로인 기존 고정 로컬 프로토콜이며 새 공식 task와 구분한다.

`KIVI_Llama.py`는 로컬 검증된 GSM8K main/test 1,319문항을 순서대로 생성하고
정답을 채점한다. 기본 모델은 `meta-llama/Llama-3.1-8B-Instruct`이며,
`model.kivi_llama.load_model/generate`를 재사용한다.

고정 조건은 FP16 가중치, K2V2·G32·residual128, train의 첫 5문항 CoT,
greedy·beam1·한 출력·자연 EOS·최대 신규 512토큰이다. 원본 plain-text prompt를
그대로 전달하며 chat template를 적용하지 않는다. `--model-name`은 Instruct 모델 ID나
로컬 경로를 받는다. Instruct 모델의 plain-text 생성은 AGX2 소수 문항에서 확인했다.
chat-template 프로토콜과 정식 평가는 별도다.
데이터 출처·고정 protocol·정답 추출 규칙은 [dataset/README.md](dataset/README.md)를 따른다.

## 실행

저장소 루트에서 둘 중 하나를 실행한다. `--limit`을 생략하면 전체 test를 평가하며,
지정하면 첫 N문항만 평가한다. N은 1~1,319의 정수다.

```sh
PYTHONPATH="/path/to/KIVI" python3 src/experiments/quality/GSM8K/KIVI_Llama.py --limit 3 --output-dir artifacts/gsm8k/smoke-01
PYTHONPATH="src:/path/to/KIVI" python3 -m experiments.quality.GSM8K.KIVI_Llama --limit 3 --output-dir artifacts/gsm8k/smoke-02
```

직접 스크립트는 절대 경로로도 실행할 수 있고, 모듈은 `PYTHONPATH`에 저장소의 `src`
절대 경로를 넣으면 다른 작업 디렉터리에서도 실행할 수 있다. 결과 상대 경로는 현재 작업
디렉터리 기준이다. `--output-dir` 생략 시 `artifacts/gsm8k/<UTC 실행 시각>/`을 만든다.
지정한 결과 디렉터리가 이미 있으면 실패하며 파일을 덮어쓰지 않는다.

GPU 로딩 전에 인자·전체 데이터의 SHA-256 pin과 결과 디렉터리 충돌을 검증한다.
실제 생성에는 CUDA 장치, 호환되는 PyTorch·Transformers·FlashAttention·Triton·Accelerate,
프로젝트 KIVI quant 커널(`kivi_gemv`)과 모델 접근 권한이 필요하다.
`/path/to/KIVI`는 `quant.new_pack`·`quant.matmul`을 import할 수 있는 호환 checkout 경로다.
기존 버전은 [provenance.py](../../../kvolt_runtime/backend/provenance.py)의 `VERSIONS`
(Transformers 4.43.1)를 참고한다. 최신 upstream 환경의 호환성을 가정하지 않는다.
이 실행기는 의존성을 설치하지 않는다.
import와 `--help`는 GPU 패키지를 로드하지 않는다.

## 결과

- `predictions.jsonl`: 문항별 `id`, `question`, 신규 continuation 그대로인 `prediction`,
  추출된 `predicted_answer`, 정답 `answer`, 0/1 `score`를 기록하고 문항마다 flush한다.
- `summary.json`: 모든 선택 문항이 정상 처리된 뒤 평가 수·정답 수·accuracy,
  전체 문항 수·부분집합 여부·첫/마지막 ID·모델/생성 설정·dataset/protocol pin·선택 prompt
  SHA-256을 저장한다. 부분집합 accuracy는 전체 test 결과로 해석하지 않는다.

정답/reference_solution은 생성 함수에 전달하지 않는다. 채점은 신규 continuation의 첫
`#### <number>` 일치를 사용하며 쉼표만 제거한다. marker 누락은 오답이고,
`18.0`과 `18`은 다르다. 프롬프트의 5-shot 정답을 채점에 포함하지 않는다.

생성 오류는 실행을 중단하고 이미 기록된 JSONL을 보존한다. 자동 재시도·이어하기는 없으며
실패한 실행에는 완료 summary를 만들지 않는다. 새 디렉터리에서 다시 실행한다.

기존 AGX2 기능 검증은 `ec21d1e`에서 명시 Instruct snapshot으로 첫 3문항을 실행한 기록이다.
이번 기본 모델 ID 교정을 새 커밋의 GPU 실측으로 다시 표시하지 않는다.

별도 탐색용 품질 실험이다. 모델/tokenizer revision과 실제 입력 토큰 길이를 동결·기록하는
정식 평가 harness는 아니다. Runtime·LUT·V10 제어 정책·qualification을 변경하거나 승계하지
않으며, CPU fake backend 테스트는 실제 GPU 추론·정확도 검증을 대신하지 않는다.
