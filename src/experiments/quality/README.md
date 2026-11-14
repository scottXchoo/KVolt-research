# KIVI + Llama-3.1-8B-Instruct Quality

데이터셋과 평가 지표를 정하기 전의 최소 스켈레톤이다. `run.py` 상단에서 모델과 K/V bits, group size, residual length, 출력 상한을 바꾼다. 기본은 **`meta-llama/Llama-3.1-8B-Instruct`**, FP16 가중치, K2V2, G=32, residual=128이다.

모델 로딩·생성은 공통 [src/model/kivi_llama.py](../../model/kivi_llama.py)에 있다. 다른 실험에서도 `from model.kivi_llama import load_model, generate`로 가져와 `load_model(model_name, group_size=32)`와 `generate(model, tokenizer, prompt, max_new_tokens=128)`을 사용한다.
`quality/run.py`에는 실험 설정, 데이터 로딩·채점 TODO와 실행 흐름을 둔다.

1. `load_dataset()`을 구현해 `[{"prompt": "...", "answer": "..."}]`를 반환한다. `prompt`에는 Instruct 모델에 줄 완성된 입력만 넣고 정답은 분리한다. 해당 실험의 입력 형식과 필요한 chat template 준비는 데이터 로더가 담당한다.
2. `score(prediction, answer)`에 선택한 데이터셋의 평가 지표를 구현한다.
3. 저장소 루트에서 실행한다.

```sh
PYTHONPATH="src:/path/to/KIVI" python src/experiments/quality/run.py
```

CUDA와 프로젝트 호환 PyTorch·Transformers·FlashAttention·Triton·Accelerate 및 설치된 KIVI quant 커널(`kivi_gemv`)이 필요하다. `/path/to/KIVI`는 호환 checkout의 `quant.new_pack`·`quant.matmul`을 import할 수 있는 경로다. 기존 환경 버전은 [provenance.py](../../kvolt_runtime/backend/provenance.py)의 `VERSIONS`를 참고한다(Transformers 4.43.1). 최신 upstream 환경의 호환성을 가정하지 않는다. Hugging Face 모델 접근 권한과 충분한 GPU 메모리도 필요하다. `MODEL_NAME`에는 로컬 모델 경로도 넣을 수 있다.

모델은 저장소의 `LlamaForCausalLM_KIVI`를 사용한다. 생성은 batch 1·greedy·자연 EOS이며 새 토큰만 디코드한다. 공통 helper는 chat template를 자동 적용하지 않고 로더가 준비한 입력을 그대로 사용한다. 별도 [GSM8K 실행기](GSM8K/README.md)는 기존 고정 plain-text 5-shot protocol을 사용한다.

기본 실행은 모델 로딩 전에 데이터셋 TODO로 중단한다. 실제 GPU 추론과 Quality 점수는 미측정이다. 정식 연구의 Instruct/RULER 평가와 별도이며 기존 LUT·Runtime 적격성을 변경하지 않는다.
