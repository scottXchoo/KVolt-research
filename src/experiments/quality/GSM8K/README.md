# GSM8K CoT + 단일 greedy 입력 묶음

공식 GSM8K **main**의 train 7,473개와 test 1,319개를 오프라인 JSONL 데이터셋으로 보존한다.
train의 첫 5개 풀이를 고정 예시로 사용하고, test 전체를 원본 순서대로 평가 입력으로 구성한다.
이 5-shot은 프로젝트의 로컬 기준선이며 공인된 유일한 GSM8K 프롬프트가 아니다.

이 묶음은 데이터 준비·채점용이다. 기존 `../run.py`의 데이터/채점 TODO를 연결하거나
모델·G·주파수를 선택하지 않는다. RULER, LUT 원본, Runtime, Method V10의 미정 정책과
기존 qualification은 변경하지 않는다. GPU 추론, FP16/G별 정확도·지연·에너지 측정은 수행하지 않았다.

## 원본과 라이선스

- 저장소: [openai/grade-school-math](https://github.com/openai/grade-school-math)
- 고정 revision: [`3101c7d5072418e28b9008a6636bde82a006892c`](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)
- [공식 train](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl),
  [공식 test](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl)
- 원본 MIT 라이선스: [LICENSE](LICENSE)
- 출처·원본 크기·행 수·SHA-256·Git blob SHA-1: [source.json](source.json)

| 파일 | 행 수 | bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `data/train.jsonl` | 7473 | 4166206 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| `data/test.jsonl` | 1319 | 749738 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |
| `LICENSE` | — | 1062 | `86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b` |

원본 `question`·`answer`와 파일 바이트는 수정하지 않았다. train/test의 exact question 교집합은 0개다.
`source.json`과 `protocol.json`은 `dataset.py`의 SHA-256 상수로 고정한다.
데이터와 출처 metadata를 함께 바꾸어도 로더가 거부한다. 의도적인 프로토콜 변경은
별도 revision·검토와 새로운 해시가 필요하다. 코드 상수까지 임의로 바꾸는 공격을 막는 서명 체계는 아니다.

## CPU에서 사용

저장소 루트에서 실행한다. Python 표준 라이브러리만 필요하며 네트워크·모델 다운로드는 없다.

```sh
python3 src/experiments/quality/GSM8K/dataset.py validate
PYTHONPATH=src python3 - <<'PY'
from experiments.quality.GSM8K.dataset import load_dataset, score

records = load_dataset()
sample = records[0]
print(sample["id"], sample["answer"])  # gsm8k/main/test/0000 18
print(sample["prompt"])             # 완성형 5-shot 입력
print(score("Reasoning.\n#### 18", sample["answer"]))  # 1.0
PY
python3 -m pytest -q tests/test_gsm8k_dataset.py
```

기본 데이터 경로는 `dataset.py`의 `__file__` 기준이므로 현재 작업 디렉터리와 무관하다.
다른 디렉터리에서는 위 CLI에 `dataset.py`의 절대 경로를 사용한다.
`load_dataset(bundle_dir=...)`, `validate_bundle(bundle_dir=...)`와 CLI의 `--bundle-dir`는
동일 pin을 가진 복사본을 검증할 때 사용할 수 있다. 누락·손상·다른 출처는 명시적으로 실패한다.
원본/프로토콜 SHA·bytes·행 수·필드·정답 추출·train/test 분리와 1,319개 prompt를 모두 검증한다.
검증 CLI의 성공 종료 코드는 0, 번들 검증 실패는 1이다.

각 레코드는 다음 필드를 제공한다.

| 필드 | 계약 |
| --- | --- |
| `id` | `gsm8k/main/test/0000`부터 `gsm8k/main/test/1318`까지 원본 행 순서 |
| `question` | 해당 test question 원문 |
| `prompt` | 지시문 + train 첫 5개 question/CoT 풀이 + 해당 test question |
| `answer` | 원본 `#### ` 뒤의 최종 숫자 문자열에서 쉼표만 제거 |
| `reference_solution` | 해당 test answer 전체 원문; prompt와 별도 보존 |

## 고정 프롬프트와 생성 계약

[protocol.json](protocol.json)은 구분자·템플릿·지시문과 생성 설정을 고정한다.
예시는 train indices `[0, 1, 2, 3, 4]`만 사용한다. test 정답은 예시 선택이나 prompt에 사용하지 않는다.
각 예시는 `Question: ...\nAnswer: <원본 CoT 풀이>`이며, 블록 구분자는 LF 두 개다.
마지막 블록은 대상 test question의 `Question: ...\nAnswer:`에서 끝난다.
고정 지시문은 단계별 풀이와 `#### <number>` 형식의 최종 줄을 요청한다.

생성 설정은 `do_sample=false`, `num_beams=1`, `num_return_sequences=1`,
`max_new_tokens=512`, `use_cache=true`다. 단일 greedy 출력이며 sampling seed를 사용하지 않는다.
도구 호출·custom stop 문자열은 없다. 향후 실행 시 모델/토크나이저의 자연 EOS 또는 출력 상한으로 종료한다.
512는 설정 상한이며 실제 생성 토큰 수를 뜻하지 않는다.

모델·토크나이저 revision, Instruct chat-template, 실제 입력 토큰 길이 L은 아직 동결하지 않았다.
이 plain-text 입력을 Instruct에 사용할 때는 tokenizer 준비 단계에서 chat-template 적용 여부와
최종 토큰 ID·해시·L을 별도로 기록해야 한다. 기존 Runtime LUT 지원 범위 밖의 L을 외삽하지 않는다.
이 번들의 문자열 prompt 해시는 chat-template 적용 후 토큰 입력 해시를 대신하지 않는다.

## 채점과 재현 기준

[공식 채점 코드](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py)의
`ANS_RE`와 첫 match 선택·쉼표 제거·문자열 exact match를 따른다.
`score(prediction, answer)`의 prediction에는 **새로 생성한 continuation만** 전달한다.
prompt까지 넣으면 train 예시의 첫 `#### `가 선택되므로 잘못된 평가가 된다.
reference는 로더의 정규화된 `answer` 또는 원본 `reference_solution`을 받는다.

`#### 1,234`와 정답 `1234`는 일치한다. `18.0`·`018`은 `18`과 다르다.
여러 marker가 있으면 첫 match를 사용한다. marker가 없는 출력은 0.0점이다.
정답 reference가 유효하지 않으면 예외를 내며, 양쪽 marker 누락을 정답으로 처리하지 않는다.
regex를 그대로 사용하므로 뒤쪽 설명을 허용하는 등 upstream 추출의 관대한 동작도 유지한다.

`validate`는 test 순서의 각 prompt에 LF 하나를 덧붙인 UTF-8 바이트를 차례로 연결한 SHA-256을 출력한다.
고정 비교값은 아래와 같다. 원본 정답이 포함된 평가 레코드를 Runtime 입력으로 넘길 때는
`prompt`만 사용하고 gold 필드는 결과 분석 경로에 따로 둔다.

- source SHA-256: `9f31829434a4713b4a27bca61159999668004bddc7bde844b0766c824d06f7cc`
- protocol SHA-256: `6dd960f1a7eb2076db75bb98cc7d654a25a65837e3d515e5a5d39342d537ca40`
- prompt SHA-256: `b9c9766a403fde5512212c66a9b853683e5af8a896ecf491808d5d8eb92c23e1`

이 데이터 추가는 기존 실측·grant의 정확도, SLO 또는 formal 적격성을 승계하지 않는다.
