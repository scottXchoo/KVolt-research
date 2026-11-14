# RULER legacy v1 대표 입력 묶음

[NVIDIA/RULER 고정 revision `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)의
**수정하지 않은 공식 생성기**로 생성한 탐색용 대표 표본이다. upstream README는
기존 requirements/evaluation 스크립트를 deprecated로 표시한다. 최신 별도 평가
프레임워크와 구분하여 **legacy v1**으로 기록한다.

| 과제 | 내용 | 문항 | 공식 출력 예약 |
| --- | --- | ---: | ---: |
| `niah_single_1` | noise 안의 단일 key/value 회수 | 10 | 128 |
| `niah_multikey_2` | 여러 needle 중 요청한 key의 value 회수 | 10 | 128 |
| `vt` | 변수 할당 체인 추적: 1 chain, 4 hops | 10 | 30 |
| `fwe` | Zipf coded words의 최빈 3단어 회수 | 10 | 50 |

총 40문항, 공식 `validation` 생성 split, `seed=42`, `max_seq_length=4096`,
`tiktoken cl100k_base`, `base` completion template을 사용했다. 외부 essay·QA
corpus는 필요 없다. 공식 13과제 전체 평가, KVolt의 기존 MQ·MV 파일럿, 본 평가
입력 동결 또는 새 Quality LUT가 아니다. GPU 추론·FP16/G별 점수·SLO·에너지·
qualification을 이번 입력 준비로 측정하거나 승계하지 않는다.

## 파일과 검증

- `data/*.jsonl`: 공식 생성기가 쓴 JSONL 4개를 그대로 복사했다. 재직렬화하지 않았다.
- `source.json`: revision, 원본 source/license hash, 생성 환경·의존성·encoding 및
  wonderwords vocabulary hash, 생성 데이터 byte/row/SHA-256.
- `protocol.json`: 과제 인자·순서·프롬프트·채점·범위 계약.
- `LICENSE`: upstream Apache-2.0 원문.
- `generation-requirements.txt`: 실제 임시 생성 환경의 전체 `pip freeze`.
- `dataset.py`, `__init__.py`, `__main__.py`: stdlib 전용 로더·채점·검증 CLI.
- `reproduce.py`: 고정 upstream source를 확인하고 공식 생성기를 호출하는 재현 도구.

`source.json`과 `protocol.json`의 SHA-256은 코드 상수로 고정했다. 메타데이터를
데이터와 함께 다시 hash해도 코드 pin이 달라져 거부한다. 과제 필터 전 전체 파일의
byte·SHA-256·행 수·필드·참조 정답을 확인한다. 누락·변조 시 실패하며 네트워크나
임시 toy 데이터로 대체하지 않는다. 기본 경로는 현재 작업 디렉터리와 무관하다.

저장소 루트에서:

```sh
python3 src/experiments/quality/RULER/dataset/dataset.py validate
PYTHONPATH=src python3 -m experiments.quality.RULER.dataset validate
```

```python
from experiments.quality.RULER.dataset import load_dataset, score

records = load_dataset()                 # 40 rows in fixed task/file order
tracking = load_dataset(task="vt")       # still validates all four files first
fraction = score("ALPHA, beta", ["alpha", "beta", "gamma"])  # 2 / 3
```

반환 필드는 `id`, `task`, `prompt`, `answers`, `max_new_tokens`, `source_index`,
`source_length`다. `id`는 `ruler/{task}/{ordinal:04d}`이며 문항 순번을 쓴다.
`source_index`는 upstream `index` 원값이다. NIAH 생성기는 이를 첫 정답의
문자 위치로 덮어쓰므로 문항 순번으로 해석하지 않는다.

`prompt = input + answer_prefix`이며 별도 gold completion을 붙이지 않는다.
회수해야 할 정답이 원 context 내부에 존재하는 것은 과제 정의다. chat template,
모델·tokenizer 선택, 추론은 로더 밖에 있다. 채점에는 **생성된 continuation만**
전달한다. prompt 전체를 채점하면 context 내부 정답 때문에 잘못된 만점이 나온다.

`source_length`는 공식 `length`, 즉 **cl100k 입력 토큰 수 + 과제별 출력 예약**이다.
40개 모두 이 합계를 실제 encoding으로 대조했다. Llama 입력 길이 `L`이 아니며,
4096도 정확한 Llama 문항 길이를 뜻하지 않는다. 실행기가 실제 모델 tokenizer로
입력 길이를 측정해야 한다. truncation·padding·chat template으로 암묵 변환하지 않는다.

## 공식 채점 의미

`evaluate.py::postprocess_pred`와 동일하게 prediction을 strip하고 ASCII control
문자 `U+0000..U+001F` 각각을 LF로 치환한 뒤 다시 strip한다. 각 정답에 대해
`answer.lower() in prediction.lower()`를 계산한 평균을 반환한다. 부분점수를
반올림하지 않고 `0..1`로 유지한다. 공식 task 집계는 이 값들의 평균에 100을
곱한 뒤 소수 둘째 자리로 반올림한다. 단어 경계·숫자 정규화·추가 오답 감점은
없다. 예를 들어 `largerword`도 `word`를 포함하므로 정답으로 계산한다.

## 공식 생성기 재현

실제 생성 환경은 macOS arm64, CPython **3.13.2**, NumPy **2.2.6**, SciPy
**1.16.3**, tiktoken **0.12.0**, wonderwords **2.2.0**, NLTK **3.9.2**다.
전체 전이 의존성은 requirements 파일에 고정했다. SciPy 1.15.3 wheel은 이 환경에서
`dlopen` 오류가 발생하여 1.16.3을 사용했다. upstream 알고리즘을 수정하지 않았다.
다른 OS/Python의 동일 출력은 아래 byte 검증을 통과했을 때만 재현 성공으로 본다.

네트워크는 최초 source/dependency/cache 준비에만 필요하다. 프로젝트 환경에
패키지를 설치하지 않고 임시 venv를 사용한다. 아래에서 `PROJECT`는 이 저장소의
절대 경로다. NLTK는 noise/needle 과제에서도 upstream prepare가 불러오므로
`download_dir`를 명시해 **임시 캐시에 미리 준비**한다.

```sh
PROJECT="$PWD"
RULER_TMP="$(mktemp -d)"
python3 -m venv "$RULER_TMP/venv"
"$RULER_TMP/venv/bin/python" -m pip install -r "$PROJECT/src/experiments/quality/RULER/dataset/generation-requirements.txt"
curl -L --fail -o "$RULER_TMP/source.tar.gz" https://api.github.com/repos/NVIDIA/RULER/tarball/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
mkdir "$RULER_TMP/upstream"
tar -xzf "$RULER_TMP/source.tar.gz" -C "$RULER_TMP/upstream" --strip-components=1
export NLTK_DATA="$RULER_TMP/nltk_data"
export TIKTOKEN_CACHE_DIR="$RULER_TMP/tiktoken_cache"
export SSL_CERT_FILE="$("$RULER_TMP/venv/bin/python" -m certifi)"
"$RULER_TMP/venv/bin/python" -c 'import os,nltk; [nltk.download(name, download_dir=os.environ["NLTK_DATA"], raise_on_error=True) for name in ("punkt", "punkt_tab")]'
"$RULER_TMP/venv/bin/python" "$PROJECT/src/experiments/quality/RULER/dataset/reproduce.py" --upstream-dir "$RULER_TMP/upstream" --output-dir "$RULER_TMP/generated"
```

`reproduce.py`는 pinned `source.json`을 필수로 읽고, 사용한 upstream source 파일
13개의 byte/SHA-256을 확인한다. 각 과제는 `prepare.py --benchmark synthetic
--tokenizer_type openai --tokenizer_path cl100k_base --max_seq_length 4096
--num_samples 10 --random_seed 42 --model_template_type base`로 실행한다.
`PATH`를 현재 Python venv로 맞춰 upstream의 `python` 하위 프로세스도 같은
환경을 사용하게 한다. 기존 생성 출력은 덮어쓰거나 재사용하지 않는다.

공식 `prepare.py`는 자식 실패를 잡고도 exit 0으로 종료할 수 있으므로 이 도구는
결과 파일의 존재와 행 수·byte·SHA-256을 추가 확인한다. **종료 코드 0은 네 과제
모두 저장된 원본과 byte 단위로 일치했다는 뜻**이다. 실행 환경이나 패키지 차이로
결과가 달라지면 성공으로 표시하지 않는다. `cl100k_base.tiktoken`의 공식 URL과
SHA-256(`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`)도
`source.json`에 남겼다. 검증 CLI와 로더는 이 생성 환경을 전혀 요구하지 않는다.
