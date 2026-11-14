# AGX3 GSM8K 환경·에너지 연결 검증

이 도구는 GitHub에서 받은 별도 checkout의 기존 GSM8K 실행기와
`experiment_control`, `EnergyMeter`를 함께 실행한다. 실제 대상은
`agx32@100.94.58.28`이며 device-tree serial SHA-256도 확인한다.
기존 `/mnt/kvolt-data/kvolt`의 모델·KIVI·Python 환경을 읽으며 설치하거나
기존 작업 checkout을 변경하지 않는다.

프로토콜은 **GSM8K test 첫 10문항, 같은 첫 문항 warmup 1회, 최대 신규 512토큰,
FP16 가중치, K2V2/G32/residual128, plain-text train 첫 5문항 CoT, greedy·beam1,
자연 EOS**다. `--limit`는 기본 10이며 다른 값은 거부한다. 기존
`model.kivi_llama.generate`를 사용하며 새 stop-string/`lm_eval` 경로를 호출하지 않는다.
부분집합 정확도를 전체 1,319문항 성능으로 해석하지 않는다.

## 실행 조건

호환되는 기존 CUDA·PyTorch·Transformers·FlashAttention·Triton·Accelerate·KIVI
native 환경과 아래 로컬 모델 snapshot이 필요하다. 실행자는 해당 라이브러리 경로를
명시해야 한다. 모델의 기본 위치는
`/mnt/kvolt-data/kvolt/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`다.
`--model-name`으로 다른 로컬 snapshot을 지정할 수 있으며 기존 provenance 검사를 통과해야 한다.

`profile.json`은 nvpmodel 0/MAXN을 확인하고 CPU policy0/4/8의 min/max를
2,201,600 kHz로 설정·확인한다. 기존 governor를 유지하고 종료 시 원 설정을 복원한다.
GPU·EMC·팬 설정은 변경하지 않는다. GPU 제어값과 swap-in/out 변화는 명시한 rule로 검사한다.
팬 서비스와 EMC bwmgr_halt를 포함한 전후 제어값도 비교한다.
온도·스로틀링 적격성 기준은 새로 만들지 않으며 미가용 센서와 read error를 별도로 보존한다.

기존 `/run/lock/kvolt-p3-hardware.lock` inode의 비대기 독점 lock을 torch import 전에
확보한다. 모델 로드 후 환경 scope에 같은 lock을 빌려 주므로 이중 획득하지 않는다.
lock busy이면 종료 코드 75이며 기다리거나 다른 작업을 중단하지 않는다.
CPU sysfs 설정 권한이 필요하므로 승인된 실행자가 적절한 권한으로 실행해야 한다.

## Git checkout에서 실행

실행 코드가 포함된 커밋을 GitHub에 push한 뒤 AGX3의 **새 검증 전용 clone**에서 받는다.
아래 `<FULL_COMMIT_SHA>`는 실제 push한 전체 40자리 commit SHA로 바꾼다.
기존 장치의 checkout에 파일을 덮어쓰지 않는다.

```sh
git clone --branch codex/agx3-gsm8k-control-20261114 \
  https://github.com/scottXchoo/KVolt-research.git \
  /mnt/kvolt-data/kvolt-agx3-gsm8k-github-20261114
git -C /mnt/kvolt-data/kvolt-agx3-gsm8k-github-20261114 checkout --detach <FULL_COMMIT_SHA>

PYTHONDONTWRITEBYTECODE=1 /mnt/kvolt-data/kvolt/.venv/bin/python -B \
  /mnt/kvolt-data/kvolt-agx3-gsm8k-github-20261114/tools/agx3_gsm8k/verify.py \
  --expected-commit <FULL_COMMIT_SHA> \
  --output-dir /mnt/kvolt-data/kvolt/artifacts/gsm8k-control-github-20261114
```

위 Python 호출은 CUDA 라이브러리 환경과 sysfs 권한이 이미 준비된 실행자 안에서 수행한다.
clone 소유자와 실행 사용자가 다르면 Git의 dubious ownership 검사가 적용된다.
승인된 `sudo` 실행에서는 전역 Git 설정을 바꾸지 않고 해당 명령에만
`GIT_CONFIG_COUNT=1`, `GIT_CONFIG_KEY_0=safe.directory`,
`GIT_CONFIG_VALUE_0=/mnt/kvolt-data/kvolt-agx3-gsm8k-github-20261114`를 명시한다.
예를 들어 `sudo env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory
GIT_CONFIG_VALUE_0=/mnt/kvolt-data/kvolt-agx3-gsm8k-github-20261114 ...` 뒤에
필요한 CUDA 환경과 위 Python 인자를 전달한다.
`--source-root`의 기본값은 실행 스크립트가 속한 repository root다.
스크립트 자체도 그 root의 `tools/agx3_gsm8k/verify.py`여야 한다.
`--profile` 기본값은 같은 디렉터리의 `profile.json`이다.
`--output-dir`는 checkout 밖의 새 디렉터리만 허용하며 그 아래 `attempt1`을 생성한다.
실패한 실행을 덮어쓰거나 자동 재시도하지 않는다.

시작·종료 시 Git HEAD가 `--expected-commit`과 같은지 확인한다.
`src/**`, `pyproject.toml`, `tools/agx3_gsm8k/**`의 tracked 파일을
`git show <commit>:<path>` 원 바이트와 SHA-256으로 대조한다.
이 범위의 tracked 변경과 untracked 파일은 거부하며 ignore된 임의 source도 거부한다.
`__pycache__` 안의 `.pyc`/`.pyo`만 bytecode cache 예외로 둔다.
`-B`와 `PYTHONDONTWRITEBYTECODE=1`로 새 bytecode 작성을 막는다.
결과 base에는 검증한 `source-manifest.json`과 실제 사용한 profile 사본을 저장한다.

## 계측 경계와 산출물

각 본 측정에서 기존 `generate` 전체 호출을 EnergyMeter로 감싼다.
tokenization·입력 H2D·생성·신규 continuation decode와 완료 동기화를 포함한다.
모델 로드·warmup·추가 token evidence 복사·채점·결과 파일 I/O·collector 준비와
우측 coverage 대기는 제외한다. 환경 관측 thread와 작은 generation hook의 오버헤드는 포함한다.
warmup은 energy 측정 없이 같은 모델·입력·512 상한으로 실행한다.

- `source-manifest.json`, `profile.json`: Git source 결속과 실제 profile 사본.
- `attempt1/gpu-validation.json`: 장치·소스·모델·환경·실제 token ID·EOS/상한·종료 정리 결과.
- `attempt1/environment/`: snapshot, events JSONL, 준비·실행·rule·복원 summary.
- `attempt1/results/`: 기존 GSM8K predictions JSONL과 subset summary.
- `attempt1/warmup.jsonl`, `energy-results.jsonl`: warmup과 문항별 에너지 연결 기록.
- `attempt1/energy/000000/` 등: `energy.json`, 원 전력 CSV·JSONL.

출력 token ID를 보존하여 prompt prefix 제외·실제 N·EOS/상한·greedy·cache·KIVI 경로를 검증한다.
N=1 첫 EOS이면 decode matmul 0회를 허용한다. 에너지는
`VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0`의 독립 3레일 합이며 GPU 단독·벽전력이 아니다.
signed 값을 보존하며 idle subtraction·0 clipping을 하지 않는다.
기본 polling 20 ms·허용 최대 gap 200 ms, `sensor_response_validated=false`다.
TTFT/TPOT·Prefill/Decode 분리·J/token·SLO·에너지 개선·formal qualification을 주장하지 않는다.

정상 완료는 10개 생성·10개 유효 에너지, 환경 준비/실행/복원 성공과 required rule 위반 0,
전후 제어값 일치, CUDA allocated/reserved 0, collector child 0, lock 재획득을 요구한다.
선택 센서의 미가용/read error 수는 환경 rule 판정과 구분하여 보고한다.
실패 시 생성 partial JSONL과 가능한 원 계측 자료를 보존하며 종료 코드 1을 반환한다.

## 독립 원자료 감사

원격 산출물을 받은 뒤 같은 commit의 checkout에서 실행한다.
감사기는 표준 라이브러리만 사용하며 프로젝트 에너지 적분기를 import하지 않는다.

```sh
python3 -B tools/agx3_gsm8k/independent_audit.py \
  --attempt /absolute/artifact-base/attempt1 \
  --source-root /absolute/checkout \
  --expected-commit <FULL_COMMIT_SHA> \
  --output /absolute/new-audit.json
```

감사기는 외부 commit·Git blob·실제 source bytes·source manifest를 교차 대조하고,
CSV/JSONL 수치·signed mV×mA·rail별 strict 경계 coverage·gap·사다리꼴 적분·합계 J/W,
GSM8K 점수·첫10문항 매핑·token prefix·EOS·환경 복원·정리를 확인한다.
저장된 token ID와 text의 대응 근거를 사용하며 tokenizer를 독립 재구현하지 않는다.
출력 JSON은 새 파일만 허용한다.
