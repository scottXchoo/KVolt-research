"""CPU checks for the pinned GSM8K input bundle; never import GPU code."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from experiments.quality.GSM8K import dataset


BUNDLE = Path(dataset.__file__).resolve().parent


def test_nested_bundle_layout_and_public_api():
    assert BUNDLE.name == "dataset"
    assert BUNDLE.parent.name == "GSM8K"
    assert sorted(path.name for path in BUNDLE.parent.iterdir() if path.name != "__pycache__") == [
        "KIVI_Llama.py", "README.md", "dataset",
    ]
    from experiments.quality.GSM8K.dataset import (
        PINNED_PROTOCOL_SHA256, PINNED_REVISION, PINNED_SOURCE_SHA256,
        extract_answer, load_dataset, main, score, validate_bundle,
    )

    assert extract_answer("#### 1,234") == "1234"
    assert score("#### 18", "18") == 1.0
    assert load_dataset is dataset.load_dataset
    assert validate_bundle is dataset.validate_bundle
    assert main is dataset.main
    assert PINNED_REVISION == "3101c7d5072418e28b9008a6636bde82a006892c"
    assert PINNED_SOURCE_SHA256 == "9f31829434a4713b4a27bca61159999668004bddc7bde844b0766c824d06f7cc"
    assert PINNED_PROTOCOL_SHA256 == "6dd960f1a7eb2076db75bb98cc7d654a25a65837e3d515e5a5d39342d537ca40"


@pytest.fixture(scope="module")
def records():
    return dataset.load_dataset()


@pytest.fixture
def bundle_copy(tmp_path):
    copied = tmp_path / "GSM8K"
    shutil.copytree(BUNDLE, copied, ignore=shutil.ignore_patterns("__pycache__"))
    return copied


def test_complete_dataset_and_stable_order(records):
    assert len(records) == 1319
    assert [row["id"] for row in records] == [
        f"gsm8k/main/test/{index:04d}" for index in range(1319)
    ]
    assert records[0]["answer"] == "18"
    assert records[0]["question"].startswith("Janet’s ducks lay 16 eggs per day.")
    assert records[-1]["answer"] == "14"
    assert dataset.load_dataset() == records


def test_train_only_five_shot_and_gold_separation(records):
    train = [json.loads(line) for line in (BUNDLE / "data/train.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (BUNDLE / "data/test.jsonl").read_text().splitlines()]
    prefix = records[0]["prompt"].rsplit("\n\nQuestion: ", 1)[0]
    for row in train[:5]:
        assert f"Question: {row['question']}\nAnswer: {row['answer']}" in prefix
    assert train[5]["answer"] not in prefix
    for record, original in zip(records, test):
        assert record["question"] == original["question"]
        assert record["reference_solution"] == original["answer"]
        assert record["prompt"] == prefix + f"\n\nQuestion: {original['question']}\nAnswer:"
        assert original["answer"] not in record["prompt"]
        assert record["prompt"].count("\nAnswer:") == 6
    assert "####" in prefix


@pytest.mark.parametrize(
    "prediction,answer,expected",
    [
        ("Reasoning.\n#### 1,234", "1234", 1.0),
        ("#### -2.5", "-2.5", 1.0),
        ("#### 18", "Steps\n#### 18", 1.0),
        ("#### 18.0", "18", 0.0),
        ("#### 018", "18", 0.0),
        ("18", "18", 0.0),
        ("####18", "18", 0.0),
        ("#### 17\n#### 18", "18", 0.0),
        ("#### 18 is the answer", "18", 1.0),
        ("No solution", "18", 0.0),
    ],
)
def test_original_numeric_string_scoring(prediction, answer, expected):
    assert dataset.score(prediction, answer) == expected


@pytest.mark.parametrize("answer", ["", "invalid", "No marker or number", "[invalid]"])
def test_invalid_reference_rejected(answer):
    with pytest.raises(ValueError, match="reference"):
        dataset.score("#### 18", answer)


def test_validation_covers_full_bundle(records):
    report = dataset.validate_bundle()
    assert report["train_rows"] == 7473
    assert report["test_rows"] == 1319
    assert report["validated_prompts"] == 1319
    assert report["revision"] == "3101c7d5072418e28b9008a6636bde82a006892c"
    assert report["prompt_sha256"] == hashlib.sha256(
        "".join(row["prompt"] + "\n" for row in records).encode("utf-8")
    ).hexdigest()
    assert len(report["protocol_sha256"]) == 64


@pytest.mark.parametrize("filename", ["data/train.jsonl", "data/test.jsonl", "LICENSE"])
def test_original_byte_tamper_rejected(bundle_copy, filename):
    path = bundle_copy / filename
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        dataset.load_dataset(bundle_copy)


def test_rehashed_source_and_data_tamper_rejected(bundle_copy):
    path = bundle_copy / "data/test.jsonl"
    contents = path.read_bytes().replace(b"16 eggs per day", b"17 eggs per day", 1)
    assert contents != path.read_bytes()
    path.write_bytes(contents)
    source_path = bundle_copy / "source.json"
    source = json.loads(source_path.read_text())
    source["files"]["data/test.jsonl"]["sha256"] = hashlib.sha256(contents).hexdigest()
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="pin"):
        dataset.validate_bundle(bundle_copy)


@pytest.mark.parametrize("mutation", ["missing", "split", "revision", "rows", "schema"])
def test_source_metadata_tamper_rejected(bundle_copy, mutation):
    path = bundle_copy / "source.json"
    source = json.loads(path.read_text())
    if mutation == "missing":
        del source["files"]["LICENSE"]
    elif mutation == "split":
        source["files"]["data/test.jsonl"]["upstream_path"] = "grade_school_math/data/train.jsonl"
    elif mutation == "revision":
        source["revision"] = "0" * 40
    elif mutation == "rows":
        source["files"]["data/test.jsonl"]["rows"] = 7473
    else:
        source["schema"] = "other"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="pin"):
        dataset.validate_bundle(bundle_copy)


def test_greedy_protocol_is_pinned(bundle_copy):
    path = bundle_copy / "protocol.json"
    protocol = json.loads(path.read_text())
    assert protocol["generation"] == {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "max_new_tokens": 512,
        "use_cache": True,
    }
    assert protocol["few_shot"]["split"] == "train"
    assert protocol["few_shot"]["indices"] == [0, 1, 2, 3, 4]
    assert protocol["evaluation"]["split"] == "test"
    protocol["generation"]["do_sample"] = True
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="protocol"):
        dataset.load_dataset(bundle_copy)


def test_missing_file_is_explicit_error(bundle_copy):
    (bundle_copy / "data/test.jsonl").unlink()
    with pytest.raises(ValueError, match="data/test.jsonl"):
        dataset.validate_bundle(bundle_copy)


def test_cli_is_offline_and_cwd_independent(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(BUNDLE / "dataset.py"), "validate"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["validated_prompts"] == 1319


def test_package_module_cli_from_another_cwd(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "experiments.quality.GSM8K.dataset", "validate"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["validated_prompts"] == 1319
    assert report["prompt_sha256"] == "b9c9766a403fde5512212c66a9b853683e5af8a896ecf491808d5d8eb92c23e1"


def test_cli_rejects_invalid_bundle(bundle_copy, tmp_path):
    (bundle_copy / "LICENSE").unlink()
    result = subprocess.run(
        [sys.executable, "-I", str(BUNDLE / "dataset.py"), "validate", "--bundle-dir", str(bundle_copy)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "LICENSE" in result.stderr
