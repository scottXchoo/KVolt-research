"""CPU-only paired RULER pilot preparation; no model weights or downloads.

Execute the existing pinned NVIDIA generator functions unchanged. Preserve
basic 4-fact and hard N40-A4 definitions and the meta-llama3 answer prefix.
An explicit, bounded neutral context suffix closes the final token gap.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
import uuid

from .scoring import RULER_COMMIT, SCORING_SOURCE, SCORING_SOURCE_SHA256

TASKS = ("niah_multiquery", "niah_multivalue")
LENGTHS = (1024, 4096, 8192)
COHORTS = ("basic60", "hard10", "hard30")
TOKENIZER_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
FACT = re.compile(r"One of the special magic (\w+) for ([^\n]+?) is: ([^\s.]+)\.")
QUERY_MARKER = "\nWhat are all the special magic"
FUNCTIONS = ("generate_random_number", "generate_random_word", "generate_random_uuid", "generate_random", "generate_input_output")
PINNED_FILES = {
    "scripts/data/synthetic/niah.py": "e9cada0a7660d274fe73a1338a90a7087e17b630169f1aaf14a8d3221c6805b5",
    "scripts/data/synthetic/constants.py": "6296e901d495ec6200dc3f68993ea13d8282e3c0dbe1a8c47967f111105d1fde",
    "scripts/data/template.py": "2e82d85152212136fffcd5624b158a2c75fd5036ec1c537f4b21eeb78fd18069",
    "scripts/eval/synthetic/constants.py": SCORING_SOURCE_SHA256,
}


def object_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def task_config(cohort, task):
    if cohort not in COHORTS or task not in TASKS:
        raise ValueError("Unknown cohort or RULER task")
    facts = 4 if cohort == "basic60" else 40
    mq = task == TASKS[0]
    return {"type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "numbers",
            "num_needle_k": facts if mq else facts // 4,
            "num_needle_q": 4 if mq else 1, "num_needle_v": 1 if mq else 4}


def family_plan(seed):
    """Assign 6/1/3 before model outputs; task-level hard10 is necessarily 1/0."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    hard10_task = int(object_hash(["kvolt-pilot-cohort-assignment-v1", seed]), 16) % 2
    plan = []
    for ti, task in enumerate(TASKS):
        for ordinal in range(5):
            cohort = "basic60" if ordinal < 3 else ("hard10" if ti == hard10_task and ordinal == 3 else "hard30")
            plan.append({"family_id": f"ruler-pilot-s{seed}-{task}-q{ordinal:02d}",
                         "task": task, "ordinal": ordinal, "cohort": cohort,
                         "difficulty": "basic" if cohort == "basic60" else "hard",
                         "task_config": task_config(cohort, task)})
    return plan


def empty_inventory():
    return {name: set() for name in ("family_ids", "item_ids", "keys", "values", "seeds", "input_hashes", "prompt_hashes", "answer_sets")}


def discover_prior_runs(artifacts_root, *, max_paths=256):
    """Bounded depth-four scan of RULER input files only, never output traces."""
    root = Path(artifacts_root)
    paths = sorted(p for cohort in root.glob("ruler-quality*") if cohort.is_dir()
                   for pattern in ("inputs.json", "*/inputs.json", "*/*/inputs.json", "*/*/*/inputs.json")
                   for p in cohort.glob(pattern))
    if len(paths) > max_paths:
        raise ValueError("Prior archive scan exceeded bounded input count")
    if not paths:
        raise ValueError("No prior RULER inputs discovered")
    return [p.parent for p in paths]


def inventory_runs(runs, *, max_runs=256, allow_unsealed=False):
    """Follow prior-run lineage; explicitly label allowed unsealed input copies."""
    inventory, receipts, seen, queue = empty_inventory(), [], set(), list(map(Path, runs))
    reference_keys = {"source_run", "source_run_dir", "parent_run", "collection_dir", "run_dir"}

    def references(value):
        if isinstance(value, dict):
            for key, entry in value.items():
                if key in reference_keys and isinstance(entry, str):
                    if not entry.startswith("/"):
                        raise ValueError(f"Relative prior input lineage is not auditable: {entry}")
                    target = Path(entry)
                    # Some historical /home paths moved to the same SSD repo.
                    if not (target / "inputs.json").is_file() and entry.startswith("/home/agx3/kvolt/"):
                        target = Path(entry.replace("/home/agx3/kvolt/", "/mnt/kvolt-data/kvolt/", 1))
                    if not (target / "inputs.json").is_file():
                        raise ValueError(f"Unresolved prior input lineage: {entry}")
                    queue.append(target)
                if isinstance(entry, (dict, list)):
                    references(entry)
        elif isinstance(value, list):
            for entry in value:
                references(entry)

    while queue:
        run = queue.pop().resolve()
        if run in seen:
            continue
        seen.add(run)
        if len(seen) > max_runs:
            raise ValueError("Prior-run lineage exceeded bounded inventory")
        inputs_path, seal_path = run / "inputs.json", run / "manifest.json"
        seal = json.loads(seal_path.read_text()) if seal_path.is_file() else {}
        declared_input_hash = seal.get("files_sha256", {}).get("inputs.json")
        if declared_input_hash is None and not allow_unsealed:
            raise ValueError(f"Prior inputs have no manifest seal: {run}")
        if declared_input_hash is not None and declared_input_hash != file_hash(inputs_path):
            raise ValueError(f"Prior inputs do not match manifest seal: {run}")
        rows = json.loads(inputs_path.read_text())
        if isinstance(rows, dict):
            rows = rows["items"]
        receipt = {"run_dir": str(run), "rows": len(rows), "files_sha256": {},
                   "input_seal": "verified" if declared_input_hash is not None else "unsealed_copy_hashed_only"}
        for name in ("inputs.json", "manifest.json", "spec.json", "input-provenance.json", "reused-result-refs.json"):
            path = run / name
            if path.is_file():
                declared = seal.get("files_sha256", {}).get(name)
                if declared is not None and file_hash(path) != declared:
                    raise ValueError(f"Prior lineage file does not match manifest seal: {path}")
                receipt["files_sha256"][name] = file_hash(path)
                if name not in ("inputs.json", "manifest.json"):
                    references(json.loads(path.read_text()))
        for row in rows:
            text = row.get("input_text", row.get("prompt", ""))
            if not text:
                raise ValueError(f"Prior raw prompt missing: {run}")
            facts = FACT.findall(text)
            if not facts:
                raise ValueError(f"Prior RULER facts missing: {run}")
            inventory["family_ids"].add(row.get("family_id", row.get("item_id", row.get("id"))))
            inventory["item_ids"].add(row.get("item_id", row.get("id")))
            inventory["keys"].update(k for _, k, _ in facts)
            inventory["values"].update(v for _, _, v in facts)
            inventory["seeds"].add(row["generator_seed"])
            inventory["input_hashes"].add(object_hash(row["input_ids"]))
            inventory["prompt_hashes"].add(sha256(text.encode()).hexdigest())
            inventory["answer_sets"].add(tuple(sorted(row["answers"])))
        receipts.append(receipt)
    if not receipts:
        raise ValueError("Prior cohorts are required for overlap audit")
    return inventory, sorted(receipts, key=lambda r: r["run_dir"])


def overlap_reasons(family, facts, variants, inventory):
    checks = {"family_id": family["family_id"] in inventory["family_ids"],
              "generator_seed": family["generator_seed"] in inventory["seeds"],
              "fact_keys": bool({k for _, k, _ in facts} & inventory["keys"]),
              "fact_values": bool({v for _, _, v in facts} & inventory["values"])}
    for row in variants:
        checks["item_id"] = checks.get("item_id", False) or row["id"] in inventory["item_ids"]
        checks["token_hash"] = checks.get("token_hash", False) or row["input_hash"] in inventory["input_hashes"]
        checks["prompt_hash"] = checks.get("prompt_hash", False) or row["prompt_sha256"] in inventory["prompt_hashes"]
        checks["answer_set"] = checks.get("answer_set", False) or tuple(sorted(row["answers"])) in inventory["answer_sets"]
    return sorted(name for name, hit in checks.items() if hit)


def exact_length(text, tokenizer, length, *, max_padding_words=32):
    """Insert bounded neutral punctuation into context; keep full prefix/query.

    Never truncate IDs, needles, queries, chat delimiters, or answer prefixes.
    This explicit adaptation makes requested L exact instead of old L-8..L.
    """
    if type(length) is not int or length < 1:
        raise ValueError("length must be a positive integer")
    if QUERY_MARKER not in text:
        raise ValueError("Official query boundary missing")
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if len(ids) > length:
        raise ValueError("Complete prompt exceeds exact length")
    if len(ids) == length:
        return {"text": text, "input_ids": ids, "padding": ""}
    before, after = text.rsplit(QUERY_MARKER, 1)
    for count in range(1, max_padding_words + 1):
        padding = " ." * count
        candidate = before + padding + QUERY_MARKER + after
        tokens = list(tokenizer.encode(candidate, add_special_tokens=False))
        if len(tokens) == length:
            return {"text": candidate, "input_ids": tokens, "padding": padding}
    raise ValueError("No exact full-prompt token length within bounded neutral padding")


def _literal(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing pinned assignment {name}")


def _generator(source, assets):
    manifest = json.loads((source / "source-manifest.json").read_text())
    if manifest["commit"] != RULER_COMMIT:
        raise ValueError("RULER revision changed")
    for name, expected in manifest["files"].items():
        if file_hash(source / name) != expected["sha256"]:
            raise ValueError(f"Official source seal changed: {name}")
    for name, expected in PINNED_FILES.items():
        if file_hash(source / name) != expected:
            raise ValueError(f"Pinned official source changed: {name}")
    asset_manifest = json.loads((assets / "assets-manifest.json").read_text())
    if not asset_manifest.get("complete") or asset_manifest["ruler_commit"] != RULER_COMMIT:
        raise ValueError("Frozen corpus incomplete or wrong revision")
    dependencies = json.loads((assets / "dependency-manifest.json").read_text())
    if file_hash(assets / "dependency-manifest.json") != asset_manifest["dependency_manifest_sha256"]:
        raise ValueError("Dependency manifest changed")
    for name, expected in dependencies["files"].items():
        if file_hash(assets / name) != expected:
            raise ValueError(f"Frozen dependency changed: {name}")
    if file_hash(assets / "PaulGrahamEssays.json") != asset_manifest["corpus_sha256"]:
        raise ValueError("Frozen corpus changed")
    sys.path.insert(0, str(assets / "deps"))
    import nltk
    import numpy as np
    import wonderwords
    nltk.data.path.insert(0, str(assets / "nltk_data"))
    nltk.data.find("tokenizers/punkt_tab/english/")
    nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
    adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
    text = (source / "scripts/data/synthetic/niah.py").read_text()
    nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    functions = "\n\n".join(ast.get_source_segment(text, nodes[name]) for name in FUNCTIONS) + "\n"
    base = _literal(source / "scripts/data/synthetic/constants.py", "TASKS")["niah"]
    model_template = _literal(source / "scripts/data/template.py", "Templates")["meta-llama3"]
    template = model_template.format(task_template=base["template"]) + base["answer_prefix"]
    return {"manifest": manifest, "assets": asset_manifest, "functions": functions, "template": template,
            "base": base, "words": sorted(set(f"{a}-{n}" for a in adjs for n in nouns)),
            "haystack": re.sub(r"\s+", " ", json.loads((assets / "PaulGrahamEssays.json").read_text())["text"]).split(" "),
            "sent_tokenize": nltk.sent_tokenize,
            "DEPTHS": list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int)),
            "dependencies": {m.__name__: {"version": importlib.metadata.version(m.__name__), "module_sha256": file_hash(m.__file__)} for m in (nltk, np, wonderwords)}}


def _renderer(generator, config, seed):
    namespace = {"args": SimpleNamespace(**config, random_seed=seed, template=generator["template"]),
                 "words": generator["words"], "haystack": generator["haystack"], "DEPTHS": generator["DEPTHS"],
                 "sent_tokenize": generator["sent_tokenize"], "uuid": uuid,
                 "needle": "One of the special magic {type_needle_v} for {key} is: {value}."}
    exec(compile(generator["functions"], "pinned-niah-functions.py", "exec"), namespace)

    def render(count):
        class ItemRandom:
            Random = staticmethod(random.Random)

            def __getattr__(self, name):
                return getattr(self.rng, name)

        rng = ItemRandom()
        rng.rng = random.Random(seed)
        namespace["random"] = rng
        return namespace["generate_input_output"](count)
    return render


def _prepare_length(render, tokenizer, length):
    cache = {}

    def evaluate(count):
        if count not in cache:
            if len(cache) >= 64:
                raise ValueError("Bounded word search exceeded 64 renders")
            text, answers = render(count)
            cache[count] = (text, answers, len(tokenizer.encode(text, add_special_tokens=False)))
        return cache[count]

    if evaluate(0)[2] > length:
        raise ValueError("Complete zero-haystack prompt exceeds length")
    low, high = 0, max(500, length)
    for _ in range(8):
        if evaluate(high)[2] > length:
            break
        low, high = high, high * 2
    else:
        raise ValueError("Could not bracket bounded word search")
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if evaluate(mid)[2] <= length:
            best, low = mid, mid + 1
        else:
            high = mid - 1
    candidates = [(evaluate(w)[2], w) for w in range(max(0, best - 8), best + 9) if evaluate(w)[2] <= length]
    for _, count in sorted(candidates, reverse=True):
        text, answers, _ = evaluate(count)
        try:
            result = exact_length(text, tokenizer, length)
        except ValueError:
            continue
        return {**result, "answers": answers, "haystack_words": count, "length_search_evaluations": len(cache)}
    raise ValueError("No exact-length complete prompt")


def validate_inputs(document, tokenizer=None):
    if document.get("schema") != "kvolt.eval_pilot.inputs.v1":
        raise ValueError("Wrong pilot input schema")
    rows = document["items"]
    if len(rows) != 30 or len({r["id"] for r in rows}) != 30:
        raise ValueError("Pilot needs 30 distinct item IDs")
    if len({r["input_hash"] for r in rows}) != 30 or len({r["prompt_sha256"] for r in rows}) != 30:
        raise ValueError("Pilot prompts and token hashes must be distinct")
    families = {}
    for row in rows:
        ids = row["input_ids"]
        if row["length"] not in LENGTHS or len(ids) != row["length"] or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("Pilot input length must be exact")
        if object_hash(ids) != row["input_hash"] or sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Frozen prompt or token hash changed")
        if tokenizer is not None and list(tokenizer.encode(row["prompt"], add_special_tokens=False)) != ids:
            raise ValueError("Prompt tokenization differs from frozen IDs")
        facts = sorted(FACT.findall(row["prompt"]))
        cfg = task_config(row["cohort"], row["task"])
        if row["difficulty"] != ("basic" if row["cohort"] == "basic60" else "hard"):
            raise ValueError("Cohort and difficulty disagree")
        if row["task_config"] != cfg or len(facts) != cfg["num_needle_k"] * cfg["num_needle_v"]:
            raise ValueError("RULER fact count or configuration changed")
        keys = Counter(k for _, k, _ in facts)
        if len(keys) != cfg["num_needle_k"] or set(keys.values()) != {cfg["num_needle_v"]} or any(t != "numbers" for t, _, _ in facts):
            raise ValueError("RULER key/value structure changed")
        query = re.search(r"\nWhat are all the special magic numbers for (.+?) mentioned in the provided text\?", row["prompt"])
        if query is None:
            raise ValueError("Official query changed")
        query_text = query.group(1)
        queried_keys = query_text.replace(", and ", ", ").split(", ")
        if len(queried_keys) != cfg["num_needle_q"] or len(set(queried_keys)) != len(queried_keys) or {v for _, k, v in facts if k in queried_keys} != set(row["answers"]):
            raise ValueError("References differ from queried keys")
        if len(row["answers"]) != 4 or len(set(row["answers"])) != 4 or any(row["prompt"].count(a) != 1 for a in row["answers"]):
            raise ValueError("Reference answers must be unique and appear once")
        expected_prefix = f" The special magic numbers for {query_text} mentioned in the provided text are"
        if row["answer_prefix"] != expected_prefix or not row["prompt"].endswith(expected_prefix):
            raise ValueError("Official answer prefix changed")
        identity = (row["task"], row["cohort"], row["difficulty"], row["generator_seed"], tuple(row["answers"]), tuple(facts), row["answer_prefix"])
        previous, lengths = families.setdefault(row["family_id"], (identity, set()))
        if previous != identity or row["length"] in lengths:
            raise ValueError("Semantic family changed across lengths")
        lengths.add(row["length"])
    if len(families) != 10 or any(lengths != set(LENGTHS) for _, lengths in families.values()):
        raise ValueError("Ten families must each have three length variants")
    counts = Counter((r["task"], r["difficulty"]) for r in rows if r["length"] == 1024)
    if counts != Counter({(t, d): n for t in TASKS for d, n in (("basic", 3), ("hard", 2))}):
        raise ValueError("Task difficulty balance changed")
    if Counter(r["cohort"] for r in rows if r["length"] == 1024) != Counter(dict(zip(COHORTS, (6, 1, 3)))):
        raise ValueError("Global 60/10/30 cohort balance changed")
    return {"items": 30, "independent_families": 10, "exact_lengths": list(LENGTHS)}


def prepare_inputs(output_dir, tokenizer_path, source_dir, assets_dir, cohort_runs, *, seed=20261103,
                   exclude_runs=(), archive_root=None):
    """Prepare only CPU tokenizer/generator inputs; never inspect model outputs."""
    output, source, assets, snapshot = map(Path, (output_dir, source_dir, assets_dir, tokenizer_path))
    if output.exists():
        raise FileExistsError("Frozen output directory must not already exist")
    if set(cohort_runs) != set(COHORTS):
        raise ValueError("Provide all three published cohort run directories")
    if snapshot.resolve().name != TOKENIZER_REVISION:
        raise ValueError("Tokenizer must use the confirmed cached snapshot revision")
    generator = _generator(source, assets)
    discovered = discover_prior_runs(archive_root) if archive_root is not None else []
    inventory, receipts = inventory_runs([*cohort_runs.values(), *exclude_runs, *discovered],
                                         allow_unsealed=archive_root is not None)
    cohort_evidence = {}
    for cohort, run in cohort_runs.items():
        prior_rows = json.loads((Path(run) / "inputs.json").read_text())
        configurations = set()
        for row in prior_rows:
            expected = task_config(cohort, row["task"])
            facts = FACT.findall(row["input_text"])
            key_counts = Counter(k for _, k, _ in facts)
            if (len(facts) != expected["num_needle_k"] * expected["num_needle_v"]
                    or len(key_counts) != expected["num_needle_k"] or set(key_counts.values()) != {expected["num_needle_v"]}
                    or ("task_config" in row and row["task_config"] != expected)):
                raise ValueError(f"Published {cohort} configuration differs from planned pilot")
            configurations.add(object_hash(expected))
        cohort_evidence[cohort] = {"run_dir": str(Path(run).resolve()), "inputs_sha256": file_hash(Path(run) / "inputs.json"), "rows": len(prior_rows), "config_hashes": sorted(configurations)}
    # Tokenizer preparation must not import the CUDA-dependent torch build.
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    plan, items, rejections = family_plan(seed), [], []
    for family_plan_row in plan:
        for attempt in range(128):
            derived_seed = int(object_hash(["kvolt-eval-pilot-new-families-v1", seed, family_plan_row["task"], family_plan_row["ordinal"], attempt])[:16], 16)
            family = {**family_plan_row, "generator_seed": derived_seed, "seed_attempt": attempt}
            render = _renderer(generator, family["task_config"], derived_seed)
            try:
                variants = []
                for length in LENGTHS:
                    prepared = _prepare_length(render, tokenizer, length)
                    text, ids = prepared["text"], prepared["input_ids"]
                    marker = generator["base"]["answer_prefix"].split("{")[0]
                    offset = text.rfind(marker)
                    if offset < 0:
                        raise ValueError("Official answer prefix missing")
                    variants.append({**family, "id": f"{family['family_id']}-L{length}", "length": length,
                        "prompt": text, "input_ids": ids, "answers": prepared["answers"],
                        "input_hash": object_hash(ids), "prompt_sha256": sha256(text.encode()).hexdigest(),
                        "answer_prefix": text[offset:], "haystack_words": prepared["haystack_words"],
                        "length_search_evaluations": prepared["length_search_evaluations"], "exact_length_padding": prepared["padding"],
                        "split": "pilot_excluded_from_future_evaluation", "minimal_answer_tokens": len(tokenizer.encode(", ".join(prepared["answers"]), add_special_tokens=False))})
                facts = sorted(FACT.findall(variants[0]["prompt"]))
                if any(sorted(FACT.findall(r["prompt"])) != facts or r["answers"] != variants[0]["answers"] or r["answer_prefix"] != variants[0]["answer_prefix"] for r in variants):
                    raise ValueError("Family facts/query changed across lengths")
                if any(r["minimal_answer_tokens"] > 112 or len(set(r["answers"])) != 4 or any(r["prompt"].count(a) != 1 for a in r["answers"]) for r in variants):
                    raise ValueError("Invalid references or insufficient answer budget")
                reasons = overlap_reasons(family, facts, variants, inventory)
                if reasons:
                    raise ValueError("Overlap: " + ",".join(reasons))
                break
            except ValueError as error:
                rejections.append({"family_id": family["family_id"], "attempt": attempt, "generator_seed": derived_seed, "reason": str(error)})
        else:
            raise RuntimeError("No structurally valid nonoverlapping family within 128 attempts")
        inventory["family_ids"].add(family["family_id"])
        inventory["keys"].update(k for _, k, _ in facts)
        inventory["values"].update(v for _, _, v in facts)
        inventory["seeds"].add(derived_seed)
        for row in variants:
            inventory["item_ids"].add(row["id"])
            inventory["input_hashes"].add(row["input_hash"])
            inventory["prompt_hashes"].add(row["prompt_sha256"])
            inventory["answer_sets"].add(tuple(sorted(row["answers"])))
        items.extend(variants)
        print(f"prepared families={len(items)//3}/10 inputs={len(items)}/30", flush=True)
    metadata = {"seed": seed, "family_plan": plan, "family_plan_sha256": object_hash(plan),
        "seed_derivation": "int(SHA256(canonical JSON [kvolt-eval-pilot-new-families-v1,seed,task,ordinal,attempt])[:16],16)",
        "lengths": list(LENGTHS), "independent_families": 10, "pilot_excluded_from_future_evaluation": True,
        "model_outputs_used_for_selection": False, "generation_device": "CPU", "model_weights_loaded": False,
        "ruler_commit": RULER_COMMIT, "source_manifest_sha256": file_hash(source / "source-manifest.json"),
        "official_sources": generator["manifest"]["files"], "official_function_sha256": sha256(generator["functions"].encode()).hexdigest(),
        "assets_manifest_sha256": file_hash(assets / "assets-manifest.json"), "assets": generator["assets"],
        "dependencies": generator["dependencies"], "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_files": {p.name: file_hash(p) for p in sorted(snapshot.glob("*")) if p.is_file() and p.suffix in (".json", ".model", ".tiktoken")},
        "tokenizer_versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
        "template_sha256": sha256(generator["template"].encode()).hexdigest(), "add_special_tokens": False,
        "input_length_definition": "complete official meta-llama3 prompt including answer prefix",
        "exact_length_adaptation": "0..32 repetitions of ' .' appended inside context before official query; full prompt retokenized and exact L required",
        "cohort_evidence": cohort_evidence, "cohort_counts": {t: dict(Counter(r["cohort"] for r in plan if r["task"] == t)) for t in TASKS},
        "small_cohort_imbalance": "6 basic60 + 1 hard10 + 3 hard30 globally; hard10 cannot split evenly over two five-family tasks; seed assigns task before outputs. Both hard cohorts use N40-A4.",
        "overlap_audit": {"status": "passed_for_inventoried_lineage", "receipts": receipts, "rejected_attempts": rejections,
                          "archive_scan": {"root": str(archive_root) if archive_root is not None else None,
                                           "max_depth": 4, "discovered_input_files": len(discovered),
                                           "unsealed_copies": [r["run_dir"] for r in receipts if r["input_seal"] != "verified"]},
                          "checks": ["family_id", "item_id", "generator_seed", "fact_keys", "fact_values", "answer_set", "prompt_hash", "token_hash"],
                          "scope": "published cohorts, explicit exclusions, depth-bounded RULER input scan when requested, and recursively referenced inputs; unsealed copies are audited by content hash only; no assertion about unavailable or out-of-depth archives"},
        "scoring": {"metric": "string_match_all", "source": SCORING_SOURCE, "source_sha256": SCORING_SOURCE_SHA256, "range": [0, 1], "rounding": "none per item"},
        "adapter_sha256": file_hash(__file__), "scorer_adapter_sha256": file_hash(Path(__file__).with_name("scoring.py"))}
    document = {"schema": "kvolt.eval_pilot.inputs.v1", "metadata": metadata, "items": items}
    audit = validate_inputs(document, tokenizer)
    output.mkdir(parents=True)
    for name, value in (("inputs.json", document), ("families.json", [{k: v for k, v in r.items() if k not in ("prompt", "input_ids")} for r in items if r["length"] == 1024])):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    manifest = {"schema": "kvolt.eval_pilot.manifest.v1", "audit": audit, "metadata": metadata,
                "files_sha256": {name: file_hash(output / name) for name in ("inputs.json", "families.json")},
                "token_hashes": {r["id"]: r["input_hash"] for r in items}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return document, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", "--model-snapshot", dest="tokenizer_path", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--exclude-run", action="append", default=[])
    parser.add_argument("--archive-root", help="Bounded depth-four RULER inputs scan; include unsealed copies with explicit receipts")
    for cohort in COHORTS:
        parser.add_argument(f"--{cohort}-run", required=True)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args(argv)
    _, manifest = prepare_inputs(args.output_dir, args.tokenizer_path, args.source_dir, args.assets_dir,
                                {c: getattr(args, f"{c}_run") for c in COHORTS}, seed=args.seed,
                                exclude_runs=args.exclude_run, archive_root=args.archive_root)
    print(json.dumps(manifest["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
