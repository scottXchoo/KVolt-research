import unittest
import copy
import json
from pathlib import Path
import tempfile

from scripts.eval_pilot import data
from scripts.eval_pilot.scoring import score_prediction


class WordTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(len(text.split())))


class PilotDataTests(unittest.TestCase):
    def document(self):
        items = []
        for fi, family in enumerate(data.family_plan(20261103)):
            cfg = family["task_config"]
            facts = [(f"key-{fi}-{k}", str(7000000 + fi * 100 + k * cfg["num_needle_v"] + v))
                     for k in range(cfg["num_needle_k"]) for v in range(cfg["num_needle_v"])]
            query_keys = [f"key-{fi}-{k}" for k in range(cfg["num_needle_q"])]
            query = ", ".join(query_keys[:-1]) + ", and " + query_keys[-1] if len(query_keys) > 1 else query_keys[0]
            prefix = f" The special magic numbers for {query} mentioned in the provided text are"
            prompt = " ".join(f"One of the special magic numbers for {k} is: {v}." for k, v in facts)
            prompt += f"\nWhat are all the special magic numbers for {query} mentioned in the provided text?" + prefix
            for length in data.LENGTHS:
                ids = list(range(fi * 10000, fi * 10000 + length))
                variant_prompt = prompt.replace("\nWhat", f" length-{length}\nWhat", 1)
                items.append({**family, "id": f"{family['family_id']}-L{length}", "length": length,
                    "input_ids": ids, "input_hash": data.object_hash(ids), "prompt": variant_prompt,
                    "prompt_sha256": data.sha256(variant_prompt.encode()).hexdigest(), "generator_seed": fi,
                    "answer_prefix": prefix, "answers": [v for k, v in facts if k in query_keys]})
        return {"schema": "kvolt.eval_pilot.inputs.v1", "items": items}

    def test_official_matching_fixtures(self):
        # Official NIAH string_match_all: every reference is an independent
        # substring hit, including inside longer words/numbers. No F1/word split.
        self.assertEqual(score_prediction("Answer: RED, blue.", ["red", "BLUE", "green", "gold"]), .5)
        self.assertEqual(score_prediction("13145 and carpet", ["314", "car"]), 1)
        self.assertEqual(score_prediction("FOO", ["foo", "bar", "baz"]), 1 / 3)
        self.assertEqual(score_prediction("foo-bar", ["foo bar"]), 0)
        self.assertEqual(score_prediction("", ["1234567"]), 0)

    def test_invalid_scoring_inputs(self):
        for answers in ([], [""], [None], "answer"):
            with self.assertRaises(ValueError):
                score_prediction("output", answers)

    def test_cohort_plan_fixed_before_outputs(self):
        plan = data.family_plan(20261103)
        self.assertEqual(plan, data.family_plan(20261103))
        self.assertEqual(len(plan), 10)
        self.assertEqual(len({row["family_id"] for row in plan}), 10)
        self.assertEqual([sum(r["cohort"] == c for r in plan) for c in data.COHORTS], [6, 1, 3])
        for task in data.TASKS:
            rows = [r for r in plan if r["task"] == task]
            self.assertEqual([r["difficulty"] for r in rows].count("basic"), 3)
            self.assertEqual([r["difficulty"] for r in rows].count("hard"), 2)
        self.assertEqual(data.task_config("basic60", data.TASKS[0])["num_needle_k"], 4)
        self.assertEqual(data.task_config("hard10", data.TASKS[0])["num_needle_k"], 40)
        self.assertEqual(data.task_config("hard30", data.TASKS[1])["num_needle_k"], 10)

    def test_exact_length_preserves_query_and_suffix(self):
        text = "context words\nWhat are all the special magic numbers? PREFIX"
        output = data.exact_length(text, WordTokenizer(), 16)
        self.assertEqual(len(output["input_ids"]), 16)
        self.assertTrue(output["text"].endswith("\nWhat are all the special magic numbers? PREFIX"))
        with self.assertRaises(ValueError):
            data.exact_length(text, WordTokenizer(), 1)

    def test_overlap_rejects_facts_and_families(self):
        prior = data.empty_inventory()
        prior["keys"].add("old-key")
        facts = [("numbers", "old-key", "1234567")]
        self.assertTrue(data.overlap_reasons({"family_id": "new", "generator_seed": 2}, facts, [], prior))
        prior = data.empty_inventory()
        prior["family_ids"].add("old")
        self.assertTrue(data.overlap_reasons({"family_id": "old", "generator_seed": 2}, [], [], prior))

    def test_validator_rejects_prefix_and_difficulty_corruption(self):
        document = self.document()
        self.assertEqual(data.validate_inputs(document)["independent_families"], 10)
        for corrupt in ("prefix", "difficulty", "references"):
            changed = copy.deepcopy(document)
            if corrupt == "prefix":
                for row in changed["items"]:
                    row["answer_prefix"] = ""
            elif corrupt == "difficulty":
                for row in changed["items"]:
                    if row["family_id"] in (document["items"][0]["family_id"], document["items"][9]["family_id"]):
                        row["difficulty"] = "hard" if row["difficulty"] == "basic" else "basic"
            else:
                changed["items"][0]["answers"] = ["missing", *changed["items"][0]["answers"][1:]]
            with self.assertRaises(ValueError):
                data.validate_inputs(changed)

    def test_inventory_rejects_relative_lineage_and_seal_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            row = self.document()["items"][0]
            (run / "inputs.json").write_text(json.dumps([row]))
            (run / "input-provenance.json").write_text(json.dumps({"source_run": "parent"}))
            (run / "manifest.json").write_text(json.dumps({"files_sha256": {"inputs.json": data.file_hash(run / "inputs.json")}}))
            with self.assertRaisesRegex(ValueError, "Relative"):
                data.inventory_runs([run])
            (run / "input-provenance.json").write_text("{}")
            seal = {"files_sha256": {name: data.file_hash(run / name) for name in ("inputs.json", "input-provenance.json")}}
            (run / "manifest.json").write_text(json.dumps(seal))
            self.assertEqual(data.inventory_runs([run])[0]["seeds"], {0})
            (run / "input-provenance.json").write_text('{"note":"changed"}')
            with self.assertRaisesRegex(ValueError, "seal"):
                data.inventory_runs([run])

    def test_bounded_archive_scan_records_unsealed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "ruler-quality-example" / "calibration"
            run.mkdir(parents=True)
            (run / "inputs.json").write_text(json.dumps([self.document()["items"][0]]))
            (root / "unrelated").mkdir()
            (root / "unrelated" / "inputs.json").write_text("[]")
            self.assertEqual(data.discover_prior_runs(root), [run])
            with self.assertRaisesRegex(ValueError, "no manifest seal"):
                data.inventory_runs([run])
            _, receipts = data.inventory_runs([run], allow_unsealed=True)
            self.assertEqual(receipts[0]["input_seal"], "unsealed_copy_hashed_only")
            self.assertIn("inputs.json", receipts[0]["files_sha256"])


if __name__ == "__main__":
    unittest.main()
