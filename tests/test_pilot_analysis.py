import copy
import unittest
from scripts.eval_pilot.analyze import summarize


class SummaryTests(unittest.TestCase):
    def rows(self):
        rows = []
        for length in (1024, 4096, 8192):
            for method, group in [("fp16", None), *(("k2v2", g) for g in (8,16,32,64,128))]:
                for f in range(10):
                    rows.append({"type":"attempt_result", "formal":True, "technical_valid":True,
                        "request_key":f"{length}:{method}:{group}:{f}", "family_id":str(f),
                        "condition":{"name":method,"group":group}, "input_length":length,
                        "score":0 if f == 0 else 1, "n":1 if f == 0 else 4,
                        "termination_reason":"eos", "wall_s":3,
                        "energy":{"request_j":10,"decode_j_per_token":None if f == 0 else 2},
                        "latencies":{"ttft_s":1,"tpot_s":None if f == 0 else .2,"request_s":2}})
        return rows

    def test_counts_do_not_treat_180_as_independent_items_or_drop_score_zero(self):
        result = summarize(self.rows())
        self.assertTrue(result["complete"])
        self.assertEqual(result["independent_semantic_families"],10)
        self.assertEqual(result["cells"][0]["mean_score"],.9)
        self.assertEqual(result["cells"][0]["n1_requests"],1)
        self.assertEqual(result["cells"][0]["mean_decode_j_per_token"],2)

    def test_failure_is_retained_but_not_counted_as_success(self):
        rows=self.rows(); failed=copy.deepcopy(rows.pop());failed["technical_valid"]=False
        result=summarize([*rows,failed])
        self.assertFalse(result["complete"])
        self.assertEqual(result["valid_formal_requests"],179)
        self.assertEqual(result["technical_failed_attempts"],1)

    def test_duplicate_success_is_rejected(self):
        rows=self.rows()
        with self.assertRaises(ValueError):summarize([*rows,rows[0]])

    def test_every_cell_requires_each_of_the_same_ten_families(self):
        rows = self.rows()
        rows[0]["family_id"] = rows[1]["family_id"]
        result = summarize(rows)
        self.assertFalse(result["complete"])
        self.assertEqual(result["valid_formal_requests"], 180)
        self.assertEqual(result["independent_semantic_families"], 10)
        self.assertEqual(result["cells"][0]["families"], 9)

    def test_unfinished_formal_attempts_are_separate_from_terminal_failures(self):
        rows = self.rows()
        rows[0]["attempt"] = 2
        key = rows[0]["request_key"]
        rows.extend([
            {"type": "attempt_start", "formal": True, "request_key": key, "attempt": 1},
            {"type": "attempt_start", "formal": True, "request_key": key, "attempt": 2},
            {"type": "attempt_start", "formal": False, "request_key": "smoke", "attempt": 1}])
        result = summarize(rows)
        self.assertTrue(result["complete"])
        self.assertEqual(result["unfinished_attempts"], 1)
        self.assertEqual(result["technical_failed_attempts"], 0)
