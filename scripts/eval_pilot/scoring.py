"""Unrounded per-item RULER NIAH ``string_match_all`` scores.

The pinned NVIDIA implementation averages case-insensitive substring hits.
Only its *batch reporting* multiplies by 100 and rounds to two decimals;
the pilot retains the underlying item fraction, as the existing Quality LUT did.
"""

RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
SCORING_SOURCE = f"https://github.com/NVIDIA/RULER/blob/{RULER_COMMIT}/scripts/eval/synthetic/constants.py"
SCORING_SOURCE_SHA256 = "6740467c17b8dc06b6b30f4f97e54ce8de81db0dd879f1538d0b6b5727f4bd5f"


def score_prediction(prediction: str, answers: list[str]) -> float:
    """Return the official reference-hit fraction without token normalization.

    Surrounding prose and substring matches intentionally count. Duplicate
    references retain their official weight; empty references are rejected as
    invalid pilot data rather than silently awarding a hit.
    """
    if not isinstance(prediction, str):
        raise ValueError("prediction must be a string")
    if not isinstance(answers, list) or not answers or any(
        not isinstance(answer, str) or not answer for answer in answers
    ):
        raise ValueError("answers must be a nonempty list of nonempty strings")
    lowered = prediction.lower()
    return sum(answer.lower() in lowered for answer in answers) / len(answers)
