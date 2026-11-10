"""Teaching policy. Coordinates are already in the model adapter's ROI space."""
from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Decision:
    accepted: bool
    direction: str
    reason: str
    score: float = 0.0


def decide(rows: Iterable[Detection], *, threshold: float) -> Decision:
    """Highest score wins; exact ties retain input order.

    Input validation belongs to this example adapter contract. It is not an
    extra area-based business rejection rule. Empty output means no detection,
    not proof that a physical part is absent.
    """
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    best = None
    for row in rows:
        if row.label not in {"front", "back"}:
            return Decision(False, "unknown", "invalid_model_output")
        if len(row.box) != 4 or not all(math.isfinite(x) for x in (*row.box, row.score)):
            return Decision(False, "unknown", "invalid_model_output")
        x1, y1, x2, y2 = row.box
        if not 0 <= row.score <= 1 or x2 <= x1 or y2 <= y1:
            return Decision(False, "unknown", "invalid_model_output")
        if best is None or row.score > best.score:
            best = row
    if best is None:
        return Decision(False, "unknown", "no_detection")
    if best.score < threshold:
        return Decision(False, "unknown", "below_threshold", best.score)
    return Decision(best.label == "front", best.label, best.label, best.score)


def match_result(expected_request: str, expected_frame: str,
                 response_request: str, response_frame: str,
                 decision: Decision) -> Decision:
    """Call only after transport/schema validation; timeout yields no decision."""
    if not expected_request or not expected_frame or (
        expected_request != response_request or expected_frame != response_frame
    ):
        return Decision(False, "unknown", "result_mismatch")
    return decision
