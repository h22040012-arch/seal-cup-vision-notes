"""All values below are invented, with no model or camera connection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seal_vision.policy import Detection, decide, match_result

rows = [Detection("front", 0.91, (10, 20, 90, 110)),
        Detection("back", 0.12, (12, 22, 88, 108))]
result = decide(rows, threshold=0.65)  # Illustrative, not a deployment setting.
print("synthetic matching frame:", match_result("request-a", "frame-a", "request-a", "frame-a", result))
print("synthetic stale response:", match_result("request-b", "frame-b", "request-a", "frame-a", result))
