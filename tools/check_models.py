"""Small replay check, not an accuracy benchmark."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import Detector

results = []
for station in ("high", "low"):
    detector = Detector(station)
    for name, expected in (("front", "front"), ("back", "back"), ("empty", "no_detection")):
        result = detector.inspect(ROOT / "samples" / station / (name + ".jpg"))
        results.append(result)
        if result["direction"] != expected:
            raise AssertionError(f"{station}/{name}: {result}")
print(json.dumps({"passed": True, "cases": results}, ensure_ascii=False, indent=2))
