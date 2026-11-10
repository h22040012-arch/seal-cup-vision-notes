"""CPU inference for the included high/low seal-cup ONNX models."""
import argparse
import configparser
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent


class Detector:
    def __init__(self, station, root=ROOT):
        if station not in {"high", "low"}:
            raise ValueError("station must be high or low")
        self.station = station
        config = configparser.ConfigParser()
        config.read(Path(root) / "profiles.ini", encoding="utf-8-sig")
        self.profile = config[station.title()]
        model = Path(root) / self.profile["Model"].replace("\\", "/")
        if hashlib.sha256(model.read_bytes()).hexdigest() != self.profile["SHA256"]:
            raise ValueError("model SHA256 mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(self.profile["Threads"])
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        expected = [1, 3, 640, 640] if station == "high" else [1, 3, 3036, 4024]
        if self.session.get_inputs()[0].shape != expected:
            raise ValueError("unexpected model input shape")

    def preprocess(self, image):
        if image.dtype != np.uint8 or image.shape[:2] != (3036, 4024):
            raise ValueError("expected original 4024x3036 uint8 image")
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError("unsupported image format")
        if self.station == "high":
            x1, y1, x2, y2 = map(int, self.profile["ROI"].split(","))
            crop = rgb[y1:y2, x1:x2]
            scale = min(640 / crop.shape[1], 640 / crop.shape[0])
            width, height = round(crop.shape[1] * scale), round(crop.shape[0] * scale)
            dx, dy = (640 - width) / 2, (640 - height) / 2
            resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.copyMakeBorder(resized, round(dy - .1), round(dy + .1),
                                     round(dx - .1), round(dx + .1), cv2.BORDER_CONSTANT,
                                     value=(114, 114, 114))
            return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
        # Low model includes ROI and letterbox; do not crop or resize externally.
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        return tensor * np.float32(1.0 / 255.0)

    def inspect(self, path):
        image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("image decoding failed")
        tensor = self.preprocess(image)
        output = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0]
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 6 or not output.shape[1]:
            raise ValueError("unexpected output shape")
        if not np.isfinite(output).all():
            raise ValueError("nonfinite model output")
        rows = output[0]
        best = rows[int(np.argmax(rows[:, 4]))]  # first row on exact ties
        score, label = float(best[4]), float(best[5])
        direction = "no_detection"
        if score >= float(self.profile["Candidate"]):
            if score > 1 or abs(label - round(label)) > 1e-5 or round(label) not in (0, 1):
                raise ValueError("invalid winning score or class")
            direction = "front" if round(label) == 0 else "back"
        return {"station": self.station, "result": "OK" if direction == "front" else "NG",
                "direction": direction, "score": score, "candidate_rows": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", choices=["high", "low"], required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(Detector(args.station).inspect(args.image), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"result": "NG", "direction": "error", "error": str(error)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
