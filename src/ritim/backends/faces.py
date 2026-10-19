"""Face detection for `inventory`: which frames show the trainer's face, and how large."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class FaceDetector(Protocol):
    def largest_face(self, frame: np.ndarray) -> tuple[float, list[float]] | None:
        """(height fraction, [x1, y1, x2, y2] normalised) of the largest face, or None."""

    def close(self) -> None: ...


class NoFaces:
    def largest_face(self, frame: np.ndarray) -> tuple[float, list[float]] | None:
        return None

    def close(self) -> None:
        pass


class MediaPipeFaces:
    """MediaPipe Tasks BlazeFace (Apache-2.0 code and model). CPU only; ~5 ms per 640 px frame."""

    def __init__(self, model_path: Path, min_confidence: float = 0.5) -> None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing; run infra/bootstrap.sh (models step)")
        opts = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._det: Any = vision.FaceDetector.create_from_options(opts)

    def largest_face(self, frame: np.ndarray) -> tuple[float, list[float]] | None:
        import mediapipe as mp

        h, w = frame.shape[:2]
        result = self._det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
        best = None
        for d in result.detections:
            bb = d.bounding_box
            if best is None or bb.height > best.height:
                best = bb
        if best is None:
            return None
        x1, y1 = max(0, best.origin_x) / w, max(0, best.origin_y) / h
        x2, y2 = min(w, best.origin_x + best.width) / w, min(h, best.origin_y + best.height) / h
        return round(y2 - y1, 4), [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]

    def close(self) -> None:
        self._det.close()


class MockFaces:
    """Tests: a face fills the frame whenever the frame's mean red channel is high."""

    def largest_face(self, frame: np.ndarray) -> tuple[float, list[float]] | None:
        if float(frame[..., 0].mean()) > 150 and float(frame[..., 2].mean()) < 100:
            return 0.5, [0.3, 0.2, 0.7, 0.7]
        return None

    def close(self) -> None:
        pass


def make_face_detector(backend: str, models_root: Path) -> FaceDetector:
    if backend == "mediapipe":
        return MediaPipeFaces(models_root / "mediapipe" / "blaze_face_short_range.tflite")
    if backend == "mock":
        return MockFaces()
    return NoFaces()
