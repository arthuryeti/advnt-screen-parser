from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from . import uitree
from .icon_detector import DEFAULT_HF_REPO, IconDetector

PROVIDER = "rapidocr"
ENGINE = "onnxruntime"
OCR_MODEL = "PP-OCRv6-tiny"
ICON_MODEL = "advnt-icons-v3-yolo11s"


def create_ocr_engine(
    model_dir: str | Path | None = None,
    threads: int | None = None,
) -> Any:
    """Create a CPU RapidOCR (PP-OCRv6-tiny) engine."""
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    resolved_dir = (
        Path(model_dir)
        if model_dir is not None
        else Path(
            os.environ.get(
                "OCR_MODEL_DIR",
                str(Path.home() / ".cache" / "advnt-screen-parser" / "rapidocr"),
            )
        )
    )
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_threads = (
        max(1, int(os.environ.get("OCR_INTRA_OP_THREADS", "2")))
        if threads is None
        else max(1, int(threads))
    )
    return RapidOCR(
        params={
            "Global.log_level": "warning",
            "Global.model_root_dir": str(resolved_dir),
            "Global.return_word_box": True,
            "Global.text_score": 0.45,
            "Global.use_cls": False,
            "EngineConfig.onnxruntime.intra_op_num_threads": resolved_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.EN,
            "Det.model_type": ModelType.TINY,
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.EN,
            "Rec.model_type": ModelType.TINY,
            "Rec.ocr_version": OCRVersion.PPOCRV6,
        }
    )


class ScreenParser:
    """End-to-end iOS screen parser: RapidOCR (PP-OCRv6-tiny) + Advnt v3 IconDetector + UI tree."""

    def __init__(
        self,
        *,
        icon_model_path: str | Path | None = None,
        ocr_model_dir: str | Path | None = None,
        repo_id: str = DEFAULT_HF_REPO,
        conf_threshold: float | None = None,
        iou_threshold: float = 0.45,
        threads: int | None = None,
        enable_ocr: bool = True,
        auto_download: bool = True,
    ) -> None:
        self._enable_ocr = enable_ocr
        self._ocr = (
            create_ocr_engine(model_dir=ocr_model_dir, threads=threads)
            if enable_ocr
            else None
        )
        self._icons = IconDetector(
            model_path=icon_model_path,
            repo_id=repo_id,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            threads=threads,
            auto_download=auto_download,
        )
        self._lock = threading.Lock()

    @property
    def icons(self) -> IconDetector:
        return self._icons

    def warm_up(self) -> None:
        if self._ocr is not None:
            image = np.full((64, 128, 3), 255, dtype=np.uint8)
            with self._lock:
                self._ocr(
                    image,
                    use_det=True,
                    use_cls=False,
                    use_rec=True,
                    return_word_box=True,
                )
        self._icons.warm_up()

    def parse_file(self, image_path: str | Path) -> dict[str, Any]:
        image_bytes = Path(image_path).read_bytes()
        return self.parse_bytes(image_bytes)

    def parse_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("Input bytes are not a valid JPEG or PNG image.")
        return self.parse_image(image)

    def parse_image(self, image_bgr: np.ndarray) -> dict[str, Any]:
        height, width = image_bgr.shape[:2]

        words: list[dict[str, Any]] = []
        ocr_ms = 0.0
        if self._ocr is not None:
            t_ocr = time.perf_counter()
            with self._lock:
                result = self._ocr(
                    image_bgr,
                    use_det=True,
                    use_cls=False,
                    use_rec=True,
                    return_word_box=True,
                )
            ocr_ms = round((time.perf_counter() - t_ocr) * 1_000, 1)
            words = _normalized_words(result, width, height)

        t_icons = time.perf_counter()
        icons = self._icons.detect(image_bgr)
        icons_ms = round((time.perf_counter() - t_icons) * 1_000, 1)

        t_tree = time.perf_counter()
        elements = uitree.build(width, height, words, image_bgr, icons=icons)
        uitree_ms = round((time.perf_counter() - t_tree) * 1_000, 1)

        return {
            "provider": PROVIDER if self._ocr is not None else "icons-only",
            "status": "ready",
            "engine": ENGINE,
            "ocrModel": OCR_MODEL if self._ocr is not None else None,
            "iconModel": ICON_MODEL,
            "width": width,
            "height": height,
            "ocrMs": ocr_ms,
            "iconsMs": icons_ms,
            "uitreeMs": uitree_ms,
            "totalMs": round(ocr_ms + icons_ms + uitree_ms, 1),
            "text": " ".join(word["text"] for word in words),
            "words": words,
            "icons": icons,
            "elements": elements,
        }


def _normalized_words(result: Any, width: int, height: int) -> list[dict[str, Any]]:
    words = [
        _word_payload(text, confidence, box, width, height)
        for text, confidence, box in _word_results(result)
    ]
    words = [word for word in words if word is not None]

    if not words:
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is not None and texts is not None and scores is not None:
            words = [
                _word_payload(text, confidence, box, width, height)
                for box, text, confidence in zip(boxes, texts, scores)
            ]
            words = [word for word in words if word is not None]

    return sorted(
        words,
        key=lambda word: (
            round((word["boxPct"][1] + word["boxPct"][3]) / 2, 1),
            word["boxPct"][0],
        ),
    )[:1_000]


def _word_results(result: Any) -> Iterable[tuple[str, float, Any]]:
    for line in getattr(result, "word_results", ()) or ():
        if _is_word_result(line):
            yield line
            continue
        for word in line or ():
            if _is_word_result(word):
                yield word


def _is_word_result(value: Any) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 3
        and isinstance(value[0], str)
        and value[2] is not None
    )


def _word_payload(
    text: Any, confidence: Any, box: Any, width: int, height: int
) -> dict[str, Any] | None:
    normalized_text = str(text).strip()[:256]
    if not normalized_text:
        return None

    points = np.asarray(box, dtype=float).reshape(-1, 2)
    if points.size == 0:
        return None
    left = _percent(float(points[:, 0].min()), width)
    top = _percent(float(points[:, 1].min()), height)
    right = _percent(float(points[:, 0].max()), width)
    bottom = _percent(float(points[:, 1].max()), height)
    if left >= right or top >= bottom:
        return None

    return {
        "text": normalized_text,
        "confidence": round(min(1.0, max(0.0, float(confidence))), 4),
        "boxPct": [left, top, right, bottom],
    }


def _percent(value: float, dimension: int) -> float:
    return round(min(100.0, max(0.0, value / dimension * 100)), 2)
