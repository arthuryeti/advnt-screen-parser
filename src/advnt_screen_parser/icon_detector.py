from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_HF_REPO = "arthuryeti/advnt-screen-parser"
DEFAULT_ONNX_FILENAME = "icons.onnx"
DEFAULT_META_FILENAME = "icons.json"


def download_weights(
    repo_id: str = DEFAULT_HF_REPO,
    filename: str = DEFAULT_ONNX_FILENAME,
    meta_filename: str = DEFAULT_META_FILENAME,
    revision: str | None = None,
) -> tuple[Path, Path | None]:
    """Download ONNX weights and class metadata from Hugging Face Hub (cached locally)."""
    from huggingface_hub import hf_hub_download

    model_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
    meta_path: Path | None = None
    try:
        meta_path = Path(
            hf_hub_download(repo_id=repo_id, filename=meta_filename, revision=revision)
        )
    except Exception:
        meta_path = None
    return model_path, meta_path


def _letterbox(
    image: np.ndarray, size: int
) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = (
        cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if (nw, nh) != (w, h)
        else image
    )
    pad_w = (size - nw) / 2.0
    pad_h = (size - nh) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, (pad_w, pad_h)


class IconDetector:
    """CPU-fast ONNX YOLOv11s detector for 47 iOS & social-app UI icon classes.

    Loads a local ONNX file if ``model_path`` or ``ADVNT_ICON_MODEL_PATH`` /
    ``OCR_ICON_MODEL_PATH`` is provided; otherwise downloads and caches the v3
    weights automatically from Hugging Face Hub (``arthuryeti/advnt-screen-parser``).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        meta_path: str | Path | None = None,
        repo_id: str = DEFAULT_HF_REPO,
        conf_threshold: float | None = None,
        iou_threshold: float = 0.45,
        threads: int | None = None,
        auto_download: bool = True,
    ) -> None:
        env_path = os.environ.get("ADVNT_ICON_MODEL_PATH") or os.environ.get(
            "OCR_ICON_MODEL_PATH"
        )
        resolved_model: Path | None = (
            Path(model_path)
            if model_path is not None
            else Path(env_path)
            if env_path
            else None
        )
        resolved_meta: Path | None = Path(meta_path) if meta_path is not None else None

        if resolved_model is None and auto_download:
            resolved_model, downloaded_meta = download_weights(repo_id=repo_id)
            if resolved_meta is None:
                resolved_meta = downloaded_meta

        self._path = resolved_model
        self._meta_path = resolved_meta
        self._conf = (
            float(os.environ.get("OCR_ICON_CONF", "0.35"))
            if conf_threshold is None
            else float(conf_threshold)
        )
        self._iou = float(iou_threshold)
        self._threads = (
            max(1, int(os.environ.get("OCR_INTRA_OP_THREADS", "2")))
            if threads is None
            else max(1, int(threads))
        )
        self._lock = threading.Lock()
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._imgsz: int = 640
        self._names: dict[int, str] = {}

        if self._path and self._path.is_file():
            self._load(self._path, self._meta_path)

    @property
    def available(self) -> bool:
        return self._session is not None

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    @property
    def imgsz(self) -> int:
        return self._imgsz

    def _load(self, path: Path, meta_path: Path | None = None) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = session.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            self._imgsz = int(shape[2])

        candidate_meta = meta_path if meta_path and meta_path.is_file() else path.with_suffix(".json")
        if candidate_meta.is_file():
            meta = json.loads(candidate_meta.read_text(encoding="utf-8"))
            if isinstance(meta.get("imgsz"), int):
                self._imgsz = meta["imgsz"]
            raw_names = meta.get("names") or {}
            if isinstance(raw_names, dict):
                self._names = {int(k): str(v) for k, v in raw_names.items()}
            elif isinstance(raw_names, list):
                self._names = {i: str(v) for i, v in enumerate(raw_names)}
        self._session = session

    def warm_up(self) -> None:
        if not self._session:
            return
        blank = np.full((128, 64, 3), 255, dtype=np.uint8)
        self.detect(blank)

    def detect_file(self, image_path: str | Path) -> list[dict[str, Any]]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.detect(image)

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        if not self._session or image_bgr is None or image_bgr.size == 0:
            return []
        orig_h, orig_w = image_bgr.shape[:2]
        padded, scale, (pad_w, pad_h) = _letterbox(image_bgr, self._imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(
            np.transpose(rgb, (2, 0, 1))[None, ...], dtype=np.float32
        )
        blob *= 1.0 / 255.0

        with self._lock:
            outputs = self._session.run(None, {self._input_name: blob})

        raw = np.asarray(outputs[0])
        if raw.ndim == 3:
            raw = raw[0]
        # Ultralytics YOLOv11 ONNX exports [4 + nc, N]; transpose to [N, 4 + nc].
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        if raw.shape[1] <= 4:
            return []

        scores = raw[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), cls_ids]
        keep = confs >= self._conf
        if not np.any(keep):
            return []

        boxes_xywh = raw[keep, :4]
        cls_ids = cls_ids[keep]
        confs = confs[keep]

        # Undo letterbox scaling & padding into original pixel space.
        cx = (boxes_xywh[:, 0] - pad_w) / scale
        cy = (boxes_xywh[:, 1] - pad_h) / scale
        bw = boxes_xywh[:, 2] / scale
        bh = boxes_xywh[:, 3] / scale

        x1 = np.clip(cx - bw / 2.0, 0.0, float(orig_w))
        y1 = np.clip(cy - bh / 2.0, 0.0, float(orig_h))
        x2 = np.clip(cx + bw / 2.0, 0.0, float(orig_w))
        y2 = np.clip(cy + bh / 2.0, 0.0, float(orig_h))

        # Per-class NMS via offset trick (standard YOLO NMSBoxes pattern).
        max_dim = float(max(orig_w, orig_h) + 1)
        offsets = cls_ids.astype(np.float32) * max_dim
        nms_boxes = np.stack([x1 + offsets, y1 + offsets, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            confs.astype(float).tolist(),
            self._conf,
            self._iou,
        )
        if len(indices) == 0:
            return []

        flat_idx = np.asarray(indices).reshape(-1)
        detections: list[dict[str, Any]] = []
        for idx in flat_idx:
            ix1, iy1, ix2, iy2 = (
                int(round(x1[idx])),
                int(round(y1[idx])),
                int(round(x2[idx])),
                int(round(y2[idx])),
            )
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            cls = int(cls_ids[idx])
            label = self._names.get(cls, f"icon_{cls}")
            detections.append(
                {
                    "label": label,
                    "confidence": round(float(confs[idx]), 3),
                    "box": [ix1, iy1, ix2, iy2],
                    "boxPct": [
                        round(ix1 / orig_w * 100, 1),
                        round(iy1 / orig_h * 100, 1),
                        round(ix2 / orig_w * 100, 1),
                        round(iy2 / orig_h * 100, 1),
                    ],
                    "centerPct": [
                        round(((ix1 + ix2) / 2.0) / orig_w * 100, 1),
                        round(((iy1 + iy2) / 2.0) / orig_h * 100, 1),
                    ],
                }
            )
        detections.sort(key=lambda d: -d["confidence"])
        return detections
