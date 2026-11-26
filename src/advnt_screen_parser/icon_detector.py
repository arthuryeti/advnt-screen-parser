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

_LOW_CANDIDATE_CONF = 0.12
_SOCIAL_PREFIXES = ("ig_", "tt_", "x_", "yt_")
_APP_VOTE_MIN_DETECTIONS = 3
_APP_VOTE_RATIO = 3
_CROSS_APP_SUPPRESS_CONF = 0.65
_TAB_BAR_SLOTS: dict[str, tuple[str, ...]] = {
    "tt_": ("tt_home_tab", "tt_friends_tab", "tt_create", "tt_inbox_tab", "tt_profile_tab"),
    "ig_": ("ig_home_tab", "ig_reels_tab", "ig_messages_tab", "ig_search_tab", "ig_profile_tab"),
    "x_": ("x_home_tab", "x_search_tab", "x_grok_tab", "x_notifications_tab", "x_messages_tab"),
    "yt_": ("yt_home_tab", "yt_reels_tab", "yt_create", "yt_subscriptions_tab", "yt_profile_tab"),
}
_TAB_BAR_MIN_SLOTS = 3
_TAB_BAR_MIN_CY = 0.84
_TAB_BAR_MAX_STD_CY = 0.035
_TAB_BAR_RESCUE_DY = 0.04
_TAB_BAR_RESCUE_IOU = 0.30


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


def _parse_imgsz(value: Any, default: tuple[int, int] = (960, 480)) -> tuple[int, int]:
    if isinstance(value, int) and value > 0:
        return (value, value)
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], int)
        and value[0] > 0
        and value[1] > 0
    ):
        return (int(value[0]), int(value[1]))
    return default


def _parse_class_conf(raw: Any, names: dict[int, str]) -> dict[int, float]:
    if not isinstance(raw, dict):
        return {}
    name_to_cls = {name: cls for cls, name in names.items()}
    parsed: dict[int, float] = {}
    for key, val in raw.items():
        try:
            thr = float(val)
        except (TypeError, ValueError):
            continue
        if not (0.0 < thr <= 1.0):
            continue
        if isinstance(key, int):
            parsed[key] = thr
        elif isinstance(key, str):
            if key in name_to_cls:
                parsed[name_to_cls[key]] = thr
            elif key.isdigit():
                parsed[int(key)] = thr
    return parsed


def _social_prefix(label: str) -> str | None:
    for prefix in _SOCIAL_PREFIXES:
        if label.startswith(prefix):
            return prefix
    return None


def _box_iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = float(iw * ih)
    union = float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / union if union > 0.0 else 0.0


def _letterbox(
    image: np.ndarray, size: int | tuple[int, int]
) -> tuple[np.ndarray, float, tuple[float, float]]:
    target_h, target_w = (size, size) if isinstance(size, int) else size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = (
        cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if (nw, nh) != (w, h)
        else image
    )
    pad_w = (target_w - nw) / 2.0
    pad_h = (target_h - nh) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, (pad_w, pad_h)


class IconDetector:
    """CPU-fast ONNX YOLOv11m detector for 99 iOS & social-app UI icon classes (v5).

    Supports portrait inputs (960x480), per-class calibrated confidence thresholds
    (``class_conf`` in ``icons.json``), cross-app consistency voting, and 5-slot
    bottom tab-bar recovery. Auto-downloads v5 weights from Hugging Face Hub
    (``arthuryeti/advnt-screen-parser``) when no local path is supplied.
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
        self._imgsz: tuple[int, int] = (960, 480)
        self._names: dict[int, str] = {}
        self._class_conf: dict[int, float] = {}

        if self._path and self._path.is_file():
            self._load(self._path, self._meta_path)

    @property
    def available(self) -> bool:
        return self._session is not None

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    @property
    def imgsz(self) -> tuple[int, int]:
        return self._imgsz

    def _class_threshold(self, cls: int) -> float:
        return self._class_conf.get(cls, self._conf)

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
        if (
            len(shape) == 4
            and isinstance(shape[2], int)
            and isinstance(shape[3], int)
            and shape[2] > 0
            and shape[3] > 0
        ):
            self._imgsz = (int(shape[2]), int(shape[3]))

        candidate_meta = meta_path if meta_path and meta_path.is_file() else path.with_suffix(".json")
        if candidate_meta.is_file():
            meta = json.loads(candidate_meta.read_text(encoding="utf-8"))
            if "imgsz" in meta:
                self._imgsz = _parse_imgsz(meta["imgsz"], self._imgsz)
            raw_names = meta.get("names") or {}
            if isinstance(raw_names, dict):
                self._names = {int(k): str(v) for k, v in raw_names.items()}
            elif isinstance(raw_names, list):
                self._names = {i: str(v) for i, v in enumerate(raw_names)}
            self._class_conf = _parse_class_conf(meta.get("class_conf"), self._names)
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
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        if raw.shape[1] <= 4:
            return []

        candidate_conf = min(self._conf, _LOW_CANDIDATE_CONF)
        if self._class_conf:
            candidate_conf = min(candidate_conf, min(self._class_conf.values()))

        scores = raw[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), cls_ids]
        keep = confs >= candidate_conf
        if not np.any(keep):
            return []

        boxes_xywh = raw[keep, :4]
        cls_ids = cls_ids[keep]
        confs = confs[keep]

        cx = (boxes_xywh[:, 0] - pad_w) / scale
        cy = (boxes_xywh[:, 1] - pad_h) / scale
        bw = boxes_xywh[:, 2] / scale
        bh = boxes_xywh[:, 3] / scale

        x1 = np.clip(cx - bw / 2.0, 0.0, float(orig_w))
        y1 = np.clip(cy - bh / 2.0, 0.0, float(orig_h))
        x2 = np.clip(cx + bw / 2.0, 0.0, float(orig_w))
        y2 = np.clip(cy + bh / 2.0, 0.0, float(orig_h))

        max_dim = float(max(orig_w, orig_h) + 1)
        offsets = cls_ids.astype(np.float32) * max_dim
        nms_boxes = np.stack([x1 + offsets, y1 + offsets, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            confs.astype(float).tolist(),
            candidate_conf,
            self._iou,
        )
        if len(indices) == 0:
            return []

        flat_idx = np.asarray(indices).reshape(-1)
        candidates: list[tuple[int, str, float, list[int]]] = []
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
            candidates.append((cls, label, float(confs[idx]), [ix1, iy1, ix2, iy2]))

        return self._postprocess(candidates, orig_w, orig_h)

    def _postprocess(
        self,
        candidates: list[tuple[int, str, float, list[int]]],
        orig_w: int,
        orig_h: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        primary: list[tuple[int, str, float, list[int]]] = []
        low: list[tuple[int, str, float, list[int]]] = []
        for item in candidates:
            cls, _label, conf, _box = item
            if conf >= self._class_threshold(cls):
                primary.append(item)
            elif conf >= _LOW_CANDIDATE_CONF:
                low.append(item)

        prefix_counts: dict[str, int] = {}
        for _cls, label, _conf, _box in primary:
            pref = _social_prefix(label)
            if pref is not None:
                prefix_counts[pref] = prefix_counts.get(pref, 0) + 1

        dominant_prefix: str | None = None
        if prefix_counts:
            ordered = sorted(prefix_counts.items(), key=lambda kv: -kv[1])
            top_pref, top_cnt = ordered[0]
            second_cnt = ordered[1][1] if len(ordered) > 1 else 0
            if top_cnt >= _APP_VOTE_MIN_DETECTIONS and (
                second_cnt == 0 or top_cnt >= _APP_VOTE_RATIO * second_cnt
            ):
                dominant_prefix = top_pref

        if dominant_prefix is not None:
            primary = [
                item
                for item in primary
                if (
                    _social_prefix(item[1]) is None
                    or _social_prefix(item[1]) == dominant_prefix
                    or item[2] >= _CROSS_APP_SUPPRESS_CONF
                )
            ]

        if orig_h > 0 and low:
            inv_h = 1.0 / float(orig_h)
            primary_boxes = [item[3] for item in primary]
            rescued: list[tuple[int, str, float, list[int]]] = []
            for prefix, slots in _TAB_BAR_SLOTS.items():
                if dominant_prefix is not None and prefix != dominant_prefix:
                    continue
                slot_set = set(slots)
                present_best: dict[str, tuple[float, float]] = {}
                for _cls, label, conf, box in primary:
                    if label not in slot_set:
                        continue
                    cy_norm = ((box[1] + box[3]) * 0.5) * inv_h
                    if cy_norm < _TAB_BAR_MIN_CY:
                        continue
                    prev = present_best.get(label)
                    if prev is None or conf > prev[0]:
                        present_best[label] = (conf, cy_norm)
                if len(present_best) < _TAB_BAR_MIN_SLOTS:
                    continue
                ys = [cy for _c, cy in present_best.values()]
                mean_y = float(np.mean(ys))
                if float(np.std(ys)) > _TAB_BAR_MAX_STD_CY:
                    continue
                missing = [s for s in slots if s not in present_best]
                for miss_label in missing:
                    best_item: tuple[int, str, float, list[int]] | None = None
                    for item in low:
                        if item[1] != miss_label:
                            continue
                        cy_norm = ((item[3][1] + item[3][3]) * 0.5) * inv_h
                        if abs(cy_norm - mean_y) > _TAB_BAR_RESCUE_DY:
                            continue
                        if any(_box_iou(item[3], pb) > _TAB_BAR_RESCUE_IOU for pb in primary_boxes):
                            continue
                        if best_item is None or item[2] > best_item[2]:
                            best_item = item
                    if best_item is not None:
                        rescued.append(best_item)
                        primary_boxes.append(best_item[3])
            primary.extend(rescued)

        detections: list[dict[str, Any]] = []
        for _cls, label, conf, box in primary:
            ix1, iy1, ix2, iy2 = box
            detections.append(
                {
                    "label": label,
                    "confidence": round(conf, 3),
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
