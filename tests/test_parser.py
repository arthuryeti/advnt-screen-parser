from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from advnt_screen_parser import IconDetector, build_ui_tree, render_overlay

W, H = 1080, 1920


def _blank() -> np.ndarray:
    return np.full((H, W, 3), 255, dtype=np.uint8)


def _pct(x1: int, y1: int, x2: int, y2: int) -> list[float]:
    return [
        round(x1 / W * 100, 2),
        round(y1 / H * 100, 2),
        round(x2 / W * 100, 2),
        round(y2 / H * 100, 2),
    ]


def _word(text: str, box: list[int]) -> dict:
    return {"text": text, "confidence": 0.99, "boxPct": _pct(*box)}


def test_button_with_label_and_where() -> None:
    img = _blank()
    cv2.rectangle(img, (190, 1750), (890, 1806), (40, 40, 40), -1)
    words = [_word("Post", (40, 80, 200, 130))]
    elements = build_ui_tree(W, H, words, img)
    buttons = [e for e in elements if e["role"] == "button"]
    assert buttons, elements
    assert buttons[0]["where"] == "bottom center"
    texts = [e for e in elements if e["role"] == "text" and e["label"] == "Post"]
    assert texts and texts[0]["where"].startswith("top")


def test_icon_labeled_from_text_below() -> None:
    img = _blank()
    cv2.rectangle(img, (100, 200), (180, 280), (0, 0, 0), -1)
    words = [_word("Home", (90, 300, 200, 330))]
    elements = build_ui_tree(W, H, words, img)
    icons = [e for e in elements if e["role"] == "icon" and e["label"] == 'near "Home"']
    assert icons, elements
    raw = [e for e in elements if e["role"] == "icon" and e["label"] == "Home"]
    assert not raw, elements


def test_ids_sequential_reading_order() -> None:
    img = _blank()
    words = [
        _word("B", (400, 400, 500, 440)),
        _word("A", (40, 40, 140, 80)),
        _word("C", (40, 800, 140, 840)),
    ]
    elements = build_ui_tree(W, H, words, img)
    texts = [e for e in elements if e["role"] == "text"]
    assert [e["id"] for e in elements] == list(range(1, len(elements) + 1))
    assert [e["label"] for e in texts] == ["A", "B", "C"]


def test_excludes_layout_roles_and_caps() -> None:
    w, h = 540, 4000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (1, 10), (3, 2500), (0, 0, 0), -1)
    words = [
        {
            "text": f"w{i}",
            "confidence": 1.0,
            "boxPct": [
                round(40 / w * 100, 2),
                round((10 + i * 12) / h * 100, 2),
                round(80 / w * 100, 2),
                round((14 + i * 12) / h * 100, 2),
            ],
        }
        for i in range(200)
    ]
    elements = build_ui_tree(w, h, words, img)
    roles = {e["role"] for e in elements}
    assert not roles & {"row", "cell", "separator"}
    assert len(elements) == 140


def test_determinism_and_speed() -> None:
    img = _blank()
    cv2.rectangle(img, (100, 200), (180, 280), (0, 0, 0), -1)
    cv2.rectangle(img, (190, 1750), (890, 1806), (40, 40, 40), -1)
    words = [_word("Home", (90, 300, 200, 330)), _word("Post", (40, 80, 200, 130))]
    a = build_ui_tree(W, H, words, img)
    b = build_ui_tree(W, H, words, img.copy())
    assert a == b
    started = time.perf_counter()
    build_ui_tree(W, H, words, img)
    assert (time.perf_counter() - started) * 1000 < 100


def test_learned_icons_keep_semantic_label_and_dedupe_blobs() -> None:
    img = _blank()
    cv2.rectangle(img, (900, 1200), (980, 1280), (0, 0, 0), -1)
    words = [_word("9,951", (890, 1295, 990, 1325))]
    icons = [{"label": "ig_share", "confidence": 0.92, "box": [895, 1195, 985, 1285]}]
    elements = build_ui_tree(W, H, words, img, icons=icons)
    share = [e for e in elements if e["role"] == "icon" and e["label"] == "share"]
    assert len(share) == 1
    assert not [e for e in elements if e["role"] == "icon" and e["label"] == "9,951"]


def test_geometric_3col_grid_slicer() -> None:
    img = _blank()
    rng = np.random.default_rng(7)
    g1, g2 = W // 3, (2 * W) // 3
    for cx1, cx2 in [(0, g1 - 2), (g1 + 2, g2 - 2), (g2 + 2, W)]:
        img[1300:1860, cx1:cx2] = rng.integers(
            10, 140, size=(560, cx2 - cx1, 3), dtype=np.uint8
        )
    words = [_word("Drafts", (80, 1780, 260, 1820))]
    elements = build_ui_tree(W, H, words, img)
    assert len([e for e in elements if e["label"] == "drafts_tile"]) == 1
    assert len([e for e in elements if e["label"] == "post_tile"]) == 2


def test_render_overlay_preserves_shape() -> None:
    img = np.zeros((400, 200, 3), dtype=np.uint8)
    out = render_overlay(
        img,
        icons=[{"label": "ig_like", "confidence": 0.95, "box": [150, 200, 190, 240]}],
        words=[{"text": "9,951", "boxPct": [75.0, 62.0, 95.0, 67.0]}],
    )
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_icon_detector_local_weights() -> None:
    model_path = os.environ.get("ADVNT_ICON_MODEL_PATH")
    if not model_path or not Path(model_path).is_file():
        pytest.skip("Set ADVNT_ICON_MODEL_PATH to run ONNX integration test")
    det = IconDetector(model_path=model_path, auto_download=False)
    assert det.available
    assert len(det.names) == 99
    assert det.imgsz == (800, 416)
    example = Path(__file__).resolve().parents[1] / "assets" / "examples" / "instagram_feed.jpg"
    if example.is_file():
        res = det.detect_file(example)
        labels = {d["label"] for d in res}
        assert {"ig_comment", "ig_share", "ig_home_tab", "ig_profile_tab"}.issubset(labels)
        assert labels & {"ig_like", "ig_liked"}
