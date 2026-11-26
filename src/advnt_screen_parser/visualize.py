from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# BGR colors by platform prefix
_PLATFORM_COLORS: dict[str, tuple[int, int, int]] = {
    "ig_": (180, 90, 255),   # warm magenta-coral
    "tt_": (220, 235, 35),   # electric cyan
    "x_": (255, 175, 45),    # crisp sky blue
    "yt_": (75, 85, 255),    # vibrant coral red
    "ios_": (95, 230, 140),  # mint green
}
_DEFAULT_ICON_COLOR = (80, 225, 255)
_TEXT_BOX_COLOR = (210, 210, 210)


def _color_for_label(label: str) -> tuple[int, int, int]:
    for prefix, color in _PLATFORM_COLORS.items():
        if label.startswith(prefix):
            return color
    return _DEFAULT_ICON_COLOR


def render_overlay(
    image_bgr: np.ndarray,
    *,
    icons: list[dict[str, Any]] | None = None,
    words: list[dict[str, Any]] | None = None,
    show_words: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """Render clean bounding boxes and pill tags for detected icons and OCR words."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    if show_words and words:
        overlay = canvas.copy()
        for word in words:
            bp = word.get("boxPct")
            if not bp or len(bp) < 4:
                continue
            x1 = int(round(bp[0] * w / 100.0))
            y1 = int(round(bp[1] * h / 100.0))
            x2 = int(round(bp[2] * w / 100.0))
            y2 = int(round(bp[3] * h / 100.0))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), _TEXT_BOX_COLOR, 1)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    placed_pills: list[tuple[int, int, int, int]] = []
    for det in icons or []:
        box = det.get("box")
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        label = str(det.get("label") or "icon")
        conf = det.get("confidence")
        color = _color_for_label(label)

        # Subtle translucent box fill + crisp outline
        sub = canvas[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if sub.size > 0:
            tint = np.full_like(sub, color)
            cv2.addWeighted(tint, 0.18, sub, 0.82, 0, sub)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

        tag = f"{label} {conf:.2f}" if (show_confidence and isinstance(conf, (int, float))) else label
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.36, min(0.48, w / 1100.0))
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(tag, font, scale, thickness)
        pad_x, pad_y = 4, 3
        pw, ph = tw + pad_x * 2, th + pad_y * 2

        # Try above box, then below, then left/right so pills never clip off-screen or stack.
        candidates = [
            (min(max(2, x1), w - pw - 2), max(2, y1 - ph - 2)),
            (min(max(2, x1), w - pw - 2), min(h - ph - 2, y2 + 2)),
            (max(2, x1 - pw - 4), min(max(2, y1), h - ph - 2)),
            (min(w - pw - 2, x2 + 4), min(max(2, y1), h - ph - 2)),
        ]
        px, py = candidates[0]
        for cx, cy in candidates:
            rect = (cx, cy, cx + pw, cy + ph)
            if not any(
                not (rect[2] <= o[0] or rect[0] >= o[2] or rect[3] <= o[1] or rect[1] >= o[3])
                for o in placed_pills
            ):
                px, py = cx, cy
                break
        placed_pills.append((px, py, px + pw, py + ph))

        # Dark pill background with colored left accent bar
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 20, 26), -1)
        cv2.rectangle(canvas, (px, py), (px + 2, py + ph), color, -1)
        cv2.putText(
            canvas,
            tag,
            (px + pad_x + 1, py + ph - pad_y - baseline // 2),
            font,
            scale,
            (245, 247, 250),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return canvas
