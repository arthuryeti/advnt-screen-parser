# Element-tree building adapted from ESP32 iPhone Agent by Henry Yuan (MIT):
# https://github.com/onehumanbeing/esp32-iphone-agent

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_MAX_W = 540
_OCR_PAD = 3
_K3 = np.ones((3, 3), np.uint8)
_K7 = np.ones((7, 7), np.uint8)


class _N:
    __slots__ = ("role", "label", "box", "children")

    def __init__(
        self,
        role: str,
        box: list[int],
        label: str | None = None,
        children: list[_N] | None = None,
    ) -> None:
        self.role = role
        self.label = label
        self.box = box
        self.children = [] if children is None else children


_PLATFORM_PREFIXES = ("ig_", "tt_", "x_", "yt_", "ios_")
_CANONICAL_OVERRIDES = {
    "x_media": "media_upload",
    "ios_app_tiktok": "app_result:tiktok",
    "ios_app_instagram": "app_result:instagram",
    "ios_app_x": "app_result:x",
    "ios_app_youtube": "app_result:youtube",
}
# Text-labeled share-sheet actions (like "Copy Link") are already anchored precisely by RapidOCR;
# circular share-sheet icon detections can collide with neighboring actions across platforms.
_IGNORED_ICON_CLASSES = {"ios_copy_link"}


def canonical_icon_label(raw: str) -> str:
    if raw in _CANONICAL_OVERRIDES:
        return _CANONICAL_OVERRIDES[raw]
    for prefix in _PLATFORM_PREFIXES:
        if raw.startswith(prefix):
            return raw[len(prefix) :]
    return raw


def _tag_post_tile(
    box: list[int], words: list[dict[str, Any]], text_boxes: list[list[int]]
) -> str:
    for word, tb in zip(words, text_boxes):
        cx = (tb[0] + tb[2]) / 2.0
        cy = (tb[1] + tb[3]) / 2.0
        if not (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]):
            continue
        txt = str(word.get("text") or "").strip().lower()
        if "draft" in txt:
            return "drafts_tile"
        if txt == "pinned" or "pinned" in txt:
            return "pinned_tile"
    return "post_tile"


def build(
    width: int,
    height: int,
    words: list[dict[str, Any]],
    image_bgr: np.ndarray,
    icons: list[dict[str, Any]] | None = None,
    *,
    max_elements: int = 140,
) -> list[dict[str, Any]]:
    """Build a reading-order UI element list from OCR words, learned icons, and CV geometry."""
    if width <= 0 or height <= 0 or image_bgr is None or image_bgr.size == 0:
        return []
    ww, wh, work = _working(image_bgr, width, height)
    text_boxes = [_to_px(word.get("boxPct"), ww, wh) for word in words]
    leaves = [
        _N("text", box, label=str(word.get("text") or ""))
        for word, box in zip(words, text_boxes)
    ]
    learned_icon_boxes: list[list[int]] = []
    icon_labels_seen: set[str] = set()
    if icons:
        sx = ww / float(width)
        sy = wh / float(height)
        for ic in icons:
            raw_box = ic.get("box")
            raw_label = str(ic.get("label") or "").strip()
            if (
                not raw_box
                or len(raw_box) < 4
                or not raw_label
                or raw_label in _IGNORED_ICON_CLASSES
            ):
                continue
            box = [
                max(0, min(ww - 1, int(round(raw_box[0] * sx)))),
                max(0, min(wh - 1, int(round(raw_box[1] * sy)))),
                max(1, min(ww, int(round(raw_box[2] * sx)))),
                max(1, min(wh, int(round(raw_box[3] * sy)))),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            label = canonical_icon_label(raw_label)
            if label == "post_tile":
                tile_label = _tag_post_tile(box, words, text_boxes)
                leaves.append(_N("image", box, label=tile_label))
            else:
                leaves.append(_N("icon", box, label=label))
                learned_icon_boxes.append(box)
                icon_labels_seen.add(label)

    has_feed_rail = bool({"like", "liked"} & icon_labels_seen) and {
        "comment",
        "share",
    }.issubset(icon_labels_seen)
    grid_tiles = _detect_grid_tiles(work, has_feed_rail=has_feed_rail)
    for gbox in grid_tiles:
        tile_label = _tag_post_tile(gbox, words, text_boxes)
        leaves.append(_N("image", gbox, label=tile_label))

    edges = _edge_map(work)
    for kind, box in _blobs(edges, text_boxes, ww, wh):
        if kind != "blob":
            continue
        if _overlaps_learned(box, learned_icon_boxes):
            continue
        if sum(_inter(box, t) for t in text_boxes) < 0.5 * _area(box):
            leaves.append(_N(_classify(work, box, ww), box))

    # Camera profile: photographed frames lose thin-stroke icons to Canny+morphology;
    # dark-glyph and saturated-fill passes recover them.
    visual = [leaf.box for leaf in leaves if leaf.role not in ("text",)]
    for box in _glyph_blobs(work, text_boxes, ww, wh):
        if _overlaps_learned(box, learned_icon_boxes):
            continue
        if not any(_iou(box, other) > 0.5 for other in visual):
            leaves.append(_N("icon", box))
            visual.append(box)
    for box in _color_buttons(work, ww, wh):
        if not any(_iou(box, other) > 0.5 for other in visual):
            leaves.append(_N("button", box))
            visual.append(box)

    _label_icons(leaves)
    rects = [_N("container", box) for box in _filled_rects(edges, ww, wh)]
    top = _prune(_nest(rects, leaves), ww, wh)
    top = _xy_cut(top, [0, 0, ww, wh])
    _finish(top, ww, wh)

    frame_area = ww * wh
    elems = [
        n
        for n in _walk(top)
        if _area(n.box) < 0.4 * frame_area
        and n.role not in ("row", "cell", "separator")
    ]
    if len(elems) > max_elements:
        # Keep labelled text/icons/tiles first so bottom controls are never truncated by unlabelled blobs.
        labelled = [n for n in elems if (n.label or "").strip()]
        unlabelled = [n for n in elems if not (n.label or "").strip()]
        keep_ids = {
            id(n)
            for n in (
                labelled[:max_elements]
                + unlabelled[: max(0, max_elements - len(labelled))]
            )
        }
        elems = [n for n in elems if id(n) in keep_ids][:max_elements]

    out: list[dict[str, Any]] = []
    for i, n in enumerate(elems, 1):
        x1, y1, x2, y2 = n.box
        out.append(
            {
                "id": i,
                "role": n.role,
                "label": n.label or "",
                "boxPct": [_pct(x1, ww), _pct(y1, wh), _pct(x2, ww), _pct(y2, wh)],
                "centerPct": [_pct((x1 + x2) / 2, ww), _pct((y1 + y2) / 2, wh)],
                "where": _where(n.box, ww, wh),
            }
        )
    return out


def _working(img: np.ndarray, w: int, h: int) -> tuple[int, int, np.ndarray]:
    if w <= _MAX_W:
        return w, h, img
    nw = _MAX_W
    nh = max(1, int(round(h * nw / w)))
    return nw, nh, cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _to_px(box_pct: Any, w: int, h: int) -> list[int]:
    l, t, r, b = [float(v) for v in (box_pct or [0, 0, 0, 0])[:4]]
    x1, y1 = int(l * w / 100.0), int(t * h / 100.0)
    x2, y2 = int(r * w / 100.0), int(b * h / 100.0)
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]


def _pct(value: float, dim: int) -> float:
    if dim <= 0:
        return 0.0
    return round(min(100.0, max(0.0, value / dim * 100)), 1)


def _area(b: list[int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a: list[int], b: list[int]) -> int:
    return _area(
        [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    )


def _inside(a: list[int], b: list[int], frac: float = 0.85) -> bool:
    return _area(a) > 0 and _inter(a, b) >= frac * _area(a)


def _walk(nodes: list[_N]) -> list[_N]:
    out: list[_N] = []
    for n in nodes:
        out.append(n)
        out.extend(_walk(n.children))
    return out


def _edge_map(bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (3, 3), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    return np.maximum.reduce([cv2.Canny(ch, 30, 90) for ch in cv2.split(lab)])


def _iou(a: list[int], b: list[int]) -> float:
    inter = _inter(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _glyph_blobs(
    bgr: np.ndarray, text_boxes: list[list[int]], w: int, h: int
) -> list[list[int]]:
    """Dark glyphs on a lighter field via adaptive local thresholding."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, -5
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K3)
    lo = max(8, w // 45)
    hi_w, hi_h = max(60, w // 6), max(60, h // 12)
    max_area = 0.02 * w * h
    out = []
    for s in cv2.connectedComponentsWithStats(mask, 8)[2][1:]:
        bw, bh = s[2], s[3]
        if not (lo <= bw <= hi_w and lo <= bh <= hi_h):
            continue
        if s[4] > max_area:
            continue
        out.append([s[0], s[1], s[0] + s[2], s[1] + s[3]])
    return out


def _color_buttons(bgr: np.ndarray, w: int, h: int) -> list[list[int]]:
    """Saturated filled rectangles (accent-color action buttons like Share / Follow)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 80, 80), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for s in stats[1:]:
        box = [s[0], s[1], s[0] + s[2], s[1] + s[3]]
        bw, bh = s[2], s[3]
        if bw < 0.12 * w or bh < 0.04 * h:
            continue
        if bw * bh > 0.3 * w * h:
            continue
        if s[4] < 0.55 * bw * bh:
            continue
        out.append(box)
    return out


def _blobs(
    edges: np.ndarray, text_boxes: list[list[int]], w: int, h: int
) -> list[tuple[str, list[int]]]:
    erased = edges.copy()
    for x1, y1, x2, y2 in text_boxes:
        erased[
            max(0, y1 - _OCR_PAD) : min(h, y2 + _OCR_PAD),
            max(0, x1 - _OCR_PAD) : min(w, x2 + _OCR_PAD),
        ] = 0
    closed = cv2.morphologyEx(cv2.dilate(erased, _K3), cv2.MORPH_CLOSE, _K7)
    out: list[tuple[str, list[int]]] = []
    for b in _components(closed):
        bw, bh = b[2] - b[0], b[3] - b[1]
        if (bw >= 0.9 * w and bh <= 6) or (bh >= 0.5 * h and bw <= 6):
            out.append(("separator", b))
        elif min(bw, bh) >= 12 and bw <= 0.6 * w and bh <= 0.35 * h:
            out.append(("blob", b))
    return out


def _filled_rects(edges: np.ndarray, w: int, h: int) -> list[list[int]]:
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _K3)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found: list[list[int]] = []
    frame = w * h
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 40 or bh < 24 or bw * bh > 0.6 * frame:
            continue
        box = [int(x), int(y), int(x + bw), int(y + bh)]
        aa = float(cv2.contourArea(cnt))
        if aa / _area(box) >= 0.8:
            found.append(box)
    found.sort(key=lambda b: -_area(b))
    keep: list[list[int]] = []
    for b in found:
        if any(
            _inter(b, k) > 0.9 * _area(b) and _area(k) < 1.2 * _area(b) for k in keep
        ):
            continue
        keep.append(b)
    return keep


def _components(mask: np.ndarray) -> list[list[int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[list[int]] = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append([x, y, x + bw, y + bh])
    return boxes


def _classify(img: np.ndarray, box: list[int], w: int) -> str:
    bw = float(box[2] - box[0])
    bh = float(box[3] - box[1])
    if bh > 0 and 1.6 <= bw / bh <= 2.4 and bh <= 50:
        return "control"
    if max(bw, bh) >= 0.22 * w and _color_spread(img, box) > 25:
        return "image"
    return "icon"


def _color_spread(img: np.ndarray, box: list[int]) -> float:
    roi = img[box[1] : box[3], box[0] : box[2]]
    if roi.size == 0:
        return 0.0
    return float(roi.reshape(-1, 3).std(axis=0).mean())


def _detect_grid_tiles(
    bgr: np.ndarray, has_feed_rail: bool = False
) -> list[list[int]]:
    """Detect 3-column iOS profile photo/video grids via paired 1/3 & 2/3 vertical gutter seams."""
    if has_feed_rail:
        return []
    h, w = bgr.shape[:2]
    if w < 120 or h < 200:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    band = max(4, int(round(0.03 * w)))
    x1_lo, x1_hi = max(5, w // 3 - band), min(w - 6, w // 3 + band)
    x2_lo, x2_hi = max(5, (2 * w) // 3 - band), min(w - 6, (2 * w) // 3 + band)
    d = max(4, int(round(0.012 * w)))
    y_min = int(round(0.18 * h))
    y_max = int(round(0.99 * h))
    if y_max - y_min < 40 or x1_lo - d < 0 or x2_hi + d >= w:
        return []

    def best_seam(
        x_lo: int, x_hi: int
    ) -> tuple[int, np.ndarray | None, np.ndarray | None]:
        best_x, best_score, best_mask, best_col = -1, -1.0, None, None
        for x in range(x_lo, x_hi + 1):
            col = gray[y_min:y_max, x]
            left = gray[y_min:y_max, x - d]
            right = gray[y_min:y_max, x + d]
            contrast = np.maximum(
                np.minimum(col - left, col - right),
                np.minimum(left - col, right - col),
            )
            vdiff = np.abs(np.diff(col, prepend=col[0]))
            mask = (contrast > 12.0) & (vdiff < 10.0)
            score = float(mask.sum())
            if score > best_score:
                best_score, best_x, best_mask, best_col = score, x, mask, col
        return best_x, best_mask, best_col

    x1, m1, c1 = best_seam(x1_lo, x1_hi)
    x2, m2, c2 = best_seam(x2_lo, x2_hi)
    if x1 < 0 or x2 < 0 or m1 is None or m2 is None or c1 is None or c2 is None:
        return []

    both = (m1 & m2 & (np.abs(c1 - c2) < 28.0)).astype(np.uint8)
    gap = max(3, int(round(0.015 * h)))
    closed = cv2.morphologyEx(
        both.reshape(-1, 1), cv2.MORPH_CLOSE, np.ones((gap * 2 + 1, 1), np.uint8)
    ).reshape(-1)

    best_run = (0, 0)
    cur_start: int | None = None
    for idx, val in enumerate(closed):
        if val and cur_start is None:
            cur_start = idx
        elif not val and cur_start is not None:
            if idx - cur_start > best_run[1] - best_run[0]:
                best_run = (cur_start, idx)
            cur_start = None
    if cur_start is not None and len(closed) - cur_start > best_run[1] - best_run[0]:
        best_run = (cur_start, len(closed))

    s0, s1 = best_run
    if s1 - s0 < max(int(round(0.08 * h)), int(0.9 * (x2 - x1))):
        return []
    if float(both[s0:s1].mean()) < 0.65:
        return []
    if float(c1[s0:s1].std()) > 22.0 or float(c2[s0:s1].std()) > 22.0:
        return []

    gy1, gy2 = y_min + s0, y_min + s1
    col_w = x2 - x1
    h_seams: list[int] = []
    hd = max(4, int(round(0.012 * h)))
    for y in range(max(gy1 + 10, hd), min(gy2 - 10, h - hd)):
        row = gray[y, 10 : w - 10]
        up = gray[y - hd, 10 : w - 10]
        dn = gray[y + hd, 10 : w - 10]
        contrast = np.maximum(
            np.minimum(row - up, row - dn), np.minimum(up - row, dn - row)
        )
        hdiff = np.abs(np.diff(row, prepend=row[0]))
        if float(((contrast > 12.0) & (hdiff < 10.0)).mean()) > 0.45:
            h_seams.append(y)

    dividers = [gy1]
    if h_seams:
        clusters: list[list[int]] = [[h_seams[0]]]
        for y in h_seams[1:]:
            if y - clusters[-1][-1] <= 8:
                clusters[-1].append(y)
            else:
                clusters.append([y])
        for cl in clusters:
            ym = int(round(sum(cl) / len(cl)))
            if ym - dividers[-1] >= int(0.65 * col_w):
                dividers.append(ym)
    if gy2 - dividers[-1] >= int(0.35 * col_w):
        dividers.append(gy2)

    final_rows: list[tuple[int, int]] = []
    for a, b in zip(dividers[:-1], dividers[1:]):
        span = b - a
        if span > int(2.1 * col_w):
            step = max(col_w, int(round(1.33 * col_w)))
            y = a
            while y + int(0.45 * col_w) <= b:
                final_rows.append((y, min(b, y + step)))
                y += step
        elif span >= int(0.35 * col_w):
            final_rows.append((a, b))

    cols = [(0, x1), (x1, x2), (x2, w)]
    out: list[list[int]] = []
    for ry1, ry2 in final_rows:
        for cx1, cx2 in cols:
            roi = bgr[ry1:ry2, cx1:cx2]
            if roi.size == 0 or float(roi.reshape(-1, 3).std(axis=0).mean()) < 14.0:
                continue
            out.append([cx1, ry1, cx2, ry2])
    return out


def _overlaps_learned(box: list[int], learned: list[list[int]]) -> bool:
    if not learned:
        return False
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    for ib in learned:
        if _iou(box, ib) > 0.25:
            return True
        if ib[0] <= cx <= ib[2] and ib[1] <= cy <= ib[3]:
            return True
    return False


def _label_icons(leaves: list[_N]) -> None:
    texts = [n for n in leaves if n.role == "text" and (n.label or "").strip()]
    for n in leaves:
        if n.role not in ("icon", "image", "control") or n.label:
            continue
        b = n.box
        cx = (b[0] + b[2]) / 2
        bh = float(b[3] - b[1])
        below = next(
            (
                t
                for t in texts
                if 0 <= t.box[1] - b[3] <= 0.6 * bh
                and t.box[0] <= cx <= t.box[2]
            ),
            None,
        )
        if below is not None:
            text = (below.label or "").strip()
            if text:
                n.label = f'near "{text}"'
            continue
        mid = (b[1] + b[3]) / 2
        slop = max(bh, 20) / 2 + 6
        side = [
            t
            for t in texts
            if abs((t.box[1] + t.box[3]) / 2 - mid) < slop
        ]
        if not side:
            continue
        nearest = min(
            side,
            key=lambda t: min(abs(t.box[0] - b[2]), abs(b[0] - t.box[2])),
        )
        text = (nearest.label or "").strip()
        if text:
            n.label = f'near "{text}"'


def _nest(parents: list[_N], leaves: list[_N]) -> list[_N]:
    items = sorted(parents + leaves, key=lambda n: -_area(n.box))
    top: list[_N] = []
    for i, n in enumerate(items):
        holder = None
        holder_area = 0
        for p in items[:i]:
            if p.role == "container" and p is not n and _inside(n.box, p.box):
                a = _area(p.box)
                if holder is None or a < holder_area:
                    holder = p
                    holder_area = a
        if holder is not None:
            holder.children.append(n)
        else:
            top.append(n)
    return top


def _prune(nodes: list[_N], w: int, h: int) -> list[_N]:
    out: list[_N] = []
    frame = w * h
    for n in nodes:
        if n.role != "container":
            out.append(n)
            continue
        n.children = _prune(n.children, w, h)
        if not n.children:
            if _area(n.box) < 0.02 * frame:
                n.role = "button"
                out.append(n)
            continue
        if len(n.children) == 1 and _area(n.children[0].box) > 0.6 * _area(n.box):
            out.append(n.children[0])
            continue
        n.children = _xy_cut(n.children, n.box)
        out.append(n)
    return out


def _xy_cut(
    nodes: list[_N], box: list[int], depth: int = 0, horizontal: bool = True
) -> list[_N]:
    if len(nodes) <= 1 or depth > 6:
        return nodes
    axes = [(1, 3), (0, 2)] if horizontal else [(0, 2), (1, 3)]
    for lo, hi in axes:
        spans = sorted((n.box[lo], n.box[hi]) for n in nodes)
        cuts: list[float] = []
        end = spans[0][1]
        for s, e in spans[1:]:
            if s > end + 4:
                cuts.append((end + s) / 2)
            end = max(end, e)
        if not cuts:
            continue
        groups: list[list[_N]] = [[] for _ in range(len(cuts) + 1)]
        for n in nodes:
            c = (n.box[lo] + n.box[hi]) / 2
            groups[sum(1 for cut in cuts if c > cut)].append(n)
        role = "row" if lo == 1 else "cell"
        out: list[_N] = []
        for g in groups:
            if len(g) == 1:
                out.append(g[0])
                continue
            if not g:
                continue
            gb = [
                min(n.box[0] for n in g),
                min(n.box[1] for n in g),
                max(n.box[2] for n in g),
                max(n.box[3] for n in g),
            ]
            ca, cb = (0, 2) if lo == 1 else (1, 3)
            gb[ca] = box[ca]
            gb[cb] = box[cb]
            out.append(_N(role, gb, children=_xy_cut(g, gb, depth + 1, lo != 1)))
        if lo == 1 and depth == 0:
            out.extend(_blanks(spans, box))
        return out
    return nodes


def _blanks(
    spans: list[tuple[int, int]], box: list[int], min_gap: float = 0.08
) -> list[_N]:
    height = float(box[3] - box[1])
    out: list[_N] = []
    end = float(box[1])
    for s, e in list(spans) + [(box[3], box[3])]:
        if s - end >= min_gap * height:
            out.append(
                _N("blank", [box[0], int(end), box[2], s], label="empty area")
            )
        end = max(end, float(e))
    return out


def _finish(nodes: list[_N], w: int, h: int) -> None:
    nodes.sort(key=lambda n: round(n.box[1] / 12) * 100_000 + n.box[0])
    for n in nodes:
        _finish(n.children, w, h)
        if n.label is None:
            words = [
                c.label
                for c in _walk(n.children)
                if c.role == "text" and c.label
            ]
            n.label = " ".join(words)[:80]


def _where(b: list[int], w: int, h: int) -> str:
    cx = (b[0] + b[2]) / 2 / w
    cy = (b[1] + b[3]) / 2 / h
    v = (
        "top"
        if cy < 0.15
        else "bottom"
        if cy > 0.85
        else "upper"
        if cy < 0.45
        else "lower"
        if cy > 0.6
        else "middle"
    )
    hz = "left" if cx < 0.33 else "right" if cx > 0.67 else "center"
    return f"{v} {hz}"
