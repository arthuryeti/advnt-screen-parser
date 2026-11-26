from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .icon_detector import DEFAULT_HF_REPO, IconDetector
from .visualize import render_overlay


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="advnt-parse",
        description="Parse an iOS screenshot into structured UI icons, OCR text, and a UI element tree.",
    )
    ap.add_argument("image", help="Path to an iOS screenshot or photographed phone frame (JPEG/PNG).")
    ap.add_argument(
        "--icons-only",
        action="store_true",
        help="Run only the 47-class v3 YOLOv11s ONNX icon detector (~30ms CPU, skips RapidOCR).",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Optional local path to icons.onnx (defaults to Hugging Face Hub auto-download).",
    )
    ap.add_argument(
        "--repo-id",
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face model repo ID (default: {DEFAULT_HF_REPO}).",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Icon confidence threshold (default: 0.35).",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=2,
        help="ONNX Runtime intra-op CPU threads (default: 2).",
    )
    ap.add_argument(
        "--viz",
        default=None,
        help="Optional output path to save an annotated visualization PNG/JPG.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional path to write JSON output instead of stdout.",
    )
    args = ap.parse_args(argv)

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"Error: image not found: {img_path}", file=sys.stderr)
        return 1

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: failed to decode image: {img_path}", file=sys.stderr)
        return 1

    if args.icons_only:
        det = IconDetector(
            model_path=args.model,
            repo_id=args.repo_id,
            conf_threshold=args.conf,
            threads=args.threads,
        )
        icons = det.detect(image)
        payload = {
            "width": image.shape[1],
            "height": image.shape[0],
            "icons": icons,
        }
    else:
        from .parser import ScreenParser

        parser = ScreenParser(
            icon_model_path=args.model,
            repo_id=args.repo_id,
            conf_threshold=args.conf,
            threads=args.threads,
            enable_ocr=True,
        )
        payload = parser.parse_image(image)

    if args.viz:
        annotated = render_overlay(
            image,
            icons=payload.get("icons"),
            words=payload.get("words"),
            show_words=not args.icons_only,
        )
        out_viz = Path(args.viz)
        out_viz.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_viz), annotated)

    text = json.dumps(payload, indent=2)
    if args.out:
        out_json = Path(args.out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
