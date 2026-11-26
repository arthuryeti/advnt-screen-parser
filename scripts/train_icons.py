#!/usr/bin/env python3
"""Train a YOLOv11s iOS UI icon detector and export a CPU-fast ONNX model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune YOLOv11s on YOLO-format iOS UI icon screenshots and export ONNX."
    )
    ap.add_argument("--data", required=True, help="Path to YOLO dataset data.yaml")
    ap.add_argument("--out", default="./runs", help="Ultralytics training output dir")
    ap.add_argument("--export-to", default="./models/icons.onnx", help="Destination ONNX file")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--name", default="icons-v3")
    ap.add_argument("--model", default="yolo11s.pt")
    args = ap.parse_args()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(
        f"Training {args.model} on device={device}, imgsz={args.imgsz}, "
        f"epochs={args.epochs}, cache={args.cache}"
    )

    model = YOLO(args.model)
    # UI icons are orientation- and layout-sensitive: disable horizontal/vertical flips and
    # heavy rotation while keeping mild scale/translation/color jitter for camera robustness.
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        cache=args.cache if args.cache != "false" else False,
        device=device,
        project=args.out,
        name=args.name,
        exist_ok=True,
        patience=10,
        fliplr=0.0,
        flipud=0.0,
        degrees=3.0,
        translate=0.08,
        scale=0.18,
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.35,
        mosaic=0.3,
        close_mosaic=8,
        workers=4,
        verbose=True,
    )

    best_pt = Path(args.out) / args.name / "weights" / "best.pt"
    best = YOLO(str(best_pt))
    val_metrics = best.val(
        data=args.data,
        imgsz=args.imgsz,
        device=device,
        project=args.out,
        name=f"{args.name}-val",
        exist_ok=True,
    )
    print(
        f"VAL mAP50={val_metrics.box.map50:.4f} mAP50-95={val_metrics.box.map:.4f} "
        f"P={val_metrics.box.mp:.4f} R={val_metrics.box.mr:.4f}"
    )

    onnx_path = Path(
        best.export(
            format="onnx",
            imgsz=args.imgsz,
            simplify=True,
            opset=17,
            dynamic=False,
        )
    )
    export_dest = Path(args.export_to)
    export_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, export_dest)

    meta = {
        "imgsz": args.imgsz,
        "names": best.names,
        "map50": round(float(val_metrics.box.map50), 4),
        "map50_95": round(float(val_metrics.box.map), 4),
        "precision": round(float(val_metrics.box.mp), 4),
        "recall": round(float(val_metrics.box.mr), 4),
    }
    meta_dest = export_dest.with_suffix(".json")
    meta_dest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Exported ONNX ({export_dest.stat().st_size / 1e6:.2f} MB) -> {export_dest}")


if __name__ == "__main__":
    main()
