#!/usr/bin/env python3
"""Train a YOLOv11 iOS UI icon detector and export a CPU-fast portrait ONNX model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/icon-yolo-v4/dataset/data.yaml")
    ap.add_argument("--out", default="/tmp/icon-yolo-v4/runs")
    ap.add_argument("--export-to", default="services/ocr/models/icons.onnx")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument(
        "--export-imgsz",
        type=int,
        nargs="+",
        default=[800, 416],
        help="Export ONNX input size: single int (square) or [H, W] portrait (e.g. 800 416).",
    )
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--name", default="icons-v4")
    ap.add_argument("--model", default="/tmp/icon-yolo-v2/yolo11s.pt")
    args = ap.parse_args()

    export_imgsz: int | list[int] = (
        args.export_imgsz[0]
        if len(args.export_imgsz) == 1
        else [int(args.export_imgsz[0]), int(args.export_imgsz[1])]
    )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(
        f"Training {args.model} on device={device}, imgsz={args.imgsz}, "
        f"export_imgsz={export_imgsz}, epochs={args.epochs}, batch={args.batch}, cache={args.cache}"
    )

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    # UI icons are orientation- and layout-sensitive: disable horizontal/vertical flip and heavy rotation,
    # keep mild scale/translation/color jitter to generalize across phone camera crops & lighting.
    # Keep hsv_h tight (0.008) so red/yellow/blue active states (*_liked, *_saved, tt_follow) preserve hue.
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
        patience=12,
        fliplr=0.0,
        flipud=0.0,
        degrees=2.5,
        translate=0.08,
        scale=0.18,
        hsv_h=0.008,
        hsv_s=0.25,
        hsv_v=0.35,
        mosaic=0.3,
        close_mosaic=10,
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
        f"VAL ({args.imgsz}) mAP50={val_metrics.box.map50:.4f} "
        f"mAP50-95={val_metrics.box.map:.4f} "
        f"P={val_metrics.box.mp:.4f} R={val_metrics.box.mr:.4f}"
    )

    onnx_path = Path(
        best.export(
            format="onnx",
            imgsz=export_imgsz,
            simplify=True,
            opset=17,
            dynamic=False,
        )
    )
    export_dest = Path(args.export_to)
    export_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, export_dest)

    # Validate the exported ONNX model directly at its actual deployment resolution!
    onnx_model = YOLO(str(export_dest), task="detect")
    onnx_val = onnx_model.val(
        data=args.data,
        imgsz=export_imgsz,
        device="cpu",
        project=args.out,
        name=f"{args.name}-onnx-val",
        exist_ok=True,
    )
    print(
        f"ONNX VAL ({export_imgsz}) mAP50={onnx_val.box.map50:.4f} "
        f"mAP50-95={onnx_val.box.map:.4f} "
        f"P={onnx_val.box.mp:.4f} R={onnx_val.box.mr:.4f}"
    )

    per_class_map50: dict[str, float] = {}
    if hasattr(onnx_val.box, "ap_class_index") and hasattr(onnx_val.box, "ap50"):
        for idx, ap in zip(onnx_val.box.ap_class_index, onnx_val.box.ap50):
            name = best.names.get(int(idx), str(idx))
            per_class_map50[name] = round(float(ap), 4)

    meta = {
        "version": args.name,
        "imgsz": export_imgsz,
        "names": best.names,
        "map50": round(float(onnx_val.box.map50), 4),
        "map50_95": round(float(onnx_val.box.map), 4),
        "precision": round(float(onnx_val.box.mp), 4),
        "recall": round(float(onnx_val.box.mr), 4),
        "pt_map50": round(float(val_metrics.box.map50), 4),
        "pt_map50_95": round(float(val_metrics.box.map), 4),
        "per_class_map50": per_class_map50,
    }
    meta_dest = export_dest.with_suffix(".json")
    meta_dest.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Exported ONNX ({export_dest.stat().st_size / 1e6:.2f} MB) -> {export_dest}")


if __name__ == "__main__":
    main()
