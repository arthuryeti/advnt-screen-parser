#!/usr/bin/env python3
"""Two-stage v5 portrait YOLOv11 training, ONNX export, and per-class threshold calibration."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def calibrate_per_class_thresholds(
    onnx_path: Path,
    meta_path: Path,
    val_img_dir: Path,
    val_lbl_dir: Path,
    names: dict[int, str],
) -> dict[str, float]:
    """Sweep confidence thresholds per class on the val set to maximize F1."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "ocr"))
    from icon_detector import IconDetector

    # Run with low floor so we capture all candidate confidences
    det = IconDetector(model_path=onnx_path, conf_threshold=0.12)
    det._class_conf = {}  # disable any prior calibration during sweep
    det.warm_up()

    # Gather per-class (conf, is_tp) + total_gt
    gt_counts = collections.Counter()
    records: dict[str, list[tuple[float, bool]]] = collections.defaultdict(list)

    for img_p in sorted(val_img_dir.glob("*.jpg")):
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]
        lbl_p = val_lbl_dir / (img_p.stem + ".txt")
        gts_by_cls: dict[str, list[list[float]]] = collections.defaultdict(list)
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                name = names[cls]
                gts_by_cls[name].append(
                    [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                )
                gt_counts[name] += 1

        # Raw NMS detections down to 0.12 (disable app-voting/slot-recovery during pure calibration)
        preds = det.detect(im)
        preds_by_cls: dict[str, list[dict]] = collections.defaultdict(list)
        for pr in preds:
            preds_by_cls[pr["label"]].append(pr)

        for cls_name, plist in preds_by_cls.items():
            plist.sort(key=lambda x: -x["confidence"])
            gts = gts_by_cls.get(cls_name, [])
            matched: set[int] = set()
            for pr in plist:
                best_i, best_j = 0.0, -1
                for j, gb in enumerate(gts):
                    if j in matched:
                        continue
                    v = box_iou(pr["box"], gb)
                    if v > best_i:
                        best_i, best_j = v, j
                if best_i >= 0.5:
                    matched.add(best_j)
                    records[cls_name].append((pr["confidence"], True))
                else:
                    records[cls_name].append((pr["confidence"], False))

    class_conf: dict[str, float] = {}
    for cls_id in sorted(names):
        name = names[cls_id]
        n_gt = gt_counts[name]
        if n_gt < 5:
            # Keep safe default for tiny-val classes
            continue
        recs = records.get(name, [])
        if not recs:
            continue
        best_f1 = -1.0
        best_t = 0.35
        for t in np.linspace(0.18, 0.55, 38):
            tp = sum(1 for c, ok in recs if c >= t and ok)
            fp = sum(1 for c, ok in recs if c >= t and not ok)
            fn = n_gt - tp
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            # Slight preference for staying close to 0.35 on ties
            if f1 > best_f1 + 1e-4 or (abs(f1 - best_f1) <= 1e-4 and abs(t - 0.35) < abs(best_t - 0.35)):
                best_f1 = f1
                best_t = float(round(t, 2))
        if abs(best_t - 0.35) >= 0.04:
            class_conf[name] = best_t

    return class_conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/icon-yolo-v5/dataset/data.yaml")
    ap.add_argument("--out", default="/tmp/icon-yolo-v5/runs")
    ap.add_argument("--export-to", default="services/ocr/models/icons.onnx")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--stage2-imgsz", type=int, default=960)
    ap.add_argument("--export-imgsz", type=int, nargs="+", default=[960, 480])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--name", default="icons-v5")
    ap.add_argument("--model", default="/tmp/icon-yolo-v5/yolo11m.pt")
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
        f"Stage 1 training {args.model} on device={device}, imgsz={args.imgsz}, "
        f"export_imgsz={export_imgsz}, epochs={args.epochs}+{args.finetune_epochs}, batch={args.batch}"
    )

    model = YOLO(args.model)
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
        patience=15,
        cos_lr=True,
        fliplr=0.0,
        flipud=0.0,
        degrees=2.5,
        translate=0.08,
        scale=0.22,
        hsv_h=0.008,
        hsv_s=0.25,
        hsv_v=0.35,
        mosaic=0.35,
        close_mosaic=min(10, max(3, args.epochs // 3)),
        workers=4,
        verbose=True,
    )

    stage1_pt = Path(args.out) / args.name / "weights" / "best.pt"
    best_pt = stage1_pt

    # Stage 2: portrait-aspect native fine-tune (`rect=True`, mosaic=0.0, low LR) at 960px
    # to eliminate train-square vs export-portrait aspect-ratio mismatch and sharpen tiny icons!
    if args.finetune_epochs > 0:
        print(
            f"Stage 2 portrait fine-tune ({args.finetune_epochs} epochs, rect=True, "
            f"imgsz={args.stage2_imgsz}) from {stage1_pt}"
        )
        ft_model = YOLO(str(stage1_pt))
        ft_model.train(
            data=args.data,
            epochs=args.finetune_epochs,
            imgsz=args.stage2_imgsz,
            rect=True,
            batch=args.batch,
            cache=args.cache if args.cache != "false" else False,
            device=device,
            project=args.out,
            name=f"{args.name}-portrait",
            exist_ok=True,
            patience=6,
            cos_lr=True,
            lr0=0.0005,
            lrf=0.05,
            warmup_epochs=1.0,
            fliplr=0.0,
            flipud=0.0,
            degrees=1.5,
            translate=0.05,
            scale=0.12,
            hsv_h=0.006,
            hsv_s=0.2,
            hsv_v=0.25,
            mosaic=0.0,
            workers=4,
            verbose=True,
        )
        stage2_pt = Path(args.out) / f"{args.name}-portrait" / "weights" / "best.pt"
        # Compare both checkpoints at target portrait resolution
        v1 = YOLO(str(stage1_pt)).val(data=args.data, imgsz=args.stage2_imgsz, rect=True, device=device, project=args.out, name=f"{args.name}-cmp1", exist_ok=True)
        v2 = YOLO(str(stage2_pt)).val(data=args.data, imgsz=args.stage2_imgsz, rect=True, device=device, project=args.out, name=f"{args.name}-cmp2", exist_ok=True)
        print(f"Stage 1 portrait val: mAP50={v1.box.map50:.4f} mAP50-95={v1.box.map:.4f}")
        print(f"Stage 2 portrait val: mAP50={v2.box.map50:.4f} mAP50-95={v2.box.map:.4f}")
        if (v2.box.map50 * 0.5 + v2.box.map * 0.5) >= (v1.box.map50 * 0.5 + v1.box.map * 0.5):
            best_pt = stage2_pt
            val_metrics = v2
            print("Selected Stage 2 portrait checkpoint!")
        else:
            best_pt = stage1_pt
            val_metrics = v1
            print("Kept Stage 1 checkpoint!")
    else:
        val_metrics = YOLO(str(best_pt)).val(
            data=args.data,
            imgsz=args.imgsz,
            device=device,
            project=args.out,
            name=f"{args.name}-val",
            exist_ok=True,
        )

    best = YOLO(str(best_pt))
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

    # Save preliminary icons.json before per-class calibration so IconDetector reads class names
    meta_dest = export_dest.with_suffix(".json")
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
    meta_dest.write_text(json.dumps(meta, indent=2) + "\n")

    ds_root = Path(args.data).parent
    class_conf = calibrate_per_class_thresholds(
        export_dest,
        meta_dest,
        ds_root / "images" / "val",
        ds_root / "labels" / "val",
        {int(k): str(v) for k, v in best.names.items()},
    )
    meta["class_conf"] = class_conf
    meta_dest.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Calibrated {len(class_conf)} per-class thresholds -> {meta_dest}")
    print(f"Exported ONNX ({export_dest.stat().st_size / 1e6:.2f} MB) -> {export_dest}")


if __name__ == "__main__":
    main()
