#!/usr/bin/env python3
"""Reproducible CPU latency benchmark for Advnt Screen Parser (v3)."""

from __future__ import annotations

import argparse
import glob
import platform
import time
from pathlib import Path

import cv2

from advnt_screen_parser import IconDetector, ScreenParser


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark CPU latency for IconDetector and ScreenParser.")
    ap.add_argument("--images", default="assets/examples/*.jpg", help="Glob of test images")
    ap.add_argument("--model", default=None, help="Optional local path to icons.onnx")
    ap.add_argument("--runs", type=int, default=5, help="Iterations per image")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4], help="CPU thread counts")
    ap.add_argument("--skip-ocr", action="store_true", help="Only benchmark IconDetector")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f"No images matched: {args.images}")
    images = [(Path(p).name, cv2.imread(p, cv2.IMREAD_COLOR)) for p in paths]

    print(f"Machine: {platform.machine()} / {platform.platform()}")
    print(f"Images: {len(images)} x {args.runs} runs\n")

    print("=== IconDetector (ONNX Runtime CPU) ===")
    for t in args.threads:
        det = IconDetector(model_path=args.model, threads=t)
        det.warm_up()
        times: list[float] = []
        for _ in range(args.runs):
            for _, im in images:
                t0 = time.perf_counter()
                det.detect(im)
                times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        med = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        print(f"  threads={t}: median={med:.1f} ms, p95={p95:.1f} ms, min={times[0]:.1f} ms")

    if not args.skip_ocr:
        print("\n=== Full ScreenParser (RapidOCR PP-OCRv6-tiny + v3 Icons + UI Tree, 2 threads) ===")
        parser = ScreenParser(icon_model_path=args.model, threads=2, enable_ocr=True)
        parser.warm_up()
        for name, im in images:
            res = parser.parse_image(im)
            print(
                f"  {name:22s}: total={res['totalMs']:5.1f} ms "
                f"(ocr={res['ocrMs']:5.1f} ms, icons={res['iconsMs']:4.1f} ms, uitree={res['uitreeMs']:4.1f} ms) "
                f"| words={len(res['words']):3d}, icons={len(res['icons']):2d}, elements={len(res['elements']):3d}"
            )


if __name__ == "__main__":
    main()
