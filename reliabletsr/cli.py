import argparse
import json
from pathlib import Path

from PIL import Image

from . import MODEL_ID, SCHEMA_VERSION
from .dataset import prepare
from .errors import compare
from .evaluation import evaluate
from .model import TATR
from .risk import assess
from .structure import build_grid, parse_xml


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def process(image_path, model, xml_path=None, group="upload", detection_threshold=0.5,
            raw_threshold=0.05, min_iou=0.5):
    image_path = Path(image_path)
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    detections = model.detect(image, raw_threshold=raw_threshold)
    prediction = build_grid(detections, threshold=detection_threshold)
    result = {"schema_version": SCHEMA_VERSION, "image": str(image_path),
              "image_size": [image.width, image.height], "group": group,
              "model": {"id": MODEL_ID, "revision": model.revision, "source": model.source},
              "settings": {"detection_threshold": detection_threshold,
                           "raw_threshold": raw_threshold, "min_iou": min_iou,
                           "device": model.device},
              "raw_detections": detections, "prediction": prediction,
              "risk": assess(prediction, detections, [image.width, image.height])}
    if xml_path:
        gt_raw = parse_xml(xml_path)
        gt = build_grid(gt_raw["objects"], threshold=0.0)
        result["ground_truth"] = {"xml": str(xml_path), "objects": gt_raw["objects"], "grid": gt}
        result["comparison"] = compare(gt, prediction, min_iou=min_iou)
    return result


def main():
    parser = argparse.ArgumentParser(description="ReliableTSR Phase 1")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Download only official structure test archives and select pairs")
    p.add_argument("--data", default="data")
    p.add_argument("--count", type=int, default=400)
    p.add_argument("--dev-count", type=int, default=250)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--keep-archives", action="store_true")
    p = sub.add_parser("infer", help="Run the real TATR checkpoint on selected test pairs")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="data/results")
    p.add_argument("--group", choices=["development", "final", "both"], default="both")
    p.add_argument("--limit", type=int)
    p.add_argument("--local-model")
    p.add_argument("--cache-dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--detection-threshold", type=float, default=0.5)
    p = sub.add_parser("predict", help="Run prediction-only warnings on one uploaded image")
    p.add_argument("image")
    p.add_argument("--out", default="prediction.json")
    p.add_argument("--local-model")
    p.add_argument("--cache-dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--evaluation", default="data/evaluation.json", help="Development-tuned threshold report")
    p = sub.add_parser("evaluate", help="Tune on development, score separate final group")
    p.add_argument("--results", default="data/results")
    p.add_argument("--out", default="data/evaluation.json")
    p = sub.add_parser("recompare", help="Refresh GT matches from saved detections without rerunning TATR")
    p.add_argument("--results", default="data/results")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.data, args.count, args.dev_count, args.seed, args.keep_archives)
        print(json.dumps({k: v for k, v in manifest.items() if k != "pairs"}, indent=2))
    elif args.command == "evaluate":
        report = evaluate(args.results)
        write_json(args.out, report)
        print(json.dumps(report, indent=2))
    elif args.command == "recompare":
        files = sorted(Path(args.results).rglob("*.json"))
        count = 0
        for path in files:
            record = json.loads(path.read_text(encoding="utf-8"))
            if "ground_truth" not in record:
                continue
            record["comparison"] = compare(record["ground_truth"]["grid"],
                                           record["prediction"],
                                           min_iou=record["settings"]["min_iou"])
            record["risk"] = assess(record["prediction"], record["raw_detections"],
                                    record["image_size"])
            write_json(path, record)
            count += 1
        print(f"Refreshed comparisons in {count} saved records; rerun evaluate.")
    else:
        model = TATR(args.local_model, args.cache_dir, args.device)
        if args.command == "predict":
            result = process(args.image, model)
            calibration = Path(args.evaluation)
            if calibration.exists():
                tuning = json.loads(calibration.read_text(encoding="utf-8"))["development_tuning"]
                result["risk"] = assess(result["prediction"], result["raw_detections"],
                                        result["image_size"], threshold=tuning["risk_threshold"],
                                        confidence_floor=tuning["confidence_floor"])
            write_json(args.out, result)
            print(json.dumps({"risk": result["risk"], "cells": len(result["prediction"]["cells"])}, indent=2))
        else:
            data = Path(args.data) / "pairs"
            groups = ["development", "final"] if args.group == "both" else [args.group]
            pairs = [(group, xml, next((data/group/"images").glob(xml.stem + ".*"), None))
                     for group in groups for xml in sorted((data/group/"xml").glob("*.xml"))]
            pairs = [(g, x, img) for g, x, img in pairs if img and img.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if args.limit:
                pairs = pairs[:args.limit]
            if not pairs:
                raise SystemExit("No selected image/XML pairs found. Run prepare first.")
            for index, (group, xml, image) in enumerate(pairs, 1):
                result = process(image, model, xml, group, args.detection_threshold)
                write_json(Path(args.out) / group / (xml.stem + ".json"), result)
                print(f"{index}/{len(pairs)} {group} {xml.stem}: {len(result['prediction']['cells'])} cells, "
                      f"{len(result['comparison']['errors'])} error labels", flush=True)


if __name__ == "__main__":
    main()
