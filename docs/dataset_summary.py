import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


def parse_id_from_filename(path: str) -> str:
    name = os.path.basename(path)
    return name.split("_")[0]


def scan_images(image_root: str, sample_for_resolution: int = 400) -> Dict:
    exts = (".jpg", ".jpeg", ".png")
    files: List[str] = []
    for root, _, names in os.walk(image_root):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))

    id_to_files: Dict[str, List[str]] = defaultdict(list)
    for p in files:
        pid = parse_id_from_filename(p)
        id_to_files[pid].append(p)

    # Per-id image count stats
    counts = np.array([len(v) for v in id_to_files.values()], dtype=np.int32) if id_to_files else np.array([], dtype=np.int32)

    # Sample some files for resolution analysis
    sample = files if len(files) <= sample_for_resolution else random.sample(files, sample_for_resolution)
    widths: List[int] = []
    heights: List[int] = []
    for p in sample:
        try:
            with Image.open(p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue

    def stats_arr(a: np.ndarray) -> Dict:
        if a.size == 0:
            return {"count": 0}
        return {
            "count": int(a.size),
            "min": float(np.min(a)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
        }

    res = {
        "n_images": len(files),
        "n_ids_with_images": len(id_to_files),
        "images_per_id": stats_arr(counts),
        "resolution_sample_size": len(widths),
        "width_stats": stats_arr(np.array(widths, dtype=np.float64)),
        "height_stats": stats_arr(np.array(heights, dtype=np.float64)),
    }
    return res


def scan_csv(csv_path: str) -> Dict:
    df = pd.read_csv(csv_path)
    info: Dict = {"csv_path": csv_path}
    info["n_rows"] = int(len(df))
    info["columns"] = list(df.columns)
    if "id" in df.columns:
        info["n_unique_id"] = int(df["id"].astype(str).nunique())
    # Target stats
    if "cena" in df.columns:
        y = df["cena"].dropna().to_numpy(dtype=np.float64)
        if y.size > 0:
            info["target_stats"] = {
                "count": int(y.size),
                "min": float(np.min(y)),
                "p10": float(np.percentile(y, 10)),
                "p25": float(np.percentile(y, 25)),
                "median": float(np.median(y)),
                "p75": float(np.percentile(y, 75)),
                "p90": float(np.percentile(y, 90)),
                "max": float(np.max(y)),
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
            }
    # Missingness
    miss = df.isna().sum()
    info["missing_counts"] = {c: int(miss[c]) for c in df.columns}
    # Category top counts (limited)
    cat_cols = [
        c for c in ["grad", "opstina", "kvart", "broj_soba", "spratnost", "stanje", "grejanje", "lift", "podrum", "terasa"]
        if c in df.columns
    ]
    topk = {}
    for c in cat_cols:
        vc = df[c].astype(str).fillna("missing").value_counts().head(10)
        topk[c] = {str(k): int(v) for k, v in vc.items()}
    info["top_categories"] = topk
    return info


def compute_overlap(csv_info: Dict, img_info: Dict, csv_path: str, image_root: str) -> Dict:
    # Build id sets
    df = pd.read_csv(csv_path)
    ids_csv = set(df["id"].astype(str).unique()) if "id" in df.columns else set()

    # Re-scan ids from images without counting resolutions again
    exts = (".jpg", ".jpeg", ".png")
    files: List[str] = []
    for root, _, names in os.walk(image_root):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    ids_img = set(parse_id_from_filename(p) for p in files)

    inter = ids_csv & ids_img
    only_csv = ids_csv - ids_img
    only_img = ids_img - ids_csv

    return {
        "ids_in_csv": len(ids_csv),
        "ids_with_images": len(ids_img),
        "ids_intersection": len(inter),
        "coverage_csv_has_images_pct": float(100.0 * len(inter) / max(len(ids_csv), 1)),
        "coverage_images_in_csv_pct": float(100.0 * len(inter) / max(len(ids_img), 1)),
        "example_ids_only_csv": list(sorted(list(only_csv))[:10]),
        "example_ids_only_images": list(sorted(list(only_img))[:10]),
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize datasets for thesis section 5.1")
    ap.add_argument("--csv", type=str, default="filtered_property_data.csv")
    ap.add_argument("--images", type=str, default="property_images")
    ap.add_argument("--out_json", type=str, default="docs/dataset_summary.json")
    ap.add_argument("--out_md", type=str, default="docs/dataset_summary.md")
    ap.add_argument("--resolution_sample", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    csv_info = scan_csv(args.csv)
    img_info = scan_images(args.images, sample_for_resolution=args.resolution_sample)
    overlap = compute_overlap(csv_info, img_info, args.csv, args.images)

    out = {"csv": csv_info, "images": img_info, "overlap": overlap}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Simple Markdown summary
    lines = []
    lines.append(f"# Dataset Summary\n")
    lines.append(f"CSV: `{args.csv}` — rows: {csv_info.get('n_rows')} — unique id: {csv_info.get('n_unique_id', 'n/a')}\n")
    if "target_stats" in csv_info:
        ts = csv_info["target_stats"]
        lines.append(
            f"Target (cena) — mean: {ts['mean']:.2f}, median: {ts['median']:.2f}, std: {ts['std']:.2f}, min/max: {ts['min']:.2f}/{ts['max']:.2f}\n"
        )
    lines.append(
        f"Images: root `{args.images}` — files: {img_info['n_images']} — ids_with_images: {img_info['n_ids_with_images']}\n"
    )
    lines.append(
        f"Images per id — mean: {img_info['images_per_id'].get('mean','-')}, median: {img_info['images_per_id'].get('median','-')}\n"
    )
    lines.append(
        f"Overlap — ids_intersection: {overlap['ids_intersection']} — coverage(csv has images): {overlap['coverage_csv_has_images_pct']:.1f}%\n"
    )
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved:")
    print(f" - {args.out_json}")
    print(f" - {args.out_md}")


if __name__ == "__main__":
    main()
