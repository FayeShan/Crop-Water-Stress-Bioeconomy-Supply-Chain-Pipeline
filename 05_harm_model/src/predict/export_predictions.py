#!/usr/bin/env python3
"""Export HARM final predictions to flat per-crop CSVs consumed by Stage 06 and Stage 08."""

import argparse
from pathlib import Path

import pandas as pd

from harm.config import load_config
from harm.constants import ALL_CROPS

KEEP_COLS = ["time", "lat", "lon", "year", "y_pred"]


def export_crop(crop, pred_root, output_dir, pred_col):
    src = pred_root / crop / "predictions.parquet"
    if not src.exists():
        print(f"[WARN] not found: {src}")
        return False

    df = pd.read_parquet(src)
    if pred_col not in df.columns:
        print(f"[WARN] {crop}: column {pred_col!r} missing (have {list(df.columns)})")
        return False

    df["y_pred"] = df[pred_col]
    if "year" not in df.columns and "time" in df.columns:
        df["year"] = pd.to_datetime(df["time"]).dt.year

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] {crop}: missing columns {missing}, skipping")
        return False

    out_path = output_dir / f"{crop}.csv"
    df[KEEP_COLS].to_csv(out_path, index=False)
    print(f"[OK] {crop}: {len(df):,} rows -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export HARM final predictions (predictions.parquet) to flat per-crop "
                    "{crop}.csv (time, lat, lon, year, y_pred) for Stage 06 and Stage 08.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", choices=["eval", "final"], default="final")
    parser.add_argument("--crops", nargs="+", default=None)
    parser.add_argument("--pred-col", default="y_pred_gre",
                        help="Parquet column to export as y_pred (default: y_pred_gre, the full G+R+E cascade)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <result_root>/predictions_csv)")
    args = parser.parse_args()

    cfg = load_config(args.config, phase=args.phase)
    crops = args.crops or ALL_CROPS

    pred_root = Path(cfg.paths.result_root) / "predictions"
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.paths.result_root) / "predictions_csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {pred_root}")
    print(f"Output: {output_dir}")
    print(f"Crops:  {len(crops)}")

    n_ok = sum(export_crop(crop, pred_root, output_dir, args.pred_col) for crop in crops)
    print(f"\nDone: exported {n_ok}/{len(crops)} crops.")


if __name__ == "__main__":
    main()
