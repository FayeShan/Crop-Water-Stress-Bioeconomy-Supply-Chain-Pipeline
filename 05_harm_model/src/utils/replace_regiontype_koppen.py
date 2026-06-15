"""Replace regiontype with Köppen-Geiger climate zones (1-30) in train/predict parquet datasets, writing to a new output directory."""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from harm.config import load_config
from harm.constants import ALL_CROPS, KOPPEN_LABELS

_cfg = load_config("config.yaml")
KOPPEN_NC = _cfg.paths.koppen_nc

# Default I/O paths (overridable via CLI)
TRAIN_INPUT  = _cfg.paths.data_root
TRAIN_OUTPUT = _cfg.paths.data_root_koppen
PREDICT_INPUT  = _cfg.paths.pred_data_root
PREDICT_OUTPUT = _cfg.paths.pred_data_root_koppen


def load_koppen(nc_path):
    """Load Köppen-Geiger DataArray."""
    logging.info(f"Loading Köppen-Geiger: {nc_path}")
    ds = xr.open_dataset(nc_path)
    koppen = ds["region"]
    logging.info(f"  Shape: {koppen.shape}, unique values: {len(np.unique(koppen.values[~np.isnan(koppen.values)]))}")
    return koppen


def match_koppen(koppen_da, lats, lons):
    """
    Match lat/lon arrays to Köppen region using nearest neighbor.
    Returns array of region values (1-30, NaN for ocean/missing).
    """
    vals = koppen_da.sel(
        lat=xr.DataArray(lats, dims="points"),
        lon=xr.DataArray(lons, dims="points"),
        method="nearest",
    ).values
    return vals


def process_parquet(parquet_path, output_path, koppen_da, dry_run=False):
    """Replace regiontype with Köppen zone in a parquet file."""
    logging.info(f"  Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    n = len(df)
    logging.info(f"  Rows: {n:,}")

    if "lat" not in df.columns or "lon" not in df.columns:
        logging.error(f"  No lat/lon columns!")
        return False

    if "regiontype" in df.columns:
        old_unique = sorted(df["regiontype"].dropna().unique())
        logging.info(f"  Old regiontype: {len(old_unique)} values: {old_unique}")

    koppen_vals = match_koppen(koppen_da, df["lat"].values, df["lon"].values)

    df["regiontype"] = koppen_vals

    n_nan = np.isnan(koppen_vals).sum()
    if n_nan > 0:
        logging.info(f"  {n_nan:,} NaN (ocean/missing) → filling with 0")
        df["regiontype"] = df["regiontype"].fillna(0)

    unique_regions = sorted(df["regiontype"].unique())
    logging.info(f"  New regiontype: {len(unique_regions)} Köppen zones")

    dist = df["regiontype"].value_counts().sort_index()
    for region_val, count in dist.items():
        r = int(region_val)
        label = KOPPEN_LABELS.get(r, "Unknown/Ocean")
        pct = count / n * 100
        if pct >= 1.0:  # Only show regions with >= 1%
            logging.info(f"    {r:>2} ({label:<4}): {count:>10,} ({pct:5.1f}%)")

    if dry_run:
        logging.info(f"  [DRY RUN] Not saving.")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info(f"  Saved: {output_path}")

    del df
    gc.collect()
    return True


def main():
    parser = argparse.ArgumentParser(description="Replace regiontype with Köppen-Geiger zones")
    parser.add_argument("--mode", required=True, choices=["train", "predict", "both"])
    parser.add_argument("--crops", nargs="*", default=None)
    parser.add_argument("--koppen", default=str(KOPPEN_NC))
    parser.add_argument("--train-input", default=str(TRAIN_INPUT))
    parser.add_argument("--train-output", default=str(TRAIN_OUTPUT))
    parser.add_argument("--predict-input", default=str(PREDICT_INPUT))
    parser.add_argument("--predict-output", default=str(PREDICT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true", help="Show stats only, don't save")
    args = parser.parse_args()

    crops = args.crops or ALL_CROPS
    koppen_da = load_koppen(args.koppen)

    logging.info("=" * 65)
    logging.info("  Replace regiontype with Köppen-Geiger")
    logging.info("=" * 65)
    logging.info(f"  Crops: {crops}")
    logging.info(f"  Mode: {args.mode}")
    if args.dry_run:
        logging.info(f"  *** DRY RUN ***")

    results = []

    if args.mode in ["train", "both"]:
        logging.info(f"\n{'='*65}")
        logging.info(f"  Processing TRAINING data")
        logging.info(f"{'='*65}")

        for crop in crops:
            logging.info(f"\n[TRAIN] {crop}")
            crop_dir = Path(args.train_input) / crop
            if not crop_dir.exists():
                logging.warning(f"  Not found: {crop_dir}")
                continue

            for pq in sorted(crop_dir.glob("*.parquet")):
                out = Path(args.train_output) / crop / pq.name
                ok = process_parquet(pq, out, koppen_da, args.dry_run)
                results.append({"crop": crop, "mode": "train", "file": pq.name, "ok": ok})

            gc.collect()

    if args.mode in ["predict", "both"]:
        logging.info(f"\n{'='*65}")
        logging.info(f"  Processing PREDICTION data")
        logging.info(f"{'='*65}")

        for crop in crops:
            logging.info(f"\n[PREDICT] {crop}")
            crop_dir = Path(args.predict_input) / crop
            if not crop_dir.exists():
                logging.warning(f"  Not found: {crop_dir}")
                continue

            for pq in sorted(crop_dir.glob("*.parquet")):
                out = Path(args.predict_output) / crop / pq.name
                ok = process_parquet(pq, out, koppen_da, args.dry_run)
                results.append({"crop": crop, "mode": "predict", "file": pq.name, "ok": ok})

            gc.collect()

    logging.info(f"\n{'='*65}")
    logging.info(f"  SUMMARY")
    logging.info(f"{'='*65}")
    df_r = pd.DataFrame(results)
    if len(df_r) > 0:
        logging.info(f"  Files processed: {len(df_r)}")
        logging.info(f"  Successful: {df_r['ok'].sum()}")

    koppen_da.close()
    logging.info("Done!")


if __name__ == "__main__":
    main()
