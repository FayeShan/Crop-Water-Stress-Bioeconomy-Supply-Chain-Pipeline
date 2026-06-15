"""CLI runner for the full HARM pipeline (Global -> Regional -> Extreme), selected via --layers."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from harm.config import load_config
from harm.utils import (
    load_crop_data, get_feature_columns, setup_logging, clean_for_json,
)
from harm.global_layer import train_global_layer
from harm.regional_layer import train_regional_layer
from harm.extreme_layer import train_extreme_layer


LAYER_ORDER = ["global", "regional", "extreme"]


def main():
    parser = argparse.ArgumentParser(description="HARM Full Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", choices=["eval", "final"], default="eval", help="Which config phase block to use (default: eval)")
    parser.add_argument("--crops", nargs="+", default=None)
    parser.add_argument(
        "--layers", nargs="+", default=None,
        help="Layers to train: global, regional, extreme, or 'all'",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, phase=args.phase if hasattr(args, 'phase') else 'eval')

    crops = args.crops or cfg.crops
    layers = args.layers or cfg.run.layers
    if "all" in layers:
        layers = LAYER_ORDER
    layers = [l for l in LAYER_ORDER if l in layers]

    if args.force:
        cfg.run.force_rerun = True

    result_root = cfg.paths.result_root
    data_root = cfg.paths.data_root
    train_years = cfg.train_years

    print("=" * 60)
    print("HARM Pipeline")
    print(f"  Layers: {' → '.join(layers)}")
    print(f"  Crops:  {crops}")
    print(f"  Years:  {train_years[0]}-{train_years[-1]}")
    print("=" * 60)

    all_results = {}

    for crop in crops:
        print(f"\n{'━' * 50}")
        print(f"  {crop}")
        print(f"{'━' * 50}")

        crop_root = result_root / crop
        global_dir = crop_root / "global"
        regional_dir = crop_root / "regional"
        extreme_dir = crop_root / "extreme"

        setup_logging(crop_root / "logs", crop)

        try:
            df = load_crop_data(data_root, crop, years=train_years)
            feature_cols = get_feature_columns(df, crop, cfg.features)

            crop_summary = {}

            if "global" in layers:
                crop_summary["global"] = train_global_layer(
                    cfg, crop, df, feature_cols, global_dir,
                )

            if "regional" in layers:
                crop_summary["regional"] = train_regional_layer(
                    cfg, crop, df, feature_cols, global_dir, regional_dir,
                )

            if "extreme" in layers:
                crop_summary["extreme"] = train_extreme_layer(
                    cfg, crop, df, feature_cols,
                    global_dir, regional_dir, extreme_dir,
                )

            crop_summary["crop"] = crop
            crop_summary["layers"] = layers
            crop_summary["timestamp"] = datetime.now().isoformat()

            with open(crop_root / "pipeline_summary.json", "w") as f:
                json.dump(clean_for_json(crop_summary), f, indent=2)

            all_results[crop] = "done"
            print(f"  ✓ {crop} complete")

        except NotImplementedError as e:
            print(f"  ⚠ {crop}: {e}")
            all_results[crop] = f"not_implemented: {e}"

        except Exception as e:
            logging.exception(f"Failed for {crop}")
            print(f"  ✗ {crop}: {e}")
            all_results[crop] = f"failed: {e}"

    print(f"\n{'=' * 60}")
    print("Results:")
    for crop, status in all_results.items():
        print(f"  {crop}: {status}")


if __name__ == "__main__":
    main()
