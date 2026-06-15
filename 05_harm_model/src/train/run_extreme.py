"""CLI runner for the HARM Extreme layer (G+R base, or GW+RW base with --weighted)."""

import argparse
import warnings
from pathlib import Path

from harm.config import load_config
from harm.constants import UNWEIGHTED_CROPS, WEIGHTED_CROPS
from harm.extreme_layer import (
    train_extreme_layer,
    train_extreme_weighted,
)
from harm.utils import (
    load_crop_data, get_feature_columns, setup_logging,
    setup_gpu_environment, clear_memory, clear_gpu_memory,
)

warnings.filterwarnings("ignore")


def run(crops, config_path="config.yaml", phase="eval",
        data_root=None, result_root=None, force=False, weighted=False):
    """Train the Extreme layer for each requested crop."""
    cfg = load_config(config_path, phase=phase)

    data_root = Path(data_root) if data_root else cfg.paths.data_root
    result_root = Path(result_root) if result_root else cfg.paths.result_root
    train_years = cfg.train_years

    gpu_available, _, device = setup_gpu_environment(cfg.common.use_gpu)

    print("=" * 60)
    print(f"HARM Extreme Layer ({'GW+RW base' if weighted else 'G+R base'})")
    print(f"  Crops:   {crops}")
    print(f"  Data:    {data_root}")
    print(f"  Results: {result_root}")
    print(f"  Train:   {train_years[0]}-{train_years[-1]}")
    print(f"  GPU:     {device}")
    print("=" * 60, flush=True)

    results = {}

    for crop in crops:
        print(f"\n{'─'*50}")
        print(f"  Processing: {crop}{'  [weighted]' if weighted else ''}")
        print(f"{'─'*50}", flush=True)

        if weighted and crop in UNWEIGHTED_CROPS:
            print(f"  [SKIP] {crop} is assigned to the unweighted pipeline "
                  f"in constants.UNWEIGHTED_CROPS; drop --weighted to run it.")
            results[crop] = "skip_unweighted"
            continue

        extreme_subdir = "extreme_weighted" if weighted else "extreme"
        extreme_dir = result_root / crop / extreme_subdir
        if not force and (extreme_dir / "summary.json").exists():
            print("  [SKIP] Already done. Pass --force to re-run.")
            results[crop] = "skipped"
            continue

        if weighted:
            base_global_dir = result_root / crop / "global_weighted"
            if not (base_global_dir / "model_q50.json").exists():
                print(f"  [SKIP] No GW model at {base_global_dir}")
                results[crop] = "no_base"
                continue
        else:
            global_dir = result_root / crop / "global"
            regional_dir = result_root / crop / "regional"
            if not (global_dir / "model_q50.json").exists():
                print(f"  [SKIP] No Global model at {global_dir}")
                results[crop] = "no_base"
                continue
            if not (global_dir / "val_predictions.parquet").exists() and \
               not (regional_dir / "val_predictions.parquet").exists():
                print("  [SKIP] No val_predictions.parquet found")
                results[crop] = "no_base"
                continue

        try:
            df = load_crop_data(data_root, crop, years=train_years)
            feature_cols = get_feature_columns(df, crop, cfg.features)
            print(f"  Data: {len(df):,} rows, {len(feature_cols)} features",
                  flush=True)

            setup_logging(extreme_dir, crop)

            if weighted:
                summary = train_extreme_weighted(
                    cfg, crop, df, result_root, extreme_dir,
                )
            else:
                summary = train_extreme_layer(
                    cfg, crop, df, feature_cols,
                    global_dir, regional_dir, extreme_dir,
                )

            n_t = summary.get("n_trained", 0) if isinstance(summary, dict) else 0
            n_total = summary.get("n_regions", "?") if isinstance(summary, dict) else "?"
            print(f"  ✓ Extreme: {n_t}/{n_total} regions adopted")
            if isinstance(summary, dict) and summary.get("trained_regions"):
                print(f"    Adopted: {summary['trained_regions']}")
            results[crop] = "done"

            del df
            clear_memory()
            if gpu_available:
                clear_gpu_memory()

        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results[crop] = f"failed: {e}"

    print(f"\n{'='*60}\nSummary:")
    for crop, status in results.items():
        print(f"  {crop}: {status}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HARM Extreme Layer runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", choices=["eval", "final"], default="eval",
                        help="Which config phase block to use (default: eval)")
    parser.add_argument("--crops", nargs="+", default=None,
                        help="Crops to process. If omitted: (a) without --weighted, "
                             "defaults to the unweighted-pipeline crops "
                             "(cot, sgc, ri1, ri2); (b) with --weighted, all "
                             "WEIGHTED_CROPS from harm.constants.")
    parser.add_argument("--weighted", action="store_true",
                        help="Use GW+RW as base instead of G+R.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--result-root", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.crops is None:
        args.crops = list(WEIGHTED_CROPS) if args.weighted \
            else ["cot", "sgc", "ri1", "ri2"]

    run(args.crops, args.config, args.phase, args.data_root,
        args.result_root, args.force, weighted=args.weighted)
