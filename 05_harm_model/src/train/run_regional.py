"""CLI runner for HARM Regional layer training; requires the Global layer's val_predictions.parquet first."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

from harm.config import load_config
from harm.utils import load_crop_data, get_feature_columns, setup_logging
from harm.regional_layer import train_regional_layer
from harm.metrics import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description="HARM Regional Layer Training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", choices=["eval", "final"], default="eval", help="Which config phase block to use (default: eval)")
    parser.add_argument("--crops", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--weighted", action="store_true",
                        help="Use global_weighted as base (output to regional_weighted)")
    parser.add_argument("--both", action="store_true",
                        help="Run both unweighted and weighted regional")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, phase=args.phase if hasattr(args, 'phase') else 'eval')
    crops = args.crops or cfg.crops
    if args.force:
        cfg.run.force_rerun = True
    if args.data_root:
        cfg.paths.data_root = Path(args.data_root)

    result_root = cfg.paths.result_root
    if args.tag:
        result_root = result_root / args.tag
    data_root = cfg.paths.data_root
    splits = cfg.validation_splits()

    print("=" * 60)
    print("HARM Regional Layer Training")
    print(f"  Crops:  {crops}")
    print(f"  Output: {result_root}")
    print("=" * 60)

    # Determine which variants to run
    if args.both:
        variants = [
            ("global", "regional"),
            ("global_weighted", "regional_weighted"),
        ]
    elif args.weighted:
        variants = [("global_weighted", "regional_weighted")]
    else:
        variants = [("global", "regional")]

    results = {}

    for crop in crops:
        print(f"\n{'─' * 40}")
        print(f"Processing: {crop}")
        print(f"{'─' * 40}")

        for global_subdir, regional_subdir in variants:

            for split_idx, (train_years, eval_years) in enumerate(splits):
                split_label = f"split{split_idx+1}" if len(splits) > 1 else ""
                base_dir = result_root / crop / split_label if split_label else result_root / crop

                global_dir = base_dir / global_subdir
                regional_dir = base_dir / regional_subdir
                result_key = f"{crop}/{regional_subdir}"

                if not (global_dir / "val_predictions.parquet").exists():
                    print(f"  [SKIP] No {global_subdir} results for {crop}.")
                    results[result_key] = {"status": "skipped", "reason": f"no_{global_subdir}"}
                    continue

                if not cfg.run.force_rerun and (regional_dir / "summary.json").exists():
                    print(f"  [SKIP] {result_key} already done. Use --force.")
                    results[result_key] = {"status": "skipped"}
                    continue

                print(f"\n  [{regional_subdir.upper()}] base={global_subdir}")
                setup_logging(base_dir / "logs", crop)

                try:
                    df = load_crop_data(data_root, crop, years=train_years)
                    feature_cols = get_feature_columns(df, crop, cfg.features)

                    summary = train_regional_layer(
                        cfg, crop, df, feature_cols, global_dir, regional_dir,
                    )

                    print(f"\n  Regional Results for {crop} ({regional_subdir}):")
                    print(f"    Regions found:   {summary['n_regions']}")
                    print(f"    Regions adopted: {summary['n_trained']}")
                    print(f"    Regions skipped: {summary['n_skipped']}")
                    print()

                    for r_str, r_info in summary.get("region_results", {}).items():
                        status = "✓" if r_info.get("use_regional") else "✗"
                        improvement = r_info.get("nrmse_improvement_pct", 0)
                        g_r2 = r_info.get("global_r2", 0)
                        c_r2 = r_info.get("combined_r2", 0)
                        n = r_info.get("n_samples", 0)
                        print(
                            f"    {status} Region {r_str:>2}: "
                            f"n={n:>8,}  "
                            f"Global R²={g_r2:.4f}  "
                            f"Combined R²={c_r2:.4f}  "
                            f"NRMSE improvement={improvement:+.2f}%"
                        )

                    _run_regional_held_out_eval(
                        cfg, crop, global_dir, regional_dir,
                        data_root, eval_years, summary,
                    )

                    results[result_key] = {"status": "done", "summary": summary}

                except Exception as e:
                    logging.exception(f"Failed for {crop}")
                    results[result_key] = {"status": "failed", "error": str(e)}
                    print(f"  [FAIL] {e}")

    print("\n" + "=" * 60)
    print("Summary:")
    for key, r in results.items():
        if r["status"] == "done":
            s = r["summary"]
            print(f"  {key}: {s['n_trained']}/{s['n_regions']} regions adopted")
        else:
            print(f"  {key}: {r['status']}")
    print("=" * 60)


def _run_regional_held_out_eval(cfg, crop, global_dir, regional_dir,
                                data_root, eval_years, summary):
    """Evaluate Global+Regional on the held-out eval set.

    Per sample: start from the Global prediction, then add the regional residual
    correction where the sample's region has an adopted Regional model.
    """
    if not eval_years:
        return

    df_eval = load_crop_data(data_root, crop, years=eval_years)
    if len(df_eval) == 0:
        print(f"    EVAL: no data for {eval_years}, skipping")
        return

    if "regiontype" not in df_eval.columns:
        print(f"    EVAL: no regiontype column, skipping regional eval")
        return

    global_scaler = joblib.load(global_dir / "scaler.joblib")
    with open(global_dir / "feature_names.json") as f:
        features = json.load(f)

    global_q50 = XGBRegressor()
    global_q50.load_model(str(global_dir / "model_q50.json"))

    global_q10, global_q90 = None, None
    if (global_dir / "model_q10.json").exists():
        global_q10 = XGBRegressor()
        global_q10.load_model(str(global_dir / "model_q10.json"))
    if (global_dir / "model_q90.json").exists():
        global_q90 = XGBRegressor()
        global_q90.load_model(str(global_dir / "model_q90.json"))

    X_eval = df_eval[features].values
    y_eval = df_eval[crop].values
    X_eval_s = global_scaler.transform(X_eval)

    # Start with Global predictions
    y_pred = np.maximum(global_q50.predict(X_eval_s), 0)
    y_pred_q10 = np.maximum(global_q10.predict(X_eval_s), 0) if global_q10 else None
    y_pred_q90 = np.maximum(global_q90.predict(X_eval_s), 0) if global_q90 else None

    # Apply Regional corrections where adopted
    trained_regions = summary.get("trained_regions", [])
    n_corrected = 0

    for region_str in trained_regions:
        region_dir_i = regional_dir / region_str
        if not (region_dir_i / "model_q50.json").exists():
            continue

        r_scaler = joblib.load(region_dir_i / "scaler.joblib")
        r_q50 = XGBRegressor()
        r_q50.load_model(str(region_dir_i / "model_q50.json"))

        r_q10, r_q90 = None, None
        if (region_dir_i / "model_q10.json").exists():
            r_q10 = XGBRegressor()
            r_q10.load_model(str(region_dir_i / "model_q10.json"))
        if (region_dir_i / "model_q90.json").exists():
            r_q90 = XGBRegressor()
            r_q90.load_model(str(region_dir_i / "model_q90.json"))

        mask = df_eval["regiontype"].astype(int).astype(str) == region_str
        if mask.sum() == 0:
            continue

        X_region_s = r_scaler.transform(X_eval_s[mask])

        # Add residual correction
        y_pred[mask] = np.maximum(y_pred[mask] + r_q50.predict(X_region_s), 0)

        if y_pred_q10 is not None and r_q10 is not None:
            y_pred_q10[mask] = np.maximum(y_pred_q10[mask] + r_q10.predict(X_region_s), 0)
        if y_pred_q90 is not None and r_q90 is not None:
            y_pred_q90[mask] = np.maximum(y_pred_q90[mask] + r_q90.predict(X_region_s), 0)

        n_corrected += int(mask.sum())

    metrics = evaluate_predictions(y_eval, y_pred, y_pred_q10, y_pred_q90, prefix="")

    eval_df = pd.DataFrame([{
        "split": "held_out",
        "eval_years": f"{eval_years[0]}-{eval_years[-1]}",
        "n_corrected_by_regional": n_corrected,
        "n_regions_adopted": len(trained_regions),
        **metrics,
    }])
    eval_df.to_csv(regional_dir / "eval_metrics.csv", index=False)

    print(f"\n    EVAL {eval_years[0]}-{eval_years[-1]} (Global+Regional):")
    print(f"      R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.0f}  "
          f"NRMSE={metrics.get('nrmse', 0):.6f}  MAE={metrics['mae']:.0f}  "
          f"n={int(metrics['n_samples'])}  corrected={n_corrected}")
    if metrics.get("hit_rate_p90") is not None:
        print(f"      hit_rate_p90={metrics['hit_rate_p90']:.4f}  "
              f"picp_80={metrics.get('picp_80', 'N/A')}")


if __name__ == "__main__":
    main()
