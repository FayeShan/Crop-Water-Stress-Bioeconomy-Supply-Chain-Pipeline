"""CLI runner for HARM Global layer training (unweighted, weighted, or both via shared Optuna)."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

from harm.config import load_config
from harm.utils import (
    load_crop_data, get_feature_columns, setup_logging,
    setup_gpu_environment, compute_sample_weights, clean_for_json,
)
from harm.global_layer import train_global_layer
from harm.metrics import evaluate_predictions, temporal_cv_splits
from harm.training import train_quantile_models, predict_with_quantiles
from sklearn.preprocessing import MinMaxScaler


def main():
    parser = argparse.ArgumentParser(description="HARM Global Layer Training")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--phase", choices=["eval", "final"], default="eval", help="Which config phase block to use (default: eval)")
    parser.add_argument("--crops", nargs="+", default=None, help="Override crop list")
    parser.add_argument("--fixed", action="store_true", help="Use fixed params (skip Optuna)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if results exist")
    parser.add_argument("--rolling", action="store_true", help="Override to rolling validation mode")
    parser.add_argument("--weight", action="store_true", help="Weighted only (full Optuna + weighted training)")
    parser.add_argument("--weight-only", action="store_true", help="Only run weighted variant, reuse existing unweighted params (fast)")
    parser.add_argument("--both", action="store_true", help="Train BOTH unweighted + weighted (one Optuna run)")
    parser.add_argument("--data-root", type=str, default=None, help="Override data_root path")
    parser.add_argument("--tag", type=str, default=None, help="Experiment tag")
    args = parser.parse_args()

    cfg = load_config(args.config, phase=args.phase if hasattr(args, 'phase') else 'eval')

    crops = args.crops or cfg.crops
    if args.fixed:
        cfg.global_layer["tuning_method"] = "fixed"
    if args.force:
        cfg.run.force_rerun = True
    if args.rolling:
        cfg._validation["mode"] = "rolling"
    if args.data_root:
        cfg.paths.data_root = Path(args.data_root)

    if args.both:
        run_mode = "both"
    elif args.weight_only:
        run_mode = "weight_only"
    elif args.weight:
        run_mode = "weighted"
        cfg.global_layer.setdefault("sample_weight", {})["enabled"] = True
    else:
        run_mode = "unweighted"

    result_root = cfg.paths.result_root
    if args.tag:
        result_root = result_root / args.tag

    data_root = cfg.paths.data_root
    splits = cfg.validation_splits()
    mode = cfg.validation_mode
    sw_cfg = cfg.global_layer.get("sample_weight", {})

    print("=" * 60)
    print("HARM Global Layer Training")
    print(f"  Config:       {args.config}")
    print(f"  Crops:        {crops}")
    print(f"  Validation:   {mode} ({len(splits)} split{'s' if len(splits) > 1 else ''})")
    for i, (tr, ev) in enumerate(splits):
        print(f"    split {i+1}: train {tr[0]}-{tr[-1]} → eval {ev[0]}-{ev[-1]}")
    print(f"  Tuning:       {cfg.global_layer['tuning_method']}")
    print(f"  Mode:         {run_mode}")
    if run_mode in ("weighted", "both"):
        print(f"  Weight:       alpha={sw_cfg.get('alpha', 1.0)}, "
              f"q={sw_cfg.get('reference_quantile', 0.9)}")
    print(f"  GPU:          {cfg.common.use_gpu}")
    print(f"  Output:       {result_root}")
    print("=" * 60)

    results = {}

    for crop in crops:
        print(f"\n{'─' * 40}")
        print(f"Processing: {crop}")
        print(f"{'─' * 40}")

        for split_idx, (train_years, eval_years) in enumerate(splits):
            split_label = f"split{split_idx+1}" if len(splits) > 1 else ""
            split_tag = f" [{split_label}]" if split_label else ""

            base_dir = result_root / crop / split_label if split_label else result_root / crop
            model_dir = base_dir / "global"
            weighted_dir = base_dir / "global_weighted"

            # Skip check
            skip = False
            if not cfg.run.force_rerun:
                if run_mode == "both":
                    skip = (model_dir / "summary.json").exists() and (weighted_dir / "summary.json").exists()
                elif run_mode == "weight_only":
                    skip = (weighted_dir / "summary.json").exists()
                else:
                    skip = (model_dir / "summary.json").exists()
            if skip:
                print(f"  [SKIP]{split_tag} {crop} already done. Use --force to re-run.")
                results[f"{crop}{split_tag}"] = {"status": "skipped"}
                continue

            setup_logging(base_dir / "logs", crop)

            try:
                df = load_crop_data(data_root, crop, years=train_years)
                feature_cols = get_feature_columns(df, crop, cfg.features)

                # Unweighted training (also step 1 of --both)
                if run_mode in ("unweighted", "both"):
                    cfg.global_layer.setdefault("sample_weight", {})["enabled"] = False
                    summary = train_global_layer(cfg, crop, df, feature_cols, model_dir)

                    m = summary["metrics"]["mean"]
                    print(f"\n  [UNWEIGHTED]{split_tag}")
                    print(f"    CV   R²={m.get('r2', 0):.4f}  RMSE={m.get('rmse', 0):.0f}  "
                          f"NRMSE={m.get('nrmse', 0):.6f}  MAE={m.get('mae', 0):.0f}")
                    _run_held_out_eval(crop, model_dir, data_root, eval_years)
                    _report_hair_importance(model_dir)

                if run_mode in ("weighted", "both", "weight_only"):
                    if run_mode in ("both", "weight_only"):
                        # Reuse params from existing unweighted run
                        if not (model_dir / "best_params.json").exists():
                            print(f"  [SKIP] {crop}: no unweighted results to reuse. Run without --weight-only first.")
                            continue
                        _train_weighted_variant(
                            cfg, crop, df, feature_cols,
                            source_dir=model_dir,
                            output_dir=weighted_dir,
                            sw_cfg=sw_cfg,
                        )
                        w_dir = weighted_dir
                    else:
                        # --weight: full Optuna + weighted, save to global_weighted/
                        cfg.global_layer.setdefault("sample_weight", {})["enabled"] = True
                        summary = train_global_layer(cfg, crop, df, feature_cols, weighted_dir)
                        w_dir = weighted_dir

                    with open(w_dir / "summary.json") as f:
                        ws = json.load(f)
                    m2 = ws["metrics"]["mean"]
                    print(f"\n  [WEIGHTED]{split_tag}")
                    print(f"    CV   R²={m2.get('r2', 0):.4f}  RMSE={m2.get('rmse', 0):.0f}  "
                          f"NRMSE={m2.get('nrmse', 0):.6f}  MAE={m2.get('mae', 0):.0f}")
                    _run_held_out_eval(crop, w_dir, data_root, eval_years)
                    _report_hair_importance(w_dir)

                results[f"{crop}{split_tag}"] = {"status": "done"}

            except Exception as e:
                logging.exception(f"Failed for {crop}{split_tag}")
                results[f"{crop}{split_tag}"] = {"status": "failed", "error": str(e)}
                print(f"  [FAIL]{split_tag} {e}")

    print("\n" + "=" * 60)
    print("Summary:")
    for key, r in results.items():
        print(f"  {key}: {r['status']}")
    print("=" * 60)


def _train_weighted_variant(cfg, crop, df, feature_cols, source_dir, output_dir, sw_cfg):
    """Train weighted models reusing hyperparams from unweighted run. Skips Optuna."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(source_dir / "best_params.json") as f:
        best_params = json.load(f)
    with open(source_dir / "feature_names.json") as f:
        saved_features = json.load(f)

    logging.info(f"[WEIGHTED] Reusing params from {source_dir}")

    gpu_available, n_gpus, device = setup_gpu_environment(cfg.common.use_gpu)
    quantiles = cfg.common.quantiles
    random_state = cfg.common.random_state
    gl_cfg = cfg.global_layer

    X = df[saved_features]
    y = df[crop]
    dates = pd.to_datetime(df["time"])
    grid_ids = df["grid50_id"]

    sample_weights = compute_sample_weights(
        y.values, enabled=True,
        alpha=sw_cfg.get("alpha", 1.0),
        reference_quantile=sw_cfg.get("reference_quantile", 0.9),
        max_weight=sw_cfg.get("max_weight", 10.0),
    )

    # CV with weights, for metrics only
    cv_splits = temporal_cv_splits(
        dates,
        test_months=gl_cfg.get("test_months", 24),
        min_train_months=gl_cfg.get("min_train_months", 12),
    )

    metrics_list, fi_records, all_val_preds = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = MinMaxScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)

        fold_w = sample_weights[train_idx] if sample_weights is not None else None
        fold_models = train_quantile_models(
            X_tr_s, y_tr.values, best_params,
            device, quantiles, random_state, fold_w,
        )

        q_preds = predict_with_quantiles(fold_models, X_val_s)
        y_pred = np.maximum(q_preds[0.5], 0)
        y_q10 = np.maximum(q_preds.get(0.1, y_pred), 0)
        y_q90 = np.maximum(q_preds.get(0.9, y_pred), 0)

        fold_metrics = evaluate_predictions(y_val.values, y_pred, y_q10, y_q90, prefix="")
        metrics_list.append({"fold": fold_idx, **fold_metrics})

        for feat, imp in zip(saved_features, fold_models[0.5].feature_importances_):
            fi_records.append({"fold": fold_idx, "feature": feat, "importance": float(imp)})

        all_val_preds.append(pd.DataFrame({
            "fold": fold_idx,
            "time": dates.iloc[val_idx].values,
            "grid50_id": grid_ids.iloc[val_idx].values,
            "y_true": y_val.values,
            "y_pred": y_pred,
            "y_pred_q10": y_q10,
            "y_pred_q90": y_q90,
            "residual": y_val.values - y_pred,
        }))

    # Train final models on ALL data with weights
    final_scaler = MinMaxScaler().fit(X)
    X_scaled = final_scaler.transform(X)
    final_models = train_quantile_models(
        X_scaled, y.values, best_params,
        device, quantiles, random_state, sample_weights,
    )

    for q, model in final_models.items():
        model.save_model(str(output_dir / f"model_q{int(q * 100)}.json"))
    joblib.dump(final_scaler, output_dir / "scaler.joblib")
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(saved_features, f, indent=2)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(output_dir / "cv_metrics.csv", index=False)
    pd.DataFrame(fi_records).to_csv(output_dir / "feature_importance.csv", index=False)
    pd.concat(all_val_preds, ignore_index=True).to_parquet(
        output_dir / "val_predictions.parquet", index=False
    )

    mean_metrics = metrics_df.mean(numeric_only=True).to_dict()
    summary = {
        "layer": "global", "variant": "weighted", "crop": crop,
        "n_features": len(saved_features), "n_samples": len(X),
        "n_folds": len(cv_splits),
        "sample_weight": {
            "alpha": sw_cfg.get("alpha", 1.0),
            "reference_quantile": sw_cfg.get("reference_quantile", 0.9),
        },
        "metrics": {"mean": clean_for_json(mean_metrics)},
        "best_params": best_params,
        "source_params_from": str(source_dir),
        "model_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"[WEIGHTED] Done. R²={mean_metrics.get('r2', 0):.4f}, "
                 f"RMSE={mean_metrics.get('rmse', 0):.0f}")


def _run_held_out_eval(crop, model_dir, data_root, eval_years):
    """Load trained model, predict on held-out eval set, save eval_metrics.csv."""
    if not eval_years:
        return

    df_eval = load_crop_data(data_root, crop, years=eval_years)
    if len(df_eval) == 0:
        print(f"    EVAL: no data for {eval_years}, skipping")
        return

    scaler = joblib.load(model_dir / "scaler.joblib")
    model_q50 = XGBRegressor()
    model_q50.load_model(str(model_dir / "model_q50.json"))

    model_q10, model_q90 = None, None
    if (model_dir / "model_q10.json").exists():
        model_q10 = XGBRegressor()
        model_q10.load_model(str(model_dir / "model_q10.json"))
    if (model_dir / "model_q90.json").exists():
        model_q90 = XGBRegressor()
        model_q90.load_model(str(model_dir / "model_q90.json"))

    with open(model_dir / "feature_names.json") as f:
        saved_features = json.load(f)

    X_eval = df_eval[saved_features]
    y_eval = df_eval[crop].values
    X_eval_s = scaler.transform(X_eval)

    y_pred = np.maximum(model_q50.predict(X_eval_s), 0)
    y_lower = np.maximum(model_q10.predict(X_eval_s), 0) if model_q10 else None
    y_upper = np.maximum(model_q90.predict(X_eval_s), 0) if model_q90 else None

    metrics = evaluate_predictions(y_eval, y_pred, y_lower, y_upper, prefix="")

    pd.DataFrame([{
        "split": "held_out",
        "eval_years": f"{eval_years[0]}-{eval_years[-1]}",
        **metrics,
    }]).to_csv(model_dir / "eval_metrics.csv", index=False)

    print(f"    EVAL {eval_years[0]}-{eval_years[-1]}:  "
          f"R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.0f}  "
          f"NRMSE={metrics.get('nrmse', 0):.6f}  MAE={metrics['mae']:.0f}  "
          f"n={metrics['n_samples']:.0f}")


def _report_hair_importance(model_dir):
    """Check if HAir appears in feature importance and print its ranking."""
    fi_path = model_dir / "feature_importance.csv"
    if not fi_path.exists():
        return

    fi = pd.read_csv(fi_path)
    avg_fi = fi.groupby("feature")["importance"].mean().sort_values(ascending=False)

    hair_cols = [c for c in avg_fi.index if "hair" in c.lower() or "irrigat" in c.lower()]

    if hair_cols:
        total_features = len(avg_fi)
        for col in hair_cols:
            rank = list(avg_fi.index).index(col) + 1
            imp = avg_fi[col]
            pct = imp / avg_fi.sum() * 100
            print(f"  [HAir] {col}: rank {rank}/{total_features}, "
                  f"importance={imp:.4f} ({pct:.1f}%)")
    else:
        print("  [HAir] No irrigated area feature found in columns.")

    print(f"  [TOP-10 Features]")
    for i, (feat, imp) in enumerate(avg_fi.head(10).items(), 1):
        marker = " ← HAir" if feat in hair_cols else ""
        print(f"    {i:2d}. {feat}: {imp:.4f}{marker}")


if __name__ == "__main__":
    main()
