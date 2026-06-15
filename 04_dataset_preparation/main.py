#!/usr/bin/env python3
"""
data_pre — CLI entry point assembling the water-stress prediction dataset.

Usage:
    python main.py setup-cropmask
    python main.py setup-coarse-grid
    python main.py setup-region9 --thz-tif <path> --mst-tif <path>
    python main.py noaa-download
    python main.py build-train
    python main.py build-predict
    python main.py add-ha --mode both
    python main.py feature-engineering --mode both
    python main.py full-pipeline

Commands:
    setup-cropmask      Build per-crop monthly masks from water-stress NCs
    setup-coarse-grid   Build 50 km coarse grid-id look-up
    setup-region9       Build 9-class agro-climatic region look-up
    noaa-download       Download NOAA climate indices as CSV
    build-train         Build raw training parquets (merged NC + WS)
    build-predict       Build raw prediction parquets (merged NC)
    add-ha              Add ha_irrigated feature (monthly + annual)
    feature-engineering Apply lag/rolling/time/loc features
    full-pipeline       Run all assembly + FE steps in order
"""

import argparse
import logging
import sys
from pathlib import Path

from config import PipelineConfig, create_directories, load_config


def _setup_logger(log_dir: str, name: str, level: str = "INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log", mode="w")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def cmd_setup_cropmask(args, config: PipelineConfig):
    from src.setup_cropmask import generate_cropmasks
    logger = _setup_logger(config.log_dir, "setup_cropmask", config.log_level)
    generate_cropmasks(
        crops=args.crops or config.crops.all,
        ws_dir=config.water_stress.ws_dir,
        output_dir=config.reference.cropmasks_dir,
        start_year=args.start_year or config.date_range.train_start,
        end_year=args.end_year or config.date_range.train_end,
        logger=logger,
    )


def cmd_setup_coarse_grid(args, config: PipelineConfig):
    from src.setup_coarse_grid import generate_coarse_grid
    logger = _setup_logger(config.log_dir, "setup_coarse_grid", config.log_level)
    generate_coarse_grid(
        template_nc=args.template or config.reference.template_nc,
        output_parquet=config.reference.coarse50km_gridid,
        coarse_factor=args.coarse_factor,
        logger=logger,
    )


def cmd_setup_region9(args, config: PipelineConfig):
    from src.setup_region9 import generate_region9
    logger = _setup_logger(config.log_dir, "setup_region9", config.log_level)
    generate_region9(
        thz_tif=args.thz_tif,
        mst_tif=args.mst_tif,
        template_nc=args.template or config.reference.template_nc,
        output_parquet=config.reference.region9,
        logger=logger,
    )


def cmd_noaa_download(args, config: PipelineConfig):
    from src.noaa_download import download_indices
    logger = _setup_logger(config.log_dir, "noaa_download", config.log_level)
    download_indices(
        output_dir=config.noaa.dir,
        logger=logger,
        cleanup_tim=not args.keep_tim,
    )


def cmd_build_train(args, config: PipelineConfig):
    from src.build_train_parquet import build_train_parquets
    logger = _setup_logger(config.log_dir, "build_train", config.log_level)
    build_train_parquets(
        crops=args.crops or config.crops.all,
        predictors_dir=config.predictors.merged_dir,
        ws_dir=config.water_stress.ws_dir,
        cropmasks_dir=config.reference.cropmasks_dir,
        output_dir=config.data.base_train_dir,
        predictor_vars=config.predictor_vars,
        start_year=args.start_year or config.date_range.train_start,
        end_year=args.end_year or config.date_range.train_end,
        overwrite=args.overwrite,
        logger=logger,
    )


def cmd_build_predict(args, config: PipelineConfig):
    from src.build_predict_parquet import build_predict_parquets
    logger = _setup_logger(config.log_dir, "build_predict", config.log_level)
    build_predict_parquets(
        crops=args.crops or config.crops.all,
        predictors_dir=config.predictors.merged_dir,
        cropmasks_dir=config.reference.cropmasks_dir,
        output_dir=config.data.base_predict_dir,
        predictor_vars=config.predictor_vars,
        start_year=args.start_year or config.date_range.predict_start,
        end_year=args.end_year or config.date_range.predict_end,
        overwrite=args.overwrite,
        logger=logger,
    )


def cmd_add_ha(args, config: PipelineConfig):
    from src.add_irrigated_ha import add_irrigated_ha
    logger = _setup_logger(config.log_dir, "add_ha", config.log_level)
    crops = args.crops or config.crops.all

    if args.mode in ("train", "both"):
        add_irrigated_ha(
            crops=crops,
            input_dir=config.data.base_train_dir,
            output_dir=config.data.base_train_dir,   # overwrite in place
            monthly_dir=config.iha.monthly_dir,
            annual_dir=config.iha.annual_dir,
            aggregate_crops=config.crops.aggregate,
            logger=logger,
        )
    if args.mode in ("predict", "both"):
        add_irrigated_ha(
            crops=crops,
            input_dir=config.data.base_predict_dir,
            output_dir=config.data.base_predict_dir,  # overwrite in place
            monthly_dir=config.iha.monthly_dir,
            annual_dir=config.iha.annual_dir,
            aggregate_crops=config.crops.aggregate,
            logger=logger,
        )


def cmd_feature_engineering(args, config: PipelineConfig):
    from src.feature_engineering import run_feature_engineering
    logger = _setup_logger(config.log_dir, "feature_engineering", config.log_level)
    fe = config.feature_engineering
    run_feature_engineering(
        mode=args.mode,
        crops=args.crops or config.crops.all,
        train_input=config.data.base_train_dir,
        predict_input=config.data.base_predict_dir,
        train_output=config.data.train_dir,
        predict_output=config.data.predict_dir,
        train_year_range=(config.date_range.train_start, config.date_range.train_end),
        predict_year_range=(config.date_range.predict_start, config.date_range.predict_end),
        grid50_path=config.reference.coarse50km_gridid,
        region_path=config.reference.region9,
        noaa_dir=config.noaa.dir,
        lag_vars=fe.lag_vars,
        lag_range=range(fe.lag_range[0], fe.lag_range[-1] + 1),
        roll_vars=fe.roll_vars,
        roll_windows=fe.roll_windows,
        prepend_years_from_train=fe.prepend_years_from_train,
        overwrite=args.overwrite,
        logger=logger,
    )


def cmd_full_pipeline(args, config: PipelineConfig):
    print("=" * 60)
    print("Running data_pre full pipeline")
    print("=" * 60)

    steps = [
        ("Cropmasks", cmd_setup_cropmask),
        ("Coarse grid", cmd_setup_coarse_grid),
        # Note: setup_region9 requires GeoTIFF paths; skipped in full-pipeline.
        ("NOAA download", cmd_noaa_download),
        ("Build train", cmd_build_train),
        ("Build predict", cmd_build_predict),
        ("Add ha_irrigated", cmd_add_ha),
        ("Feature engineering", cmd_feature_engineering),
    ]
    for name, fn in steps:
        print(f"\n=== {name} ===")
        try:
            fn(args, config)
            print(f"OK: {name}")
        except Exception as e:
            print(f"FAIL: {name} — {e}")
            if not args.continue_on_error:
                sys.exit(1)
    print("\nfull-pipeline finished.")


def _add_common_year_args(p):
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)


def _add_common_crop_args(p):
    p.add_argument("--crops", nargs="*", default=None)


def main():
    parser = argparse.ArgumentParser(
        description="data_pre CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", default=None,
                        help="Path to config.yaml (default: ./config/config.yaml)")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    p = sub.add_parser("setup-cropmask", help="Build per-crop monthly masks")
    _add_common_year_args(p)
    _add_common_crop_args(p)

    p = sub.add_parser("setup-coarse-grid", help="Build 50 km coarse grid-id parquet")
    p.add_argument("--template", default=None, help="0.05-deg template NC")
    p.add_argument("--coarse-factor", type=int, default=6,
                   help="Cells per coarse cell per axis (default: 6 ~ 50 km)")

    p = sub.add_parser("setup-region9", help="Build 9-class region parquet")
    p.add_argument("--thz-tif", required=True, help="GAEZ thermal zone GeoTIFF")
    p.add_argument("--mst-tif", required=True, help="GAEZ moisture zone GeoTIFF")
    p.add_argument("--template", default=None, help="0.05-deg template NC")

    p = sub.add_parser("noaa-download", help="Download NOAA indices as CSV")
    p.add_argument("--keep-tim", action="store_true",
                   help="Keep intermediate .tim files")

    p = sub.add_parser("build-train", help="Build raw training parquets")
    _add_common_year_args(p)
    _add_common_crop_args(p)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("build-predict", help="Build raw prediction parquets")
    _add_common_year_args(p)
    _add_common_crop_args(p)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("add-ha", help="Add ha_irrigated feature")
    p.add_argument("--mode", choices=["train", "predict", "both"], default="both")
    _add_common_crop_args(p)

    p = sub.add_parser("feature-engineering",
                       help="Apply lag/rolling/time/loc features")
    p.add_argument("--mode", choices=["train", "predict", "both"], default="both")
    _add_common_crop_args(p)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("full-pipeline", help="Run all assembly + FE steps")
    _add_common_year_args(p)
    _add_common_crop_args(p)
    p.add_argument("--mode", choices=["train", "predict", "both"], default="both")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--keep-tim", action="store_true")
    p.add_argument("--template", default=None)
    p.add_argument("--coarse-factor", type=int, default=6)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    create_directories(config)

    commands = {
        "setup-cropmask": cmd_setup_cropmask,
        "setup-coarse-grid": cmd_setup_coarse_grid,
        "setup-region9": cmd_setup_region9,
        "noaa-download": cmd_noaa_download,
        "build-train": cmd_build_train,
        "build-predict": cmd_build_predict,
        "add-ha": cmd_add_ha,
        "feature-engineering": cmd_feature_engineering,
        "full-pipeline": cmd_full_pipeline,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
