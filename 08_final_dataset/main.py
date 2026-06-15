#!/usr/bin/env python3
"""final_dataset CLI: package HARM predictions into the published NetCDF / GeoTIFF / country-CSV dataset (Zenodo + GEE dashboard)."""

import argparse
import logging
import sys
from pathlib import Path

from config import PipelineConfig, create_directories, load_config


def _setup_logger(log_dir, name, level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log", mode="w")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def _years(args, config):
    return (args.start_year or config.years.start, args.end_year or config.years.end)


def cmd_build_netcdf(args, config: PipelineConfig):
    from src.build_netcdf import build_netcdf
    logger = _setup_logger(config.log_dir, "build_netcdf", config.log_level)
    y0, y1 = _years(args, config)
    build_netcdf(
        pred_dir=config.input.predictions_dir,
        reference_grid_nc=config.input.reference_grid_nc,
        monthly_dir=config.output.netcdf_monthly_dir,
        yearly_dir=config.output.netcdf_yearly_dir,
        year_start=y0, year_end=y1, logger=logger,
    )


def cmd_build_geotiff(args, config: PipelineConfig):
    from src.build_geotiff import build_geotiff
    logger = _setup_logger(config.log_dir, "build_geotiff", config.log_level)
    y0, y1 = _years(args, config)
    build_geotiff(
        nc_dir=config.output.netcdf_monthly_dir,
        monthly_out=config.output.geotiff_monthly_dir,
        yearly_out=config.output.geotiff_yearly_dir,
        year_start=y0, year_end=y1, logger=logger,
    )


def cmd_build_country_csv(args, config: PipelineConfig):
    from src.build_country_csv import build_country_csv
    logger = _setup_logger(config.log_dir, "build_country_csv", config.log_level)
    build_country_csv(
        nc_dir=config.output.netcdf_monthly_dir,
        csv_dir=config.output.csv_dir,
        shapefile=config.input.shapefile,
        logger=logger,
    )


def cmd_recompress(args, config: PipelineConfig):
    from src.recompress_netcdf import recompress_netcdf
    logger = _setup_logger(config.log_dir, "recompress_netcdf", config.log_level)
    recompress_netcdf(
        monthly_dir=config.output.netcdf_monthly_dir,
        yearly_dir=config.output.netcdf_yearly_dir,
        complevel=config.netcdf.compression_level,
        skip_threshold_mb=config.netcdf.recompress_skip_threshold_mb,
        logger=logger,
    )


def cmd_full_pipeline(args, config: PipelineConfig):
    print("=" * 70)
    print("final_dataset full pipeline")
    print("=" * 70)
    steps = [
        ("1. build-netcdf", cmd_build_netcdf),
        ("2. build-geotiff", cmd_build_geotiff),
        ("3. build-country-csv", cmd_build_country_csv),
    ]
    if args.recompress:
        steps.append(("4. recompress", cmd_recompress))
    for name, fn in steps:
        print(f"\n=== {name} ===")
        try:
            fn(args, config)
            print(f"OK: {name}")
        except Exception as e:
            print(f"FAIL: {name} -- {e}")
            if not args.continue_on_error:
                sys.exit(1)


def _add_year_args(p):
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)


def main():
    parser = argparse.ArgumentParser(
        description="final_dataset CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", default=None,
                        help="Path to config.yaml (default: ./config/config.yaml)")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("build-netcdf", help="Predictions {crop}.csv -> monthly + yearly NetCDF")
    _add_year_args(p)
    p = sub.add_parser("build-geotiff", help="Monthly NetCDF -> GeoTIFF (monthly + yearly, 27 bands)")
    _add_year_args(p)
    sub.add_parser("build-country-csv", help="NetCDF -> global + per-crop per-country CSVs")
    sub.add_parser("recompress", help="Recompress dataset NetCDF in place (zlib)")

    p = sub.add_parser("full-pipeline", help="Run build-netcdf -> build-geotiff -> build-country-csv")
    _add_year_args(p)
    p.add_argument("--recompress", action="store_true", help="Also recompress NetCDF at the end")
    p.add_argument("--continue-on-error", action="store_true")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    create_directories(config)

    commands = {
        "build-netcdf": cmd_build_netcdf,
        "build-geotiff": cmd_build_geotiff,
        "build-country-csv": cmd_build_country_csv,
        "recompress": cmd_recompress,
        "full-pipeline": cmd_full_pipeline,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
