#!/usr/bin/env python3
"""iHA (Irrigated Harvested Area) pipeline main CLI entry point."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from config import PipelineConfig, create_directories, load_config


def _setup_logger(log_dir: str, name: str = "iha", level: str = "INFO"):
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


def _run(cmd, logger):
    """Run a sub-pipeline script; raises on non-zero exit."""
    logger.info(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        logger.error(f"  FAILED (exit code {r.returncode})")
        sys.exit(r.returncode)


def cmd_download_data(args, config: PipelineConfig):
    logger = _setup_logger(config.log_dir, "download_data", config.log_level)
    logger.info("=" * 60)
    logger.info("Stage 0: download LUH2 + CROPGRIDS + FAOSTAT raw inputs")
    logger.info("=" * 60)
    cmd = [sys.executable, "src/download_data.py", "--all"]
    _run(cmd, logger)


def cmd_run_pipeline(args, config: PipelineConfig):
    logger = _setup_logger(config.log_dir, "run_pipeline", config.log_level)
    logger.info("=" * 60)
    logger.info("Stage 1: Method B  →  per-crop annual HA (0.05° grid)")
    logger.info("=" * 60)
    cmd = [
        sys.executable, "src/run_pipeline.py",
        "--luh2_dir",            config.data.luh2_dir,
        "--cropgrids_dir",       config.data.cropgrids_dir,
        "--faostat_change",      config.data.faostat_change,
        "--faostat_change_2024", config.data.faostat_change_2024,
        "--output_dir",          config.output.pipeline_dir,
    ]
    if args.crops:
        cmd += ["--crops"] + list(args.crops)
    _run(cmd, logger)


def cmd_aggregate_oac_pec_pdc(args, config: PipelineConfig):
    logger = _setup_logger(config.log_dir, "aggregate_oac_pec_pdc", config.log_level)
    logger.info("=" * 60)
    logger.info("Stage 2a: aggregate oac / pec / pdc  →  annual NCs (5-arcmin)")
    logger.info("=" * 60)
    cats = args.categories or config.aggregate.categories
    cmd = [
        sys.executable, "src/aggregate_oac_pec_pdc.py",
        "--pipeline_dir", config.output.pipeline_dir,
        "--output_dir",   config.aggregate.output_dir,
        "--categories",   *cats,
    ]
    _run(cmd, logger)


def cmd_downscale_monthly(args, config: PipelineConfig):
    logger = _setup_logger(config.log_dir, "downscale_monthly", config.log_level)
    logger.info("=" * 60)
    logger.info("Stage 2b: downscale 24 ACEA crops to monthly (5-arcmin)")
    logger.info("=" * 60)
    crops = args.crops or config.monthly.acea_crops
    cmd = [
        sys.executable, "src/monthly_downscale.py",
        "--pipeline_dir", config.output.pipeline_dir,
        "--calendar_dir", config.crop_calendar.dir,
        "--output_dir",   config.monthly.output_dir,
        "--crops",        *crops,
    ]
    _run(cmd, logger)


def cmd_validate_vs_spam(args, config: PipelineConfig):
    logger = _setup_logger(config.log_dir, "validate_vs_spam", config.log_level)
    logger.info("=" * 60)
    logger.info("Stage 3: validate per-crop 2020 iHA vs SPAM2020")
    logger.info("=" * 60)
    cmd = [
        sys.executable, "src/validate_vs_spam.py",
        "--pipeline_dir", config.output.pipeline_dir,
        "--spam_dir",     config.validation.spam_dir,
        "--output_dir",   config.validation.output_dir,
    ]
    _run(cmd, logger)


def cmd_full_pipeline(args, config: PipelineConfig):
    print("=" * 60)
    print("Running iHA full pipeline (stages 0 → 3)")
    print("=" * 60)
    cmd_download_data(args, config)
    cmd_run_pipeline(args, config)
    cmd_aggregate_oac_pec_pdc(args, config)
    cmd_downscale_monthly(args, config)
    cmd_validate_vs_spam(args, config)


def main():
    parser = argparse.ArgumentParser(
        description="iHA Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c", "--config", default=None,
        help="Path to config.yaml (default: ./config/config.yaml)"
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("download-data",
                   help="Download LUH2, CROPGRIDS, FAOSTAT raw inputs")

    p = sub.add_parser("run-pipeline",
                       help="Stage 1: Method B  →  per-crop annual HA")
    p.add_argument("--crops", nargs="*", default=None,
                   help="Subset of crops to process (default: all)")

    p = sub.add_parser("aggregate-oac-pec-pdc",
                       help="Stage 2a: aggregate oac/pec/pdc annual NCs")
    p.add_argument("--categories", nargs="*", default=None,
                   choices=["oac", "pec", "pdc"],
                   help="Categories (default: from config)")

    p = sub.add_parser("downscale-monthly",
                       help="Stage 2b: downscale 24 ACEA crops to monthly")
    p.add_argument("--crops", nargs="*", default=None,
                   help="ACEA crop codes (default: all 24 from config)")

    sub.add_parser("validate-vs-spam",
                   help="Stage 3: validate per-crop 2020 iHA vs SPAM2020")

    p = sub.add_parser("full-pipeline",
                       help="Run stages 0 → 3 in order")
    p.add_argument("--crops", nargs="*", default=None)
    p.add_argument("--categories", nargs="*", default=None,
                   choices=["oac", "pec", "pdc"])

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    create_directories(config)

    commands = {
        "download-data":          cmd_download_data,
        "run-pipeline":           cmd_run_pipeline,
        "aggregate-oac-pec-pdc":  cmd_aggregate_oac_pec_pdc,
        "downscale-monthly":      cmd_downscale_monthly,
        "validate-vs-spam":       cmd_validate_vs_spam,
        "full-pipeline":          cmd_full_pipeline,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
