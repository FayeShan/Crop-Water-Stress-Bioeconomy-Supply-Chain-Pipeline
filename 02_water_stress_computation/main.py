"""Water stress computation CLI entry point."""

import argparse
import logging
import sys
from pathlib import Path

from config import PipelineConfig, create_directories, load_config


def _setup_logger(log_dir: str, name: str = "water_stress", level: str = "INFO"):
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


def cmd_compute_ws(args, config: PipelineConfig):
    """Weight monthly WF by AWARE and write per-year NC files."""
    from src.compute_water_stress import compute_water_stress

    logger = _setup_logger(config.log_dir, "compute_ws", config.log_level)

    start = args.start_year if args.start_year is not None else config.date_range.start_year
    end = args.end_year if args.end_year is not None else config.date_range.end_year

    compute_water_stress(
        wf_dir=config.wf.input_dir,
        weights_nc=config.aware.weights_nc,
        output_dir=config.output.waterstress_dir,
        filename_regex=config.wf.filename_regex,
        wf_variable=config.wf.variable,
        weights_variable=config.aware.weights_variable,
        crops=args.crops,
        start_year=start,
        end_year=end,
        compression_level=config.output.compression_level,
        logger=logger,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Water Stress Computation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c", "--config", default=None,
        help="Path to config.yaml (default: ./config/config.yaml)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p = subparsers.add_parser(
        "compute-ws",
        help="Compute water stress: wfp_blue_ir * aware_monthly_weight"
    )
    p.add_argument("--crops", nargs="*", default=None,
                   help="Crop codes to process (default: all found in wf/)")
    p.add_argument("--start-year", type=int, default=None,
                   help="First year (overrides config)")
    p.add_argument("--end-year", type=int, default=None,
                   help="Last year (overrides config)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    create_directories(config)

    commands = {
        "compute-ws": cmd_compute_ws,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
