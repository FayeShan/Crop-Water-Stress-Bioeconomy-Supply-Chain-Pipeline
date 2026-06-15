#!/usr/bin/env python3
"""gloria_mapping CLI: map HARM water-stress data into GLORIA's 14 crop sectors and build the hybrid satellite account for the GLORIA MRIO model."""

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
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def cmd_agg_to_country(args, config: PipelineConfig):
    from src.agg_to_country import run_agg_to_country
    logger = _setup_logger(config.log_dir, "agg_to_country", config.log_level)
    crops = args.crops or config.crops.all_27
    run_agg_to_country(
        crops=crops,
        pred_dir=config.harm_predictions.input_dir,
        shapefile=config.harm_predictions.shapefile,
        out_dir=config.harm_predictions.country_csv_dir,
        y_pred_column=config.harm_predictions.y_pred_column,
        logger=logger,
    )


def cmd_merge_hist_pred(args, config: PipelineConfig):
    from src.merge_hist_pred import merge_historical_and_predictions
    logger = _setup_logger(config.log_dir, "merge_hist_pred", config.log_level)
    crops = args.crops or config.crops.all_27
    merge_historical_and_predictions(
        historical_csv=config.historical.crops_27_csv,
        pred_country_dir=config.harm_predictions.country_csv_dir,
        crops=crops,
        output_csv=config.combined.output_csv,
        overlap_rule=args.overlap_rule,
        logger=logger,
    )


def cmd_sector_mapping(args, config: PipelineConfig):
    from src.sector_mapping import generate_sector_mapping
    logger = _setup_logger(config.log_dir, "sector_mapping", config.log_level)
    generate_sector_mapping(
        output_csv=config.reference.crop_mapping_175_to_27_to_14,
        logger=logger,
    )


def cmd_weights_computation(args, config: PipelineConfig):
    from src.weights_computation import compute_weights
    logger = _setup_logger(config.log_dir, "weights_computation", config.log_level)
    compute_weights(
        crops_175_csv=config.historical.crops_175_csv,
        crop_mapping_csv=config.reference.crop_mapping_175_to_27_to_14,
        country_mapping_csv=config.reference.country_mapping_175_to_14,
        rolling_csv=config.weights.rolling_csv,
        recent_csv=config.weights.recent_csv,
        years_175=tuple(config.historical.years),
        rolling_window=config.weights.rolling_window,
        recent_period=tuple(config.weights.recent_period),
        split_crops=config.weights.split_crops,
        logger=logger,
    )


def cmd_mapping_27_to_14(args, config: PipelineConfig):
    from src.mapping_27_to_14 import apply_mapping_27_to_14
    logger = _setup_logger(config.log_dir, "mapping_27_to_14", config.log_level)
    apply_mapping_27_to_14(
        crops_27_csv=config.combined.output_csv,
        country_mapping_csv=config.reference.country_mapping_27_to_14,
        rolling_csv=config.weights.rolling_csv,
        recent_csv=config.weights.recent_csv,
        gloria_readme=config.reference.gloria_readme,
        output_csv=config.mapping.long_csv,
        n_regions=config.mapping.n_regions,
        split_crops=config.weights.split_crops,
        logger=logger,
    )


def cmd_satellite_preparation(args, config: PipelineConfig):
    from src.satellite_preparation import build_our_satellite_arrays
    logger = _setup_logger(config.log_dir, "satellite_preparation", config.log_level)
    build_our_satellite_arrays(
        long_csv=config.mapping.long_csv,
        output_dir=config.satellite.our_dir,
        n_regions=config.mapping.n_regions,
        n_sectors_gloria=config.satellite.n_sectors_gloria,
        logger=logger,
    )


def cmd_gloria_unzip_satellite(args, config: PipelineConfig):
    from src.gloria_unzip_satellite import unzip_satellite_zips
    logger = _setup_logger(config.log_dir, "gloria_unzip_satellite", config.log_level)
    unzip_satellite_zips(
        zip_dir=config.gloria_raw.sat_zips_dir,
        output_dir=config.gloria_processed.satellite_dir,
        start_year=args.start_year or config.date_range.gloria_start,
        end_year=args.end_year or config.date_range.gloria_end,
        logger=logger,
    )


def cmd_gloria_unzip_mrio(args, config: PipelineConfig):
    from src.gloria_unzip_mrio import unzip_mrio_zips
    logger = _setup_logger(config.log_dir, "gloria_unzip_mrio", config.log_level)
    unzip_mrio_zips(
        zip_dir=config.gloria_raw.mrio_zips_dir,
        output_dir=config.gloria_processed.mrio_unzipped_dir,
        start_year=args.start_year or config.date_range.gloria_start,
        end_year=args.end_year or config.date_range.gloria_end,
        logger=logger,
    )


def cmd_gloria_sut_to_mrio(args, config: PipelineConfig):
    from src.gloria_sut_to_mrio import convert_sut_to_mrio
    logger = _setup_logger(config.log_dir, "gloria_sut_to_mrio", config.log_level)
    convert_sut_to_mrio(
        unzipped_dir=config.gloria_processed.mrio_unzipped_dir,
        zarr_dir=config.gloria_processed.mrio_zarr_dir,
        start_year=args.start_year or config.date_range.gloria_start,
        end_year=args.end_year or config.date_range.gloria_end,
        logger=logger,
    )


def cmd_build_hybrid_satellite(args, config: PipelineConfig):
    from src.build_hybrid_satellite import build_hybrid_satellite
    logger = _setup_logger(config.log_dir, "build_hybrid_satellite", config.log_level)
    build_hybrid_satellite(
        our_satellite_dir=config.satellite.our_dir,
        gloria_satellite_dir=config.gloria_processed.satellite_dir,
        output_dir=config.satellite.hybrid_dir,
        start_year=args.start_year or config.date_range.historical_start,
        end_year=args.end_year or config.date_range.predict_end,
        ag_row_index=config.satellite.ag_row_index,
        non_ag_row_index=config.satellite.non_ag_row_index,
        include_non_ag=config.satellite.include_non_ag,
        n_regions=config.mapping.n_regions,
        n_sectors=config.satellite.n_sectors_gloria,
        unit_scale=config.satellite.unit_scale,
        logger=logger,
    )


def cmd_validate_vs_gloria(args, config: PipelineConfig):
    from src.validate_vs_gloria import validate_vs_gloria
    logger = _setup_logger(config.log_dir, "validate_vs_gloria", config.log_level)
    validate_vs_gloria(
        our_satellite_dir=config.satellite.our_dir,
        gloria_satellite_dir=config.gloria_processed.satellite_dir,
        output_metrics_csv=config.validation.metrics_csv,
        output_composition_csv=config.validation.composition_csv,
        start_year=args.start_year or config.validation.start_year,
        end_year=args.end_year or config.validation.end_year,
        ag_row_index=config.satellite.ag_row_index,
        n_regions=config.mapping.n_regions,
        n_sectors=config.satellite.n_sectors_gloria,
        unit_scale=config.satellite.unit_scale,
        logger=logger,
    )


def cmd_full_pipeline(args, config: PipelineConfig):
    """Run the core CWS -> hybrid satellite chain (6 steps).

    GLORIA MRIO zarr conversion is NOT included by default because it
    depends on the 40+ GB GLORIA Part-I zips. Run it separately via
    ``gloria-unzip-mrio`` + ``gloria-sut-to-mrio`` if needed.
    """
    print("=" * 70)
    print("gloria_mapping full pipeline")
    print("=" * 70)
    steps = [
        ("1. agg-to-country",       cmd_agg_to_country),
        ("2. merge-hist-pred",      cmd_merge_hist_pred),
        ("3. weights-computation",  cmd_weights_computation),
        ("4. mapping-27-to-14",     cmd_mapping_27_to_14),
        ("5. satellite-preparation", cmd_satellite_preparation),
        ("6. gloria-unzip-satellite", cmd_gloria_unzip_satellite),
        ("7. build-hybrid-satellite", cmd_build_hybrid_satellite),
    ]
    for name, fn in steps:
        print(f"\n=== {name} ===")
        try:
            fn(args, config)
            print(f"OK: {name}")
        except Exception as e:
            print(f"FAIL: {name} -- {e}")
            if not args.continue_on_error:
                sys.exit(1)


def _add_common_year_args(p):
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)


def _add_crops_arg(p):
    p.add_argument("--crops", nargs="*", default=None,
                   help="Subset of 27 crop codes (default: all)")


def main():
    parser = argparse.ArgumentParser(
        description="gloria_mapping CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", default=None,
                        help="Path to config.yaml (default: ./config/config.yaml)")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("agg-to-country",
                       help="Aggregate grid-level HARM predictions to country level")
    _add_crops_arg(p)

    p = sub.add_parser("merge-hist-pred",
                       help="Merge historical + prediction into crops_27_1990_2024.csv")
    _add_crops_arg(p)
    p.add_argument("--overlap-rule",
                   choices=["keep_historical", "keep_prediction"],
                   default="keep_historical")

    p = sub.add_parser("sector-mapping",
                       help="(Re)generate crop_mapping_175_to_27_to_14.csv")

    p = sub.add_parser("weights-computation",
                       help="Compute split weights for oac / pdc / pec")

    p = sub.add_parser("mapping-27-to-14",
                       help="Produce water_stress_164x14_long.csv")

    p = sub.add_parser("satellite-preparation",
                       help="Build our 14-sector-block .npy per year")

    p = sub.add_parser("gloria-unzip-satellite",
                       help="Unzip GLORIA satellite-account zips")
    _add_common_year_args(p)

    p = sub.add_parser("gloria-unzip-mrio",
                       help="Unzip GLORIA MRIO SUT zips (optional)")
    _add_common_year_args(p)

    p = sub.add_parser("gloria-sut-to-mrio",
                       help="Convert GLORIA SUT CSVs -> MRIO zarr (optional)")
    _add_common_year_args(p)

    p = sub.add_parser("build-hybrid-satellite",
                       help="Merge our 14 sectors + GLORIA 15-120 -> hybrid .npy")
    _add_common_year_args(p)

    p = sub.add_parser("validate-vs-gloria",
                       help="Compare our 14-sector satellite to GLORIA agriculture row 390")
    _add_common_year_args(p)

    p = sub.add_parser("full-pipeline",
                       help="Run the 7-step CWS->hybrid satellite chain")
    _add_common_year_args(p)
    _add_crops_arg(p)
    p.add_argument("--overlap-rule",
                   choices=["keep_historical", "keep_prediction"],
                   default="keep_historical")
    p.add_argument("--continue-on-error", action="store_true")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    create_directories(config)

    commands = {
        "agg-to-country": cmd_agg_to_country,
        "merge-hist-pred": cmd_merge_hist_pred,
        "sector-mapping": cmd_sector_mapping,
        "weights-computation": cmd_weights_computation,
        "mapping-27-to-14": cmd_mapping_27_to_14,
        "satellite-preparation": cmd_satellite_preparation,
        "gloria-unzip-satellite": cmd_gloria_unzip_satellite,
        "gloria-unzip-mrio": cmd_gloria_unzip_mrio,
        "gloria-sut-to-mrio": cmd_gloria_sut_to_mrio,
        "build-hybrid-satellite": cmd_build_hybrid_satellite,
        "validate-vs-gloria": cmd_validate_vs_gloria,
        "full-pipeline": cmd_full_pipeline,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
