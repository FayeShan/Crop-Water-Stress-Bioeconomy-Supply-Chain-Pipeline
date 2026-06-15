"""Water Data Pipeline - main CLI entry point (epilog text used as argparse help).

Usage:
    python main.py <command> [options]

Commands:
    era5-download             Download ERA5 monthly-means from CDS
    era5-unzip                Unzip downloaded ERA5 files
    era5-fix                  Apply ECMWF workaround for 2022-2024 bug
    modis-download            Download MOD13C2 (VI) via Earthaccess
    modis-hdftonc             Convert MOD13C2 HDF to NetCDF
    modis-green               Download+process MOD09CMG Green reflectance
    modis-landcover           Download+process MCD12Q2 phenology
    modis-landuse             Download+process MCD12C1 IGBP
    smia-tifftonc             Convert SMIA TIFF to NetCDF
    twsa-resample             Resample TWSA (with missing-month fill)
    resample-era5             Resample ERA5 to reference grid
    resample-modis            Resample MOD13C2 VI
    resample-modis-green      Resample MOD09CMG Green
    resample-modis-landcover  Resample MCD12Q2 phenology
    resample-modis-landuse    Resample MCD12C1 IGBP
    resample-smia             Resample SMIA to reference grid
    merge                     Merge all enabled data sources
    full-pipeline             Run the complete pipeline end-to-end
"""

import argparse
import sys
import os

from config import load_config, create_directories, PipelineConfig


def cmd_era5_download(args, config: PipelineConfig):
    """Download ERA5 data."""
    from src.downloaders.era5 import download_era5
    
    start = args.start_year or config.train_start
    end = args.end_year or config.predict_end
    
    download_era5(
        output_dir=config.era5.output_dir,
        variables=config.era5.variables,
        months=config.era5.months,
        start_year=start,
        end_year=end,
        log_dir=config.log_dir,
        force=args.force
    )


def cmd_era5_process(args, config: PipelineConfig):
    """Process ERA5 data."""
    from src.downloaders.era5 import process_era5
    
    start = args.start_year or config.train_start
    end = args.end_year or config.predict_end
    
    process_era5(
        input_dir=config.era5.output_dir,
        output_dir=config.era5.processed_dir,
        start_year=start,
        end_year=end,
        log_dir=config.log_dir,
        delete_zip=args.delete_zip
    )


def cmd_modis_download(args, config: PipelineConfig):
    """Download MODIS data."""
    from src.downloaders.modis import download_modis
    
    start = max(args.start_year or config.modis.start_year, 2000)
    end = args.end_year or config.predict_end
    
    print("Note: Make sure you have logged in to NASA Earthaccess.")
    print("Run 'earthaccess.login()' interactively if not yet authenticated.")
    
    download_modis(
        output_dir=config.modis.raw_dir,
        start_year=start,
        end_year=end,
        short_name=config.modis.short_name,
        version=config.modis.version,
        log_dir=config.log_dir
    )


def cmd_modis_process(args, config: PipelineConfig):
    """Process MODIS HDF to NetCDF."""
    from src.downloaders.modis import process_modis
    
    start = max(args.start_year or config.modis.start_year, 2000)
    end = args.end_year or config.predict_end

    selected_vars = config.modis.selected_vars if config.modis.selected_vars else None
    
    process_modis(
        input_dir=config.modis.raw_dir,
        output_dir=config.modis.nc_dir,
        start_year=start,
        end_year=end,
        var_mapping=config.modis.var_mapping or None,
        selected_vars=selected_vars,
        fill_value=config.modis.fill_value,
        log_dir=config.log_dir
    )


def cmd_smia_process(args, config: PipelineConfig):
    """Process SMIA data."""
    from src.processors.smia import process_smia
    
    start = args.start_year or config.train_start
    end = args.end_year or config.predict_end
    
    process_smia(
        input_dir=config.smia.input_dir,
        output_dir=config.smia.output_dir,
        days=config.smia.days,
        start_year=start,
        end_year=end,
        log_dir=config.log_dir
    )


def cmd_twsa_process(args, config: PipelineConfig):
    """Process TWSA data."""
    from src.processors.twsa import process_twsa
    
    start = max(args.start_year or config.twsa.start_year, 2002)
    end = args.end_year or config.predict_end

    fill_missing = not getattr(args, 'no_fill', False)
    
    process_twsa(
        input_dir=config.twsa.input_dir,
        output_dir=config.twsa.output_dir,
        start_year=start,
        end_year=end,
        reference_grid_path=config.resampling.reference_grid,
        fill_missing_months=fill_missing,
        log_dir=config.log_dir
    )


def cmd_era5_fix(args, config: PipelineConfig):
    """Fix ERA5 accumulated variables using the by-hour-of-day product.

    Applies the ECMWF-recommended workaround for the known bug affecting
    ERA5-Land monthly averaged accumulated fields (ssr, ssrd, tp, ro) for
    September 2022 – February 2024. Rewrites the existing processed monthly
    NC files in place. No-op for years outside 2022-2024.
    """
    from src.downloaders.era5_fix import fix_era5_accumulated

    start = args.start_year or config.predict_start
    end = args.end_year or config.predict_end

    fix_era5_accumulated(
        processed_dir=config.era5.processed_dir,
        start_year=start,
        end_year=end,
        log_dir=config.log_dir,
        force=getattr(args, "force", False),
    )


def cmd_modis_green(args, config: PipelineConfig):
    """Download and process MOD09CMG Green reflectance (monthly mean)."""
    from src.downloaders.modis_green import process_modis_green

    if config.modis_green is None:
        print("ERROR: modis_green section not found in config.yaml")
        sys.exit(1)

    start = max(args.start_year or config.modis_green.start_year,
                config.modis_green.start_year)
    end = args.end_year or config.predict_end

    print("Note: Make sure you have logged in to NASA Earthaccess.")
    print("Run 'earthaccess.login()' interactively if not yet authenticated.")

    process_modis_green(
        raw_dir=config.modis_green.raw_dir,
        nc_dir=config.modis_green.nc_dir,
        start_year=start,
        end_year=end,
        keep_hdf=config.modis_green.keep_hdf,
        log_dir=config.log_dir,
    )


def cmd_modis_landcover(args, config: PipelineConfig):
    """Download and process MCD12Q2 Land Cover Dynamics (phenology)."""
    from src.downloaders.modis_landcover import process_modis_landcover

    if config.modis_landcover is None:
        print("ERROR: modis_landcover section not found in config.yaml")
        sys.exit(1)

    start = max(args.start_year or config.modis_landcover.start_year,
                config.modis_landcover.start_year)
    end = args.end_year or config.predict_end

    print("Note: Make sure you have logged in to NASA Earthaccess.")

    process_modis_landcover(
        raw_dir=config.modis_landcover.raw_dir,
        nc_dir=config.modis_landcover.nc_dir,
        start_year=start,
        end_year=end,
        keep_hdf=config.modis_landcover.keep_hdf,
        log_dir=config.log_dir,
    )


def cmd_modis_landuse(args, config: PipelineConfig):
    """Download and process MCD12C1 IGBP Land Cover Type."""
    from src.downloaders.modis_landuse import process_modis_landuse

    if config.modis_landuse is None:
        print("ERROR: modis_landuse section not found in config.yaml")
        sys.exit(1)

    start = max(args.start_year or config.modis_landuse.start_year,
                config.modis_landuse.start_year)
    end = args.end_year or config.predict_end

    print("Note: Make sure you have logged in to NASA Earthaccess.")

    process_modis_landuse(
        raw_dir=config.modis_landuse.raw_dir,
        nc_dir=config.modis_landuse.nc_dir,
        start_year=start,
        end_year=end,
        keep_hdf=config.modis_landuse.keep_hdf,
        log_dir=config.log_dir,
    )


def cmd_resample_modis_green(args, config: PipelineConfig):
    """Resample MOD09CMG Green reflectance."""
    from src.processors.resampling import resample_modis_green

    if config.modis_green is None:
        print("ERROR: modis_green section not found in config.yaml")
        sys.exit(1)

    start = args.start_year or config.modis_green.start_year
    end = args.end_year or config.predict_end

    resample_modis_green(
        input_dir=config.modis_green.nc_dir,
        output_dir=config.modis_green.resampled_dir,
        reference_grid_path=config.resampling.reference_grid,
        start_year=start,
        end_year=end,
        method=config.resampling.method,
        log_dir=config.log_dir,
    )


def cmd_resample_modis_landcover(args, config: PipelineConfig):
    """Resample MCD12Q2 phenology (nearest-neighbour)."""
    from src.processors.resampling import resample_modis_landcover

    if config.modis_landcover is None:
        print("ERROR: modis_landcover section not found in config.yaml")
        sys.exit(1)

    start = args.start_year or config.modis_landcover.start_year
    end = args.end_year or config.predict_end

    resample_modis_landcover(
        input_dir=config.modis_landcover.nc_dir,
        output_dir=config.modis_landcover.resampled_dir,
        reference_grid_path=config.resampling.reference_grid,
        start_year=start,
        end_year=end,
        log_dir=config.log_dir,
    )


def cmd_resample_modis_landuse(args, config: PipelineConfig):
    """Resample MCD12C1 IGBP (nearest-neighbour)."""
    from src.processors.resampling import resample_modis_landuse

    if config.modis_landuse is None:
        print("ERROR: modis_landuse section not found in config.yaml")
        sys.exit(1)

    start = args.start_year or config.modis_landuse.start_year
    end = args.end_year or config.predict_end

    resample_modis_landuse(
        input_dir=config.modis_landuse.nc_dir,
        output_dir=config.modis_landuse.resampled_dir,
        reference_grid_path=config.resampling.reference_grid,
        start_year=start,
        end_year=end,
        log_dir=config.log_dir,
    )


def cmd_resample_era5(args, config: PipelineConfig):
    """Resample ERA5 data."""
    from src.processors.resampling import resample_era5
    
    start = args.start_year or config.train_start
    end = args.end_year or config.predict_end
    
    resample_era5(
        input_dir=config.era5.processed_dir,
        output_dir=config.era5.resampled_dir,
        reference_grid_path=config.resampling.reference_grid,
        start_year=start,
        end_year=end,
        method=config.resampling.method,
        log_dir=config.log_dir
    )


def cmd_resample_modis(args, config: PipelineConfig):
    """Resample MODIS data."""
    from src.processors.resampling import resample_modis
    
    start = max(args.start_year or config.modis.start_year, 2000)
    end = args.end_year or config.predict_end
    
    resample_modis(
        input_dir=config.modis.nc_dir,
        output_dir=config.modis.resampled_dir,
        reference_grid_path=config.resampling.reference_grid,
        start_year=start,
        end_year=end,
        method=config.resampling.method,
        log_dir=config.log_dir
    )


def cmd_resample_smia(args, config: PipelineConfig):
    """Resample SMIA data."""
    from src.processors.resampling import resample_smia
    
    start = args.start_year or config.train_start
    end = args.end_year or config.predict_end
    
    resample_smia(
        input_dir=config.smia.output_dir,
        output_dir=config.smia.resampled_dir,
        reference_grid_path=config.resampling.reference_grid,
        start_year=start,
        end_year=end,
        method=config.resampling.method,
        log_dir=config.log_dir
    )


def cmd_merge(args, config: PipelineConfig):
    """Merge all enabled data sources."""
    from src.processors.merger import merge_data

    start = args.start_year or config.train_start
    end = args.end_year or config.predict_end

    merge_data(
        era5_dir=config.era5.resampled_dir,
        modis_dir=config.modis.resampled_dir,
        smia_dir=config.smia.resampled_dir,
        twsa_dir=config.twsa.output_dir,
        modis_green_dir=config.modis_green.resampled_dir if config.modis_green else None,
        modis_landcover_dir=config.modis_landcover.resampled_dir if config.modis_landcover else None,
        modis_landuse_dir=config.modis_landuse.resampled_dir if config.modis_landuse else None,
        output_dir=config.merge.output_dir,
        start_year=start,
        end_year=end,
        sources_enabled=config.merge.sources,
        extra_files=config.merge.extra_files if config.merge.extra_files else None,
        compression_level=config.merge.compression_level,
        require_all=args.require_all,
        log_dir=config.log_dir,
    )


def cmd_full_pipeline(args, config: PipelineConfig):
    """Run the complete pipeline."""
    print("=" * 60)
    print("Running Full Pipeline")
    print("=" * 60)
    
    enabled = config.merge.sources or {}

    steps = [
        ("ERA5 Download", cmd_era5_download),
        ("ERA5 Unzip", cmd_era5_process),
        ("ERA5 Fix (2022-2024)", cmd_era5_fix),
        ("MODIS Download", cmd_modis_download),
        ("MODIS HDF to NC", cmd_modis_process),
        ("SMIA TIFF to NC", cmd_smia_process),
        ("TWSA Resample", cmd_twsa_process),
    ]
    if enabled.get("modis_green"):
        steps.append(("MODIS Green (download+process)", cmd_modis_green))
    if enabled.get("modis_landcover"):
        steps.append(("MODIS Landcover (download+process)", cmd_modis_landcover))
    if enabled.get("modis_landuse"):
        steps.append(("MODIS Landuse (download+process)", cmd_modis_landuse))
    steps += [
        ("Resample ERA5", cmd_resample_era5),
        ("Resample MODIS", cmd_resample_modis),
        ("Resample SMIA", cmd_resample_smia),
    ]
    if enabled.get("modis_green"):
        steps.append(("Resample MODIS Green", cmd_resample_modis_green))
    if enabled.get("modis_landcover"):
        steps.append(("Resample MODIS Landcover", cmd_resample_modis_landcover))
    if enabled.get("modis_landuse"):
        steps.append(("Resample MODIS Landuse", cmd_resample_modis_landuse))
    steps.append(("Merge Data", cmd_merge))
    
    for name, func in steps:
        print(f"\n{'='*60}")
        print(f"Step: {name}")
        print("=" * 60)
        
        try:
            func(args, config)
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            if not args.continue_on_error:
                sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Water Data Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "-c", "--config",
        help="Path to config file",
        default=None
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    def add_year_args(p):
        p.add_argument("--start-year", type=int, help="Start year")
        p.add_argument("--end-year", type=int, help="End year")

    p = subparsers.add_parser("era5-download", help="Download ERA5 data")
    add_year_args(p)
    p.add_argument("--force", action="store_true", help="Force re-download")

    p = subparsers.add_parser("era5-unzip", help="Unzip ERA5 downloaded files")
    add_year_args(p)
    p.add_argument("--delete-zip", action="store_true", help="Delete zip after extraction")

    p = subparsers.add_parser("modis-download", help="Download MODIS data")
    add_year_args(p)

    p = subparsers.add_parser("modis-hdftonc", help="Convert MODIS HDF to NetCDF")
    add_year_args(p)

    p = subparsers.add_parser("smia-tifftonc", help="Convert SMIA TIFF to NetCDF")
    add_year_args(p)

    p = subparsers.add_parser("twsa-resample", help="Resample TWSA data")
    add_year_args(p)
    p.add_argument("--no-fill", action="store_true", 
                   help="Disable automatic filling of missing months")
    
    # MODIS Green (MOD09CMG Band 4 monthly mean)
    p = subparsers.add_parser(
        "modis-green",
        help="Download and process MOD09CMG Green reflectance (Band 4, monthly mean)"
    )
    add_year_args(p)

    # MODIS Landcover (MCD12Q2 phenology)
    p = subparsers.add_parser(
        "modis-landcover",
        help="Download and process MCD12Q2 Land Cover Dynamics (phenology)"
    )
    add_year_args(p)

    # MODIS Landuse (MCD12C1 IGBP)
    p = subparsers.add_parser(
        "modis-landuse",
        help="Download and process MCD12C1 IGBP Land Cover Type"
    )
    add_year_args(p)

    # Resample MODIS additional products
    p = subparsers.add_parser(
        "resample-modis-green",
        help="Resample MOD09CMG Green reflectance to reference grid"
    )
    add_year_args(p)

    p = subparsers.add_parser(
        "resample-modis-landcover",
        help="Resample MCD12Q2 phenology to reference grid (nearest-neighbour)"
    )
    add_year_args(p)

    p = subparsers.add_parser(
        "resample-modis-landuse",
        help="Resample MCD12C1 IGBP to reference grid (nearest-neighbour)"
    )
    add_year_args(p)

    # ERA5 Fix (workaround for Sep 2022 - Feb 2024 bug in monthly_averaged_reanalysis)
    p = subparsers.add_parser(
        "era5-fix",
        help="Correct the 4 accumulated flux variables (ssr, ssrd, tp, ro) for "
             "Sep 2022 - Feb 2024 using the ECMWF-recommended workaround "
             "(monthly_averaged_reanalysis_by_hour_of_day); rewrites processed/ in place"
    )
    add_year_args(p)
    p.add_argument("--force", action="store_true", help="Force re-download")

    p = subparsers.add_parser("resample-era5", help="Resample ERA5 data")
    add_year_args(p)

    p = subparsers.add_parser("resample-modis", help="Resample MODIS data")
    add_year_args(p)

    p = subparsers.add_parser("resample-smia", help="Resample SMIA data")
    add_year_args(p)

    p = subparsers.add_parser("merge", help="Merge all data sources")
    add_year_args(p)
    p.add_argument("--require-all", action="store_true",
                   help="Skip years with missing data")

    p = subparsers.add_parser("full-pipeline", help="Run complete pipeline")
    add_year_args(p)
    p.add_argument("--force", action="store_true", help="Force re-download")
    p.add_argument("--delete-zip", action="store_true", help="Delete zip files")
    p.add_argument("--require-all", action="store_true", 
                   help="Require all data sources for merge")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continue pipeline on step failure")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    create_directories(config)

    commands = {
        "era5-download": cmd_era5_download,
        "era5-unzip": cmd_era5_process,
        "modis-download": cmd_modis_download,
        "modis-hdftonc": cmd_modis_process,
        "smia-tifftonc": cmd_smia_process,
        "twsa-resample": cmd_twsa_process,
        "era5-fix": cmd_era5_fix,
        "modis-green": cmd_modis_green,
        "modis-landcover": cmd_modis_landcover,
        "modis-landuse": cmd_modis_landuse,
        "resample-modis-green": cmd_resample_modis_green,
        "resample-modis-landcover": cmd_resample_modis_landcover,
        "resample-modis-landuse": cmd_resample_modis_landuse,
        "resample-era5": cmd_resample_era5,
        "resample-modis": cmd_resample_modis,
        "resample-smia": cmd_resample_smia,
        "merge": cmd_merge,
        "full-pipeline": cmd_full_pipeline,
    }
    
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
