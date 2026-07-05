#!/usr/bin/env python3
"""Aggregate ~103 remaining crops into oac/pec/pdc annual NC files at 5 arcmin.

Memory strategy: process one year at a time, flushing each to a temporary NC
file, then concat all year-files at the end to keep peak RAM at ~2 GB.
"""

import argparse
import gc
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Target grid: 5 arcmin
TARGET_NLAT = 2160
TARGET_NLON = 4320
TARGET_LAT = np.linspace(90 - 5 / 120, -90 + 5 / 120, TARGET_NLAT)
TARGET_LON = np.linspace(-180 + 5 / 120, 180 - 5 / 120, TARGET_NLON)

OAC_CROPS = [
    'bambara', 'broadbean', 'buckwheat', 'canaryseed',
    'castor', 'cerealnes', 'chicory', 'fibrenes', 'flax',
    'fonio', 'ginger', 'hempseed', 'jute', 'lentil',
    'linseed', 'lupin', 'melonseed', 'mixedgrain', 'mustard',
    'oats', 'oilseednes', 'pea', 'peppermint', 'pigeonpea',
    'poppy', 'pulsenes', 'pyrethrum', 'quinoa', 'ramie',
    'rootnes', 'safflower', 'sesame', 'spicenes', 'strawberry',
    'stringbean', 'sugarnes', 'sweetpotato', 'taro', 'tobacco',
    'triticale', 'vetch', 'yam', 'yautia',
] # 'aniseetc'

PEC_CROPS = [
    'abaca', 'agave', 'areca', 'avocado', 'banana',
    'carob', 'cashew', 'cashewapple', 'cinnamon', 'citrusnes',
    'clove', 'coconut', 'date', 'grapefruitetc', 'kapokfiber',
    'karite', 'kolanut', 'lemonlime', 'mango', 'nutmeg',
    'olive', 'orange', 'papaya', 'pepper', 'pineapple',
    'plantain', 'rubber', 'sisal', 'tangetc', 'tea',
    'tropicalnes', 'tung', 'vanilla',
]

PDC_CROPS = [
    'almond', 'apple', 'apricot', 'berrynes', 'blueberry',
    'cherry', 'chestnut', 'cranberry', 'currant', 'fig',
    'fruitnes', 'gooseberry', 'grape', 'hazelnut', 'hop',
    'kiwi', 'peachetc', 'pear', 'persimmon', 'pistachio',
    'plum', 'quince', 'rasberry', 'sourcherry', 'stonefruitnes',
    'walnut',
]

CATEGORY_MAP = {
    'oac': OAC_CROPS,
    'pec': PEC_CROPS,
    'pdc': PDC_CROPS,
}


def get_grid_info(pipeline_dir, crop_list):
    """Get grid dimensions and year list from the first available crop file."""
    pipeline_dir = Path(pipeline_dir)
    for crop in crop_list:
        nc_path = pipeline_dir / f"{crop}.nc"
        if nc_path.exists():
            with xr.open_dataset(nc_path) as ds:
                years = ds['time'].values
                lat = ds['lat'].values
                lon = ds['lon'].values
            return years, lat, lon
    raise FileNotFoundError("No crop files found")


def find_available_crops(pipeline_dir, crop_list, category_name):
    """Check which crop NC files exist."""
    pipeline_dir = Path(pipeline_dir)
    found = [c for c in crop_list if (pipeline_dir / f"{c}.nc").exists()]
    missing = [c for c in crop_list if c not in found]

    logging.info(f"  [{category_name}] Found {len(found)}/{len(crop_list)} crop files")
    if missing:
        logging.warning(f"  [{category_name}] Missing: {missing}")
    if not found:
        raise FileNotFoundError(f"No crop files found for {category_name}")

    return found


def aggregate_one_year(pipeline_dir, crop_list, year_idx, nlat, nlon):
    """
    Load ONE year-slice from all crop NC files using raw netCDF4 to
    minimize memory overhead (no xarray caching).
    """
    import netCDF4 as nc4

    pipeline_dir = Path(pipeline_dir)

    ha_total = np.zeros((nlat, nlon), dtype=np.float64)
    ha_irrigated = np.zeros_like(ha_total)
    ha_rainfed = np.zeros_like(ha_total)

    for crop in crop_list:
        nc_path = pipeline_dir / f"{crop}.nc"
        if not nc_path.exists():
            continue

        ncf = nc4.Dataset(str(nc_path), 'r')

        arr = ncf.variables['harvested_area'][year_idx, :, :]
        ha_total += np.nan_to_num(np.asarray(arr), nan=0.0)

        arr = ncf.variables['harvested_area_irrigated'][year_idx, :, :]
        ha_irrigated += np.nan_to_num(np.asarray(arr), nan=0.0)

        arr = ncf.variables['harvested_area_rainfed'][year_idx, :, :]
        ha_rainfed += np.nan_to_num(np.asarray(arr), nan=0.0)

        ncf.close()
        del ncf, arr

    return (ha_total.astype(np.float32),
            ha_irrigated.astype(np.float32),
            ha_rainfed.astype(np.float32))


def regrid_2d(arr_2d, src_nlat, src_nlon):
    """Regrid 2D array to target resolution, conserving total."""
    if src_nlat == TARGET_NLAT and src_nlon == TARGET_NLON:
        return arr_2d

    zoom_lat = TARGET_NLAT / src_nlat
    zoom_lon = TARGET_NLON / src_nlon

    regridded = zoom(arr_2d, (zoom_lat, zoom_lon), order=1).astype(np.float32)

    src_total = np.nansum(arr_2d)
    if src_total > 0:
        dst_total = np.nansum(regridded)
        if dst_total > 0:
            regridded *= (src_total / dst_total)

    return regridded


def save_one_year_tmp(ha_tot, ha_irr, ha_rain, lat, lon, year_val, tmpdir):
    """Save one year's data to a temporary NC file."""
    tmpfile = Path(tmpdir) / f"year_{int(year_val)}.nc"

    ds = xr.Dataset(
        {
            'harvested_area': (['lat', 'lon'], ha_tot),
            'harvested_area_irrigated': (['lat', 'lon'], ha_irr),
            'harvested_area_rainfed': (['lat', 'lon'], ha_rain),
        },
        coords={
            'time': year_val,
            'lat': lat,
            'lon': lon,
        },
    )

    encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                for var in ['harvested_area', 'harvested_area_irrigated',
                            'harvested_area_rainfed']}
    ds.to_netcdf(tmpfile, encoding=encoding)
    ds.close()
    del ds
    return tmpfile


def concat_year_files(tmp_files, outfile, category_name, found_crops):
    """Concat all yearly tmp files into one final NC along time dim."""
    logging.info(f"  [{category_name}] Concatenating {len(tmp_files)} year files...")

    datasets = []
    for f in sorted(tmp_files):
        ds = xr.open_dataset(f)
        # Expand time dim (was stored as scalar coord)
        ds = ds.expand_dims('time')
        datasets.append(ds)

    combined = xr.concat(datasets, dim='time')

    combined.attrs = {
        'title': f'Gridded harvested area for {category_name} (aggregate)',
        'source': 'Aggregated from individual crop pipeline outputs '
                  '(LUH2-GCB2025 + CROPGRIDS v1.08 + FAOSTAT)',
        'resolution': '5 arcminute (~9.3 km)',
        'units': 'hectares (ha)',
        'n_crops': len(CATEGORY_MAP[category_name]),
        'crops_included': ', '.join(found_crops),
        'created_by': 'aggregate_oac_pec_pdc.py',
    }
    combined['harvested_area'].attrs = {
        'units': 'ha', 'long_name': f'{category_name} total harvested area'}
    combined['harvested_area_irrigated'].attrs = {
        'units': 'ha', 'long_name': f'{category_name} irrigated harvested area'}
    combined['harvested_area_rainfed'].attrs = {
        'units': 'ha', 'long_name': f'{category_name} rainfed harvested area'}

    encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                for var in ['harvested_area', 'harvested_area_irrigated',
                            'harvested_area_rainfed']}
    combined.to_netcdf(outfile, encoding=encoding)

    for ds in datasets:
        ds.close()
    combined.close()
    del datasets, combined


def process_category(pipeline_dir, output_dir, category_name):
    """Process one category year-by-year with tmp files."""
    crop_list = CATEGORY_MAP[category_name]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outfile = output_dir / f"{category_name}.nc"
    if outfile.exists():
        logging.info(f"  [{category_name}] {outfile} already exists, skip")
        return

    found = find_available_crops(pipeline_dir, crop_list, category_name)
    years, src_lat, src_lon = get_grid_info(pipeline_dir, found)
    src_nlat, src_nlon = len(src_lat), len(src_lon)
    nyears = len(years)
    need_regrid = (src_nlat != TARGET_NLAT or src_nlon != TARGET_NLON)

    logging.info(f"  [{category_name}] Source grid: {src_nlat}x{src_nlon}, "
                 f"years: {years[0]}-{years[-1]} ({nyears} years)")
    if need_regrid:
        logging.info(f"  [{category_name}] Will regrid to {TARGET_NLAT}x{TARGET_NLON}")

    out_lat = TARGET_LAT if need_regrid else src_lat
    out_lon = TARGET_LON if need_regrid else src_lon

    tmpdir = output_dir / f"_tmp_{category_name}"
    tmpdir.mkdir(exist_ok=True)

    tmp_files = []
    for t_idx in range(nyears):
        year_int = int(years[t_idx])

        # Check if this year's tmp already exists (resume support)
        tmpfile = tmpdir / f"year_{year_int}.nc"
        if tmpfile.exists():
            logging.info(f"  [{category_name}] {year_int}: tmp exists, skip")
            tmp_files.append(tmpfile)
            continue

        logging.info(f"  [{category_name}] {year_int}: aggregating {len(found)} crops...")

        ha_tot, ha_irr, ha_rain = aggregate_one_year(
            pipeline_dir, found, t_idx, src_nlat, src_nlon)

        if need_regrid:
            ha_tot = regrid_2d(ha_tot, src_nlat, src_nlon)
            ha_irr = regrid_2d(ha_irr, src_nlat, src_nlon)
            ha_rain = regrid_2d(ha_rain, src_nlat, src_nlon)

        irr_mha = np.nansum(ha_irr) / 1e6
        tot_mha = np.nansum(ha_tot) / 1e6
        irr_pct = (irr_mha / tot_mha * 100) if tot_mha > 0 else 0
        logging.info(f"  [{category_name}] {year_int}: total={tot_mha:.2f} M ha, "
                     f"irrigated={irr_mha:.2f} M ha ({irr_pct:.1f}%)")

        tmpf = save_one_year_tmp(ha_tot, ha_irr, ha_rain,
                                 out_lat, out_lon, years[t_idx], tmpdir)
        tmp_files.append(tmpf)

        del ha_tot, ha_irr, ha_rain
        gc.collect()

    concat_year_files(tmp_files, outfile, category_name, found)

    for f in tmp_files:
        os.remove(f)
    tmpdir.rmdir()

    gc.collect()
    logging.info(f"  [{category_name}] Done -> {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate remaining crops into oac/pec/pdc annual HA files"
    )
    parser.add_argument("--pipeline_dir", required=True,
                        help="Directory with annual crop NC files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for aggregate NC files")
    parser.add_argument("--categories", nargs="*", default=['oac', 'pec', 'pdc'],
                        choices=['oac', 'pec', 'pdc'],
                        help="Which categories to process (default: all three)")
    args = parser.parse_args()

    logging.info("=" * 65)
    logging.info("  Aggregate Crops -> OAC / PEC / PDC Annual HA")
    logging.info("=" * 65)
    logging.info(f"  Pipeline dir: {args.pipeline_dir}")
    logging.info(f"  Output dir:   {args.output_dir}")
    logging.info(f"  Categories:   {args.categories}")

    for category in args.categories:
        logging.info(f"\n{'=' * 65}")
        logging.info(f"  Processing {category.upper()} "
                     f"({len(CATEGORY_MAP[category])} crops)")
        logging.info(f"{'=' * 65}")

        process_category(args.pipeline_dir, args.output_dir, category)

    logging.info("\nAll done!")


if __name__ == "__main__":
    main()
