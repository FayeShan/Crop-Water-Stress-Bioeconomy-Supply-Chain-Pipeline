#!/usr/bin/env python3
"""Downscale annual irrigated/rainfed harvested area to monthly using crop calendars."""

import argparse
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from scipy.ndimage import zoom
import warnings
import gc
warnings.filterwarnings("ignore")

# Target grid: 5 arcmin (matching water stress model)
TARGET_NLAT = 2160
TARGET_NLON = 4320
# lat: 89.958... to -89.958..., N→S (descending)
TARGET_LAT = np.linspace(90 - 5/120, -90 + 5/120, TARGET_NLAT)  # cell centers
# lon: -179.958... to 179.958...
TARGET_LON = np.linspace(-180 + 5/120, 180 - 5/120, TARGET_NLON)


# ACEA code → pipeline crop name(s) mapping
ACEA_TO_PIPELINE = {
    'wh':  ['wheat'],
    'ri1': ['rice'],       # main season
    'ri2': ['rice'],       # second season (same pipeline crop, different calendar)
    'cot': ['cotton'],
    'sgc': ['sugarcane'],
    'mai': ['maize'],
    'rap': ['rapeseed'],
    'sor': ['sorghum'],
    'pot': ['potato'],
    'nut': ['groundnut'],
    'sgb': ['sugarbeet'],
    'aff': ['vegfor'],     # alfalfa ≈ fodder
    'bar': ['barley'],
    'ckp': ['chickpea'],
    'bea': ['bean'],
    'sun': ['sunflower'],
    'soy': ['soybean'],
    'mil': ['millet'],
    'cof': ['coffee'],
    'coc': ['cocoa'],
    'cwp': ['cowpea'],
    'cas': ['cassava'],
    'plm': ['oilpalm'],
    'vgt': ['tomato', 'onion', 'cabbage', 'lettuce', 'cucumberetc', 'eggplant',
            'spinach', 'garlic', 'greenonion', 'okra', 'cauliflower', 'carrot',
            'artichoke', 'asparagus', 'pumpkinetc', 'chilleetc', 'watermelon', 'melonetc'],
}

# All 24 ACEA crop codes (excluding oac, pec, pdc aggregates)
ALL_ACEA = list(ACEA_TO_PIPELINE.keys())


def compute_monthly_weights(planting_day, maturity_day):
    """
    Given planting_day and maturity_day arrays (lat, lon), compute
    a (12, lat, lon) weight array indicating fraction of each month
    that falls within the growing season.
    
    Handles cross-year seasons (planting_day > maturity_day, e.g., Nov→Apr).
    
    Weight = (days of month within growing season) / (total growing season days)
    Sum of weights across 12 months = 1.0 for each grid cell.
    """
    nlat, nlon = planting_day.shape
    weights = np.zeros((12, nlat, nlon), dtype=np.float32)
    
    # Month start/end in day-of-year
    # Jan: 1-31, Feb: 32-59, Mar: 60-90, ..., Dec: 336-365
    month_starts = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 336])
    month_ends   = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 335, 365])
    month_days   = month_ends - month_starts + 1  # [31, 28, 31, 30, ...]
    
    # Valid mask: both planting and maturity defined
    valid = np.isfinite(planting_day) & np.isfinite(maturity_day) & (planting_day > 0) & (maturity_day > 0)
    
    p = planting_day.copy()
    m = maturity_day.copy()
    
    # Compute growing season length
    # If p <= m: normal season (e.g., Apr→Sep), length = m - p + 1
    # If p > m:  cross-year (e.g., Nov→Apr), length = (365 - p + 1) + m
    normal = p <= m
    gs_length = np.where(normal, m - p + 1, (365 - p + 1) + m)
    gs_length = np.maximum(gs_length, 1)  # avoid division by zero
    
    for month_idx in range(12):
        ms = month_starts[month_idx]
        me = month_ends[month_idx]
        
        # For each grid cell, compute how many days of this month
        # fall within the growing season [p, m] (potentially wrapping)
        
        # Normal season (p <= m): growing days = overlap of [p, m] with [ms, me]
        overlap_normal = np.maximum(0, np.minimum(me, m) - np.maximum(ms, p) + 1)
        
        # Cross-year season (p > m): season covers [p, 365] + [1, m]
        # Overlap with [ms, me] = overlap([p,365],[ms,me]) + overlap([1,m],[ms,me])
        overlap_tail = np.maximum(0, np.minimum(me, 365) - np.maximum(ms, p) + 1)  # [p, 365]
        overlap_head = np.maximum(0, np.minimum(me, m) - np.maximum(ms, 1) + 1)    # [1, m]
        overlap_cross = overlap_tail + overlap_head
        
        overlap_days = np.where(normal, overlap_normal, overlap_cross)
        overlap_days = np.clip(overlap_days, 0, month_days[month_idx])
        
        # Weight = days in growing season this month / total growing season length
        weights[month_idx] = np.where(valid, overlap_days / gs_length, 0).astype(np.float32)
    
    return weights


def aggregate_to_5arcmin(data_2d, src_lat, src_lon):
    """
    Aggregate a 2D array from pipeline grid (0.05°, S→N, 3600×7200)
    to 5-arcmin target grid (N→S, 2160×4320) by summing.
    
    Since 0.05° and 5 arcmin are not exact multiples, we use
    xarray's groupby_bins for accurate spatial aggregation.
    """
    # Quick path: 0.05° = 3 arcmin, target = 5 arcmin
    # Not an integer ratio, so we use bin-based aggregation
    
    # Ensure src is S→N (pipeline convention)
    if len(src_lat) > 1 and src_lat[0] > src_lat[-1]:
        data_2d = data_2d[::-1, :]
        src_lat = src_lat[::-1]
    
    da = xr.DataArray(
        data_2d,
        dims=['lat', 'lon'],
        coords={'lat': src_lat, 'lon': src_lon}
    )
    
    # Define bin edges for 5-arcmin grid
    lat_step = 5 / 60  # 0.08333...
    lat_edges = np.linspace(-90, 90, TARGET_NLAT + 1)
    lon_edges = np.linspace(-180, 180, TARGET_NLON + 1)
    
    binned = da.groupby_bins('lat', lat_edges, labels=TARGET_LAT).sum(dim='lat')
    binned = binned.groupby_bins('lon', lon_edges, labels=TARGET_LON).sum(dim='lon')

    result = binned.values.astype(np.float32)
    result = result[::-1, :]  # S→N → N→S to match target grid

    result = np.where(np.isfinite(result), result, 0)
    
    return result


def aggregate_to_5arcmin_fast(data_2d, src_lat, src_lon):
    """
    Fast approximate aggregation using scipy.ndimage.zoom.
    
    Pipeline: 3600×7200 (0.05°) → Target: 2160×4320 (5 arcmin)
    Zoom factor: 0.6 × 0.6
    
    For harvested area (extensive variable), we need to SUM not average.
    Strategy: zoom with order=1 (bilinear) then scale by area ratio.
    
    Actually, for best accuracy with non-integer ratios, we use a 
    reshape-based approach via an intermediate common grid.
    """
    # Ensure S→N
    if len(src_lat) > 1 and src_lat[0] > src_lat[-1]:
        data_2d = data_2d[::-1, :]
        src_lat = src_lat[::-1]
    
    # Approach: resample via intermediate grid
    # 0.05° = 3', target = 5'. LCM = 15'. 
    # Pipeline has 3600 lat cells (3' each), target has 2160 (5' each)
    # Group every 5 pipeline lat rows? No: 5' / 3' = 1.667, not integer.
    #
    # Better: use the exact bin approach but with numpy for speed
    
    nlat_src, nlon_src = data_2d.shape
    result = np.zeros((TARGET_NLAT, TARGET_NLON), dtype=np.float64)
    
    # Map each source cell to target cell
    # Source lat: S→N from src_lat[0] to src_lat[-1]
    # Target lat: N→S from TARGET_LAT[0] to TARGET_LAT[-1]
    
    lat_step_tgt = 5 / 60  # 0.08333°
    lon_step_tgt = 5 / 60
    
    # For each source cell, find which target cell it falls into
    # Target lat is N→S, so target_lat_idx = (90 - src_lat) / lat_step_tgt
    src_lat_idx = ((90 - src_lat) / lat_step_tgt).astype(int)
    src_lat_idx = np.clip(src_lat_idx, 0, TARGET_NLAT - 1)
    
    src_lon_idx = ((src_lon + 180) / lon_step_tgt).astype(int)
    src_lon_idx = np.clip(src_lon_idx, 0, TARGET_NLON - 1)
    
    # Use np.add.at for accumulation
    for i in range(nlat_src):
        ti = src_lat_idx[i]
        np.add.at(result[ti], src_lon_idx, data_2d[i])
    
    return result.astype(np.float32)


def load_pipeline_annual(pipeline_dir, pipeline_crops):
    """
    Load and sum harvested_area_irrigated/rainfed from pipeline outputs,
    then aggregate from 0.05° to 5-arcmin target grid.
    Returns: (ha_irr, ha_rain, years)  — shapes (nyears, 2160, 4320)
    """
    pipeline_dir = Path(pipeline_dir)
    ha_irr_raw = None
    ha_rain_raw = None
    src_lat = src_lon = years = None
    
    for crop_name in pipeline_crops:
        nc_path = pipeline_dir / f"{crop_name}.nc"
        if not nc_path.exists():
            print(f"    WARNING: {nc_path.name} not found, skipping")
            continue
        
        ds = xr.open_dataset(nc_path)
        ha_irr = ds['harvested_area_irrigated'].values
        ha_rain = ds['harvested_area_rainfed'].values
        
        ha_irr = np.where(np.isfinite(ha_irr), ha_irr, 0)
        ha_rain = np.where(np.isfinite(ha_rain), ha_rain, 0)
        
        if ha_irr_raw is None:
            ha_irr_raw = ha_irr
            ha_rain_raw = ha_rain
            src_lat = ds['lat'].values
            src_lon = ds['lon'].values
            years = ds['time'].values
        else:
            ha_irr_raw += ha_irr
            ha_rain_raw += ha_rain
        
        ds.close()
    
    if ha_irr_raw is None:
        return None, None, None
    
    nyears = len(years)
    print(f"    Aggregating {nyears} years from 0.05° to 5 arcmin...")
    
    ha_irr_5m = np.zeros((nyears, TARGET_NLAT, TARGET_NLON), dtype=np.float32)
    ha_rain_5m = np.zeros((nyears, TARGET_NLAT, TARGET_NLON), dtype=np.float32)
    
    for yi in range(nyears):
        ha_irr_5m[yi] = aggregate_to_5arcmin_fast(ha_irr_raw[yi], src_lat, src_lon)
        ha_rain_5m[yi] = aggregate_to_5arcmin_fast(ha_rain_raw[yi], src_lat, src_lon)
    
    del ha_irr_raw, ha_rain_raw
    gc.collect()
    
    return ha_irr_5m, ha_rain_5m, years


def load_crop_calendar(calendar_dir, acea_code, irrf='ir'):
    """
    Load planting_day and maturity_day from crop calendar.
    Returns: (planting_day, maturity_day, cal_lat, cal_lon)
    """
    calendar_dir = Path(calendar_dir)
    cal_file = calendar_dir / f"{acea_code}_{irrf}_crop_calendar.nc"
    
    if not cal_file.exists():
        return None, None, None, None
    
    ds = xr.open_dataset(cal_file)
    pday = ds['planting_day'].values   # (lat, lon), float32, day-of-year
    mday = ds['maturity_day'].values
    cal_lat = ds['lat'].values
    cal_lon = ds['lon'].values
    ds.close()
    
    return pday, mday, cal_lat, cal_lon


def regrid_calendar_to_target(cal_data, cal_lat, cal_lon):
    """
    Regrid crop calendar (0.5°, 360×720) to target 5-arcmin grid (2160×4320).
    Uses nearest-neighbor. Both grids are N→S.
    """
    from scipy.ndimage import zoom
    
    # Ensure calendar is N→S (it already is: lat starts at 89.75)
    cal_descending = cal_lat[0] > cal_lat[-1]
    if not cal_descending:
        cal_data = cal_data[::-1, :]
    
    # Handle NaN before zoom
    valid_mask = np.isfinite(cal_data) & (cal_data > 0)
    data_clean = np.where(valid_mask, cal_data, 0)
    
    # 360→2160 = 6x, 720→4320 = 6x (exact integer ratio!)
    zoom_lat = TARGET_NLAT / len(cal_lat)  # 2160/360 = 6.0
    zoom_lon = TARGET_NLON / len(cal_lon)  # 4320/720 = 6.0
    
    data_zoomed = zoom(data_clean, (zoom_lat, zoom_lon), order=0)
    mask_zoomed = zoom(valid_mask.astype(np.float32), (zoom_lat, zoom_lon), order=0) > 0.5
    
    h = min(data_zoomed.shape[0], TARGET_NLAT)
    w = min(data_zoomed.shape[1], TARGET_NLON)
    
    result = np.full((TARGET_NLAT, TARGET_NLON), np.nan, dtype=np.float32)
    result[:h, :w] = np.where(mask_zoomed[:h, :w], data_zoomed[:h, :w], np.nan)
    
    return result


def process_crop(acea_code, pipeline_dir, calendar_dir, output_dir):
    """
    For one ACEA crop:
      1. Load annual irrigated/rainfed HA from pipeline, aggregate to 5 arcmin
      2. Load crop calendar (planting_day, maturity_day), regrid to 5 arcmin
      3. Compute monthly weights
      4. Distribute annual HA → 12 months per year
      5. Save as NetCDF
    """
    pipeline_crops = ACEA_TO_PIPELINE[acea_code]
    
    # 1. Load annual pipeline data (aggregated to 5 arcmin)
    ha_irr, ha_rain, years = load_pipeline_annual(pipeline_dir, pipeline_crops)
    if ha_irr is None:
        print(f"    No pipeline data found, skipping")
        return None
    
    nyears = len(years)
    nlat = TARGET_NLAT
    nlon = TARGET_NLON
    print(f"    Data: {nyears} years, {nlat}×{nlon} (5 arcmin)")
    
    # 2. Load crop calendars (irrigated and rainfed separately)
    pday_ir, mday_ir, cal_lat, cal_lon = load_crop_calendar(calendar_dir, acea_code, 'ir')
    pday_rf, mday_rf, _, _ = load_crop_calendar(calendar_dir, acea_code, 'rf')
    
    if pday_ir is None and pday_rf is None:
        print(f"    No crop calendar found, skipping")
        return None
    
    # 3. Regrid calendar to 5-arcmin target grid
    print(f"    Regridding calendar 360×720 → {nlat}×{nlon}...")
    
    if pday_ir is not None:
        pday_ir_5m = regrid_calendar_to_target(pday_ir, cal_lat, cal_lon)
        mday_ir_5m = regrid_calendar_to_target(mday_ir, cal_lat, cal_lon)
        weights_ir = compute_monthly_weights(pday_ir_5m, mday_ir_5m)
        print(f"    IR calendar: {(np.isfinite(pday_ir_5m) & (pday_ir_5m > 0)).sum()} valid cells")
    else:
        weights_ir = np.zeros((12, nlat, nlon), dtype=np.float32)
    
    if pday_rf is not None:
        pday_rf_5m = regrid_calendar_to_target(pday_rf, cal_lat, cal_lon)
        mday_rf_5m = regrid_calendar_to_target(mday_rf, cal_lat, cal_lon)
        weights_rf = compute_monthly_weights(pday_rf_5m, mday_rf_5m)
        print(f"    RF calendar: {(np.isfinite(pday_rf_5m) & (pday_rf_5m > 0)).sum()} valid cells")
    else:
        weights_rf = np.zeros((12, nlat, nlon), dtype=np.float32)
    
    # 4. Distribute annual → monthly for each year
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for yi, year in enumerate(years):
        year_int = int(year) if not hasattr(year, 'year') else year.year if hasattr(year, 'year') else int(year)
        
        # Annual HA for this year (already at 5 arcmin)
        annual_irr = ha_irr[yi]   # (nlat, nlon)
        annual_rain = ha_rain[yi]
        
        # Monthly = annual × weight_month
        monthly_irr = weights_ir * annual_irr[np.newaxis, :, :]    # (12, nlat, nlon)
        monthly_rain = weights_rf * annual_rain[np.newaxis, :, :]
        monthly_total = monthly_irr + monthly_rain
        
        time_coords = pd.date_range(f"{year_int}-01-01", periods=12, freq='MS')
        
        ds_out = xr.Dataset(
            {
                'harvested_area_irrigated': (['time', 'lat', 'lon'], monthly_irr.astype(np.float32)),
                'harvested_area_rainfed':   (['time', 'lat', 'lon'], monthly_rain.astype(np.float32)),
                'harvested_area_total':     (['time', 'lat', 'lon'], monthly_total.astype(np.float32)),
            },
            coords={
                'time': time_coords,
                'lat': TARGET_LAT,
                'lon': TARGET_LON,
            },
            attrs={
                'title': f'Monthly harvested area for {acea_code} ({year_int})',
                'source': 'Pipeline (LUH2+CROPGRIDS+FAOSTAT) + crop calendar downscaling',
                'acea_code': acea_code,
                'pipeline_crops': ','.join(pipeline_crops),
                'resolution': '5 arcmin (~0.0833 degree)',
                'units': 'hectares (ha)',
                'downscaling_method': 'Growing season proportional allocation',
            }
        )
        
        for var in ['harvested_area_irrigated', 'harvested_area_rainfed', 'harvested_area_total']:
            ds_out[var].attrs = {'units': 'ha', 'long_name': f'Monthly {var.replace("_", " ")}'}
        
        outfile = output_dir / f"{acea_code}_{year_int}.nc"
        encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                    for var in ['harvested_area_irrigated', 'harvested_area_rainfed', 'harvested_area_total']}
        ds_out.to_netcdf(outfile, encoding=encoding)
        ds_out.close()
        saved_files.append(outfile)
    
    y0 = int(years[0]) if not hasattr(years[0], 'year') else years[0]
    monthly_irr_sum = (weights_ir * ha_irr[0][np.newaxis, :, :]).sum(axis=(1, 2)) / 1e6
    print(f"    Year {y0}: irrigated HA by month (M ha):")
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for m in range(12):
        bar = '█' * int(monthly_irr_sum[m] * 2) if monthly_irr_sum[m] > 0.01 else ''
        print(f"      {month_names[m]}: {monthly_irr_sum[m]:6.2f} {bar}")
    
    print(f"    Saved {len(saved_files)} files to {output_dir}/")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Monthly downscale of annual irrigated/rainfed HA")
    parser.add_argument("--pipeline_dir", required=True, help="Directory with pipeline .nc outputs")
    parser.add_argument("--calendar_dir", required=True, help="Directory with crop calendar .nc files")
    parser.add_argument("--output_dir", default="./monthly", help="Output directory")
    parser.add_argument("--crops", nargs="*", default=None, 
                        help="ACEA crop codes to process (default: all 24)")
    args = parser.parse_args()
    
    crops = args.crops if args.crops else ALL_ACEA
    
    print("=" * 65)
    print("  Monthly Downscaling: Annual → 12 Months")
    print("=" * 65)
    print(f"  Crops to process: {len(crops)}")
    print(f"  Pipeline dir: {args.pipeline_dir}")
    print(f"  Calendar dir: {args.calendar_dir}")
    print(f"  Output dir:   {args.output_dir}")
    
    results = []
    
    for i, acea in enumerate(crops):
        print(f"\n  [{i+1}/{len(crops)}] {acea}")
        files = process_crop(acea, args.pipeline_dir, args.calendar_dir, args.output_dir)
        if files:
            results.append((acea, len(files)))
        gc.collect()
    
    print(f"\n{'=' * 65}")
    print(f"  DONE! Processed {len(results)} crops")
    for acea, nfiles in results:
        print(f"    {acea}: {nfiles} yearly files")
    print(f"  Output: {args.output_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
