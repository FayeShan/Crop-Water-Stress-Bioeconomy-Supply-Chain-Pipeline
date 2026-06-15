#!/usr/bin/env python3
"""Gridded irrigated/rainfed harvested area pipeline (Method B): LUH2 + CROPGRIDS + FAOSTAT."""

import argparse
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import zoom
from scipy.ndimage import uniform_filter
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# CFT Mapping: CROPGRIDS crop name → LUH2 Crop Functional Type
# Rules:
#   c3ann  = C3 annual crops (wheat, rice, barley, potato, sugar beet, rapeseed, cotton...)
#   c4ann  = C4 annual crops (maize, sorghum, millet, sugarcane)
#   c3per  = C3 perennial/tree crops (coffee, cocoa, tea, grapes, most fruits, rubber, oil palm...)
#   c4per  = C4 perennial crops (very few — we map oil palm here as c3per since LUH2 treats it so)
#   c3nfx  = C3 nitrogen-fixing crops (soybean, groundnut, all pulses/legumes)

# Note: type='treecrop' from colleague's data → likely c3per
#        type='crop' + N-fixing (legumes) → c3nfx
#        type='crop' + C4 pathway → c4ann
#        type='crop' + C3 annual → c3ann

CFT_MAP = {
    # === C4 annuals (c4ann) ===
    "maize": "c4ann", "sorghum": "c4ann", "millet": "c4ann",
    "sugarcane": "c4ann", "fonio": "c4ann",

    # === C3 nitrogen-fixing (c3nfx) — legumes/pulses ===
    "soybean": "c3nfx", "groundnut": "c3nfx", "bean": "c3nfx",
    "chickpea": "c3nfx", "lentil": "c3nfx", "pea": "c3nfx",
    "cowpea": "c3nfx", "pigeonpea": "c3nfx", "broadbean": "c3nfx",
    "lupin": "c3nfx", "vetch": "c3nfx", "bambara": "c3nfx",
    "pulsenes": "c3nfx", "stringbean": "c3nfx",

    # === C3 perennials (c3per) — tree/plantation crops ===
    "abaca": "c3per", "agave": "c3per", "almond": "c3per",
    "apple": "c3per", "apricot": "c3per", "areca": "c3per",
    "avocado": "c3per", "banana": "c3per", "carob": "c3per",
    "cashew": "c3per", "cashewapple": "c3per", "cherry": "c3per",
    "chestnut": "c3per", "citrusnes": "c3per", "cocoa": "c3per",
    "coconut": "c3per", "coffee": "c3per", "date": "c3per",
    "fig": "c3per", "fruitnes": "c3per", "grape": "c3per",
    "grapefruitetc": "c3per", "hazelnut": "c3per", "karite": "c3per",
    "kiwi": "c3per", "kolanut": "c3per", "lemonlime": "c3per",
    "mango": "c3per", "oilpalm": "c3per", "olive": "c3per",
    "orange": "c3per", "papaya": "c3per", "peachetc": "c3per",
    "pear": "c3per", "persimmon": "c3per", "pineapple": "c3per",
    "pistachio": "c3per", "plantain": "c3per", "plum": "c3per",
    "quince": "c3per", "rasberry": "c3per", "rubber": "c3per",
    "sourcherry": "c3per", "stonefruitnes": "c3per", "tea": "c3per",
    "tung": "c3per", "vanilla": "c3per", "walnut": "c3per",

    # === C3 annuals (c3ann) — everything else ===
    "aniseetc": "c3ann", "artichoke": "c3ann", "asparagus": "c3ann",
    "barley": "c3ann", "berrynes": "c3ann", "blueberry": "c3ann",
    "buckwheat": "c3ann", "cabbage": "c3ann", "canaryseed": "c3ann",
    "carrot": "c3ann", "cassava": "c3ann", "castor": "c3ann",
    "cauliflower": "c3ann", "cerealnes": "c3ann", "chicory": "c3ann",
    "chilleetc": "c3ann", "cinnamon": "c3ann", "clove": "c3ann",
    "cotton": "c3ann", "cranberry": "c3ann", "cucumberetc": "c3ann",
    "currant": "c3ann", "eggplant": "c3ann", "fibrenes": "c3ann",
    "flax": "c3ann", "garlic": "c3ann", "ginger": "c3ann",
    "gooseberry": "c3ann", "greenonion": "c3ann", "hempseed": "c3ann",
    "hop": "c3ann", "jute": "c3ann", "kapokfiber": "c3ann",
    "lettuce": "c3ann", "linseed": "c3ann", "melonetc": "c3ann",
    "melonseed": "c3ann", "mixedgrain": "c3ann", "mustard": "c3ann",
    "nutmeg": "c3ann", "oats": "c3ann", "oilseednes": "c3ann",
    "okra": "c3ann", "onion": "c3ann", "other": "c3ann",
    "pepper": "c3ann", "peppermint": "c3ann", "poppy": "c3ann",
    "potato": "c3ann", "pumpkinetc": "c3ann", "pyrethrum": "c3ann",
    "quinoa": "c3ann", "ramie": "c3ann", "rapeseed": "c3ann",
    "rice": "c3ann", "rootnes": "c3ann", "rye": "c3ann",
    "safflower": "c3ann", "sesame": "c3ann", "sisal": "c3ann",
    "spicenes": "c3ann", "spinach": "c3ann", "strawberry": "c3ann",
    "sugarbeet": "c3ann", "sugarnes": "c3ann", "sunflower": "c3ann",
    "sweetpotato": "c3ann", "tangetc": "c3ann", "taro": "c3ann",
    "tobacco": "c3ann", "tomato": "c3ann", "triticale": "c3ann",
    "tropicalnes": "c3ann", "vegfor": "c3ann", "watermelon": "c3ann",
    "wheat": "c3ann", "yam": "c3ann", "yautia": "c3ann",
}

CFTS = ["c3ann", "c4ann", "c3per", "c4per", "c3nfx"]
YEARS = list(range(2000, 2025))


def load_luh2(luh2_dir):
    """
    Load LUH2 states + management, subset to 2000-2024.
    Returns CFT fractions, irrigation fractions, and cell area.
    All at 0.25° resolution.
    """
    luh2_dir = Path(luh2_dir)

    # Find files (support both GCB2025 and standard naming)
    states_file = None
    mgmt_file = None
    static_file = None

    for name in ["states4.nc", "states.nc", "LUH_update_states4.nc"]:
        f = luh2_dir / name
        if f.exists():
            states_file = f
            break

    for name in ["management4.nc", "management.nc", "LUH_update_management4.nc"]:
        f = luh2_dir / name
        if f.exists():
            mgmt_file = f
            break

    for name in ["staticData_quarterdeg.nc", "staticData_quarterdeg4.nc"]:
        f = luh2_dir / name
        if f.exists():
            static_file = f
            break

    if states_file is None:
        raise FileNotFoundError(f"No LUH2 states file found in {luh2_dir}. "
                                f"Expected states4.nc or states.nc")
    if mgmt_file is None:
        raise FileNotFoundError(f"No LUH2 management file found in {luh2_dir}")

    print(f"  States file: {states_file.name}")
    print(f"  Management file: {mgmt_file.name}")
    print(f"  Static file: {static_file.name if static_file else 'NOT FOUND'}")

    print("  Loading states (CFT fractions)...")
    ds_states = xr.open_dataset(states_file, decode_times=False)

    # Decode time: LUH2 uses "years since 850-01-01" with 365_day calendar
    # With decode_times=False, raw values are floats: 0.0=year850, 1.0=year851, ...
    if 'time' in ds_states.dims:
        raw_time = ds_states['time'].values
        actual_years = np.round(raw_time).astype(int)
        if actual_years[0] < 100:
            actual_years = actual_years + 850

        year_mask = (actual_years >= 2000) & (actual_years <= 2024)
        time_indices = np.where(year_mask)[0]

        if len(time_indices) == 0:
            print(f"  WARNING: No years 2000-2024 found. Year range: {actual_years[0]}-{actual_years[-1]}")
            print(f"  First 5 years: {actual_years[:5]}, Last 5: {actual_years[-5:]}")
            sys.exit(1)

        print(f"  Subsetting to {len(time_indices)} years ({actual_years[time_indices[0]]}-{actual_years[time_indices[-1]]})")

        cft_data = {}
        for cft in CFTS:
            if cft in ds_states:
                cft_data[cft] = ds_states[cft].isel(time=time_indices).values  # (time, lat, lon)
                print(f"    {cft}: shape {cft_data[cft].shape}")
            else:
                print(f"    WARNING: {cft} not found in states file")
        
        selected_years = actual_years[time_indices]
    else:
        raise ValueError("No time dimension found in LUH2 states file")

    lat_luh = ds_states['lat'].values
    lon_luh = ds_states['lon'].values
    ds_states.close()

    print("  Loading management (irrigation fractions)...")
    ds_mgmt = xr.open_dataset(mgmt_file, decode_times=False)

    irrig_data = {}
    for cft in CFTS:
        irrig_var = f"irrig_{cft}"
        if irrig_var in ds_mgmt:
            irrig_data[cft] = ds_mgmt[irrig_var].isel(time=time_indices).values
            print(f"    {irrig_var}: shape {irrig_data[cft].shape}")
        else:
            print(f"    WARNING: {irrig_var} not found in management file")
    ds_mgmt.close()

    carea = None
    if static_file is not None:
        ds_static = xr.open_dataset(static_file)
        if 'carea' in ds_static:
            carea = ds_static['carea'].values  # km²
            print(f"  Cell area: shape {carea.shape}, range {np.nanmin(carea):.1f}-{np.nanmax(carea):.1f} km²")
        ds_static.close()

    if carea is None:
        # Compute approximate cell area from latitude
        print("  Computing cell area from latitude...")
        R = 6371.0  # Earth radius in km
        dlat = np.abs(lat_luh[1] - lat_luh[0])  # 0.25°
        dlon = np.abs(lon_luh[1] - lon_luh[0])
        lat_rad = np.deg2rad(lat_luh)
        # Area of each cell = R² × dlat_rad × dlon_rad × cos(lat)
        cell_height = R * np.deg2rad(dlat)
        cell_width = R * np.deg2rad(dlon) * np.cos(lat_rad)
        carea = np.outer(cell_height * cell_width, np.ones(len(lon_luh)))
        # Fix: outer gives wrong shape, do it properly
        carea = (cell_width * cell_height)[:, np.newaxis] * np.ones((1, len(lon_luh)))

    return {
        'cft_data': cft_data,       # dict: cft_name → (nyears, nlat, nlon)
        'irrig_data': irrig_data,    # dict: cft_name → (nyears, nlat, nlon)
        'carea': carea,              # (nlat, nlon) in km²
        'lat': lat_luh,
        'lon': lon_luh,
        'years': selected_years,
    }


def load_cropgrids(cropgrids_dir, crops_to_process=None):
    """
    Load CROPGRIDS harvested area per crop.
    Returns crop_ha dict and country_mask.
    """
    cropgrids_dir = Path(cropgrids_dir)

    nc_files = sorted(cropgrids_dir.glob("CROPGRIDSv1.08_*.nc"))
    if not nc_files:
        nc_files = sorted(cropgrids_dir.glob("*.nc"))

    print(f"  Found {len(nc_files)} NetCDF files in {cropgrids_dir}")

    country_file = cropgrids_dir.parent / "Countries_2018.nc"
    if not country_file.exists():
        country_file = cropgrids_dir / "Countries_2018.nc"
    if not country_file.exists():
        # Search more broadly
        for p in [cropgrids_dir.parent.parent, cropgrids_dir.parent]:
            cf = p / "Countries_2018.nc"
            if cf.exists():
                country_file = cf
                break

    country_mask = None
    if country_file.exists():
        ds_c = xr.open_dataset(country_file)
        for var in ds_c.data_vars:
            country_mask = ds_c[var].values
            break
        ds_c.close()
        print(f"  Country mask: shape {country_mask.shape}, {int(np.nanmax(country_mask[~np.isnan(country_mask)]))} countries")
    else:
        print(f"  WARNING: Countries_2018.nc not found. FAOSTAT scaling will not work spatially.")

    crop_ha = {}
    crop_lat = None
    crop_lon = None

    for nc_file in nc_files:
        crop_name = nc_file.stem.replace("CROPGRIDSv1.08_", "").lower()

        if crop_name in ["countries_2018", "countries"]:
            continue

        if crops_to_process is not None and crop_name not in crops_to_process:
            continue

        # Skip crops not in our CFT map
        if crop_name not in CFT_MAP:
            continue

        try:
            ds = xr.open_dataset(nc_file)

            # Get harvarea variable (CROPGRIDS v1.08 uses 'harvarea')
            ha_var = None
            for var in ds.data_vars:
                if var == "harvarea":
                    ha_var = var
                    break
            if ha_var is None:
                for var in ds.data_vars:
                    if "harvest" in var.lower() or var.lower() == "ha":
                        ha_var = var
                        break

            if ha_var is not None:
                data = ds[ha_var].values  # (lat, lon)
                data = np.where(np.isfinite(data) & (data > 0), data, 0).astype(np.float32)
                crop_ha[crop_name] = data

                if crop_lat is None:
                    crop_lat = ds['lat'].values
                    crop_lon = ds['lon'].values

            ds.close()
        except Exception as e:
            print(f"    WARNING: Could not load {nc_file.name}: {e}")

    print(f"  Loaded harvested area for {len(crop_ha)} crops")

    return crop_ha, country_mask, crop_lat, crop_lon


def compute_crop_shares(crop_ha):
    """
    For each CFT, compute the fractional share of each crop.
    share(i, g) = HA_cropgrids(i, g) / sum_j_in_CFT(HA_cropgrids(j, g))
    """
    cft_totals = {}
    for cft in CFTS:
        crops_in_cft = [c for c, k in CFT_MAP.items() if k == cft and c in crop_ha]
        if crops_in_cft:
            total = np.zeros_like(list(crop_ha.values())[0], dtype=np.float64)
            for c in crops_in_cft:
                total += crop_ha[c].astype(np.float64)
            cft_totals[cft] = total
            print(f"    {cft}: {len(crops_in_cft)} crops, max total HA = {total.max():.0f}")

    crop_shares = {}
    for crop_name, ha in crop_ha.items():
        cft = CFT_MAP.get(crop_name)
        if cft and cft in cft_totals:
            with np.errstate(divide='ignore', invalid='ignore'):
                share = ha.astype(np.float64) / cft_totals[cft]
            share = np.where(np.isfinite(share), share, 0).astype(np.float32)
            crop_shares[crop_name] = share

    print(f"  Computed shares for {len(crop_shares)} crops")
    return crop_shares


def load_faostat_change(faostat_path, faostat_2024_path=None):
    """
    Load colleague's pre-computed FAOSTAT change factors.
    change_prop = FAOSTAT_area(crop, country, year) / CROPGRIDS_national_sum

    Returns: dict keyed by (cropgrids_name, country_code, year) → change_prop
    """
    df = pd.read_csv(faostat_path)
    print(f"  Loaded {len(df)} records from {Path(faostat_path).name}")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Crops: {df['CROPGRIDS'].nunique()}, Countries: {df['Area'].nunique()}")

    if faostat_2024_path is not None and Path(faostat_2024_path).exists():
        df24 = pd.read_csv(faostat_2024_path)
        df24 = df24.rename(columns={"area_ha": "area_y"})
        if 'base_area' not in df24.columns:
            df24['base_area'] = df24['bbase_area']
        df = pd.concat([df, df24], ignore_index=True)
        print(f"  + Added 2024: total {len(df)} records")

    # Clean: remove 'other', 'yautia', NaN crop names
    df = df[df['CROPGRIDS'].notna()]
    df = df[~df['CROPGRIDS'].isin(['other', 'yautia', 'No', 'No '])]

    # Clip extreme change_prop values
    df['change_prop'] = df['change_prop'].clip(0, 10)

    change_dict = {}
    for _, row in df.iterrows():
        key = (row['CROPGRIDS'], int(row['country_code']) if pd.notna(row['country_code']) else -1, int(row['year']))
        change_dict[key] = row['change_prop']

    print(f"  Built lookup with {len(change_dict)} entries")

    country_codes = df['country_code'].dropna().unique().astype(int)
    return change_dict, country_codes


def regrid_luh2_to_cropgrids(luh2_2d, luh2_lat, luh2_lon, cg_lat, cg_lon):
    """
    Regrid a single 2D LUH2 field (0.25°) to CROPGRIDS grid (0.05°).
    Uses nearest-neighbor (each LUH2 cell → 5×5 CROPGRIDS cells).
    Handles lat flip (LUH2=N→S, CROPGRIDS=S→N) and NaN in ocean cells.
    """
    data = luh2_2d.copy()

    # 1. Replace NaN with 0 (ocean cells) BEFORE zoom to prevent NaN spreading
    data = np.where(np.isfinite(data), data, 0).astype(np.float32)

    # 2. Flip lat if LUH2 is N→S but CROPGRIDS is S→N
    luh2_descending = luh2_lat[0] > luh2_lat[-1]
    cg_ascending = cg_lat[0] < cg_lat[-1]
    if luh2_descending and cg_ascending:
        data = data[::-1, :]

    # 3. Zoom to CROPGRIDS resolution
    zoom_lat = len(cg_lat) / len(luh2_lat)
    zoom_lon = len(cg_lon) / len(luh2_lon)
    regridded = zoom(data, (zoom_lat, zoom_lon), order=0)

    # 4. Handle potential size mismatch
    if regridded.shape[0] != len(cg_lat) or regridded.shape[1] != len(cg_lon):
        result = np.zeros((len(cg_lat), len(cg_lon)), dtype=np.float32)
        h = min(regridded.shape[0], len(cg_lat))
        w = min(regridded.shape[1], len(cg_lon))
        result[:h, :w] = regridded[:h, :w]
        return result

    return regridded.astype(np.float32)


def compute_cropgrids_cell_area(cg_lat, cg_lon):
    """Compute cell area in hectares at CROPGRIDS resolution (0.05°)."""
    R = 6371.0  # km
    dlat = np.abs(cg_lat[1] - cg_lat[0])  # 0.05°
    dlon = np.abs(cg_lon[1] - cg_lon[0])
    lat_rad = np.deg2rad(cg_lat)
    cell_height_km = R * np.deg2rad(dlat)
    cell_width_km = R * np.deg2rad(dlon) * np.cos(lat_rad)
    carea_km2 = (cell_width_km * cell_height_km)[:, np.newaxis] * np.ones((1, len(cg_lon)))
    carea_ha = carea_km2 * 100  # 1 km² = 100 ha
    return carea_ha.astype(np.float32)


def process_crop(crop_name, crop_share, luh2, cg_lat, cg_lon, cg_carea_ha,
                 change_dict, country_mask):
    """
    Process a single crop: compute HA_total, HA_irrigated, HA_rainfed for all years.

    Formula:
      HA(i, g, t) = [LUH2_CFT_frac(k, g, t) × CellArea(g)] × share(i, g) × f(i, c(g), t)
      HA_irr(i, g, t) = HA(i, g, t) × irrig_frac(k, g, t)
      HA_rain(i, g, t) = HA(i, g, t) × (1 - irrig_frac(k, g, t))
    """
    cft = CFT_MAP[crop_name]
    nyears = len(luh2['years'])

    # Preallocate output arrays at CROPGRIDS resolution
    ha_total = np.zeros((nyears, len(cg_lat), len(cg_lon)), dtype=np.float32)
    ha_irrig = np.zeros_like(ha_total)
    ha_rain = np.zeros_like(ha_total)

    cft_fracs = luh2['cft_data'].get(cft)
    irrig_fracs = luh2['irrig_data'].get(cft)

    if cft_fracs is None:
        print(f"    Skipping {crop_name}: CFT {cft} not in LUH2")
        return None

    for ti, year in enumerate(luh2['years']):
        # 1. Get LUH2 CFT fraction for this year and regrid to CROPGRIDS resolution
        cft_frac_025 = cft_fracs[ti]  # (nlat_luh, nlon_luh), fraction 0-1
        cft_frac_005 = regrid_luh2_to_cropgrids(
            cft_frac_025, luh2['lat'], luh2['lon'], cg_lat, cg_lon
        )

        # 2. Convert fraction to absolute area (ha) at CROPGRIDS resolution
        cft_area_ha = cft_frac_005 * cg_carea_ha  # ha

        # 3. Allocate to this crop using CROPGRIDS share
        ha_raw = cft_area_ha * crop_share  # ha

        # 4. Apply FAOSTAT scaling per country
        if country_mask is not None:
            unique_codes = np.unique(country_mask[~np.isnan(country_mask)]).astype(int)
            for code in unique_codes:
                factor = change_dict.get((crop_name, code, int(year)), None)
                if factor is not None:
                    mask = (country_mask == code)
                    ha_raw[mask] *= factor
            # For countries with no factor, keep the raw LUH2×share value (factor=1 implicit)

        # 5. Irrigation split
        if irrig_fracs is not None:
            irrig_frac_025 = irrig_fracs[ti]
            irrig_frac_005 = regrid_luh2_to_cropgrids(
                irrig_frac_025, luh2['lat'], luh2['lon'], cg_lat, cg_lon
            )
            irrig_frac_005 = np.clip(irrig_frac_005, 0, 1)
            ha_irrig[ti] = ha_raw * irrig_frac_005
            ha_rain[ti] = ha_raw * (1 - irrig_frac_005)
        else:
            ha_rain[ti] = ha_raw

        ha_total[ti] = ha_raw

    return {
        'ha_total': ha_total,
        'ha_irrigated': ha_irrig,
        'ha_rainfed': ha_rain,
    }


def save_crop_netcdf(crop_name, result, years, cg_lat, cg_lon, output_dir):
    """Save per-crop results as NetCDF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            'harvested_area': (['time', 'lat', 'lon'], result['ha_total']),
            'harvested_area_irrigated': (['time', 'lat', 'lon'], result['ha_irrigated']),
            'harvested_area_rainfed': (['time', 'lat', 'lon'], result['ha_rainfed']),
        },
        coords={
            'time': years,
            'lat': cg_lat,
            'lon': cg_lon,
        },
        attrs={
            'title': f'Gridded harvested area for {crop_name}',
            'source': 'LUH2-GCB2025 + CROPGRIDS v1.08 + FAOSTAT',
            'resolution': '0.05 degree (~5.6 km)',
            'units': 'hectares (ha)',
            'cft': CFT_MAP.get(crop_name, 'unknown'),
            'created_by': 'run_pipeline.py',
        }
    )

    ds['harvested_area'].attrs = {'units': 'ha', 'long_name': f'{crop_name} total harvested area'}
    ds['harvested_area_irrigated'].attrs = {'units': 'ha', 'long_name': f'{crop_name} irrigated harvested area'}
    ds['harvested_area_rainfed'].attrs = {'units': 'ha', 'long_name': f'{crop_name} rainfed harvested area'}

    outfile = output_dir / f"{crop_name}.nc"
    encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                for var in ['harvested_area', 'harvested_area_irrigated', 'harvested_area_rainfed']}
    ds.to_netcdf(outfile, encoding=encoding)
    ds.close()

    return outfile


def main():
    parser = argparse.ArgumentParser(description="Gridded Irrigated/Rainfed Harvested Area Pipeline")
    parser.add_argument("--luh2_dir", required=True, help="Directory containing LUH2 states4.nc, management4.nc, staticData_quarterdeg.nc")
    parser.add_argument("--cropgrids_dir", required=True, help="Directory containing CROPGRIDSv1.08_*.nc files")
    parser.add_argument("--faostat_change", required=True, help="Path to FAOSTAT_change_2000-2023.csv")
    parser.add_argument("--faostat_change_2024", default=None, help="Path to FAOSTAT_change_2024.csv (optional)")
    parser.add_argument("--output_dir", default="./output", help="Output directory for NetCDF files")
    parser.add_argument("--crops", nargs="*", default=None, help="Subset of crops to process (default: all)")
    parser.add_argument("--country_mask_tif", default=None, help="Path to country_raster_CG.tif (colleague's raster). If not provided, uses Countries_2018.nc")
    args = parser.parse_args()

    print("=" * 70)
    print("  Gridded Irrigated/Rainfed Harvested Area Pipeline")
    print("=" * 70)

    print("\n[Step 1/6] Loading LUH2 data...")
    luh2 = load_luh2(args.luh2_dir)

    print("\n[Step 2/6] Loading CROPGRIDS data...")
    crop_ha, country_mask_nc, cg_lat, cg_lon = load_cropgrids(
        args.cropgrids_dir, 
        crops_to_process=args.crops
    )

    # Use colleague's country raster if provided (it matches their FAOSTAT codes)
    if args.country_mask_tif is not None:
        try:
            import rasterio
            with rasterio.open(args.country_mask_tif) as src:
                country_mask = src.read(1).astype(np.float32)
            print(f"  Using colleague's country raster: {args.country_mask_tif}")
            print(f"  Shape: {country_mask.shape}")
        except ImportError:
            print("  WARNING: rasterio not available, falling back to Countries_2018.nc")
            country_mask = country_mask_nc
    else:
        country_mask = country_mask_nc

    print("\n[Step 3/6] Computing within-CFT crop shares...")
    crop_shares = compute_crop_shares(crop_ha)

    print("\n[Step 4/6] Loading FAOSTAT change factors...")
    change_dict, fao_country_codes = load_faostat_change(
        args.faostat_change, args.faostat_change_2024
    )

    print("\n  Computing cell areas at CROPGRIDS resolution...")
    cg_carea_ha = compute_cropgrids_cell_area(cg_lat, cg_lon)
    print(f"  Cell area range: {cg_carea_ha.min():.1f} - {cg_carea_ha.max():.1f} ha")

    print("\n[Step 5/6] Processing crops...")
    crops_to_run = sorted(crop_shares.keys())
    print(f"  Will process {len(crops_to_run)} crops")

    output_dir = Path(args.output_dir)
    results_summary = []

    for idx, crop_name in enumerate(crops_to_run):
        print(f"\n  [{idx+1}/{len(crops_to_run)}] {crop_name} (CFT: {CFT_MAP[crop_name]})")

        result = process_crop(
            crop_name=crop_name,
            crop_share=crop_shares[crop_name],
            luh2=luh2,
            cg_lat=cg_lat,
            cg_lon=cg_lon,
            cg_carea_ha=cg_carea_ha,
            change_dict=change_dict,
            country_mask=country_mask,
        )

        if result is None:
            continue

        outfile = save_crop_netcdf(crop_name, result, luh2['years'], cg_lat, cg_lon, output_dir)

        total_ha = result['ha_total'].sum(axis=(1, 2))  # per year
        irrig_ha = result['ha_irrigated'].sum(axis=(1, 2))
        irrig_pct = np.where(total_ha > 0, irrig_ha / total_ha * 100, 0)

        results_summary.append({
            'crop': crop_name,
            'cft': CFT_MAP[crop_name],
            'mean_total_ha_M': total_ha.mean() / 1e6,
            'mean_irrig_pct': irrig_pct.mean(),
            'file': str(outfile),
        })

        print(f"    Total HA: {total_ha.mean()/1e6:.2f} M ha (mean across years)")
        print(f"    Irrigated: {irrig_pct.mean():.1f}%")
        print(f"    Saved: {outfile}")

        del result

    print("\n[Step 6/6] Saving summary...")
    summary_df = pd.DataFrame(results_summary)
    summary_file = output_dir / "pipeline_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"  Summary: {summary_file}")

    print("\n" + "=" * 70)
    print(f"  DONE! Processed {len(results_summary)} crops")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
