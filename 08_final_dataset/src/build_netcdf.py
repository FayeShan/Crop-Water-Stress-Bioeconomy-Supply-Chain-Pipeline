"""Build monthly + yearly NetCDF from the flat per-crop HARM prediction CSVs."""

import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.constants import CROPS, CROP_NAMES


def get_reference_grid(reference_grid_nc):
    with xr.open_dataset(reference_grid_nc) as ds:
        return ds["lat"].values, ds["lon"].values


def load_all_predictions(pred_dir, crops, logger):
    pred_dir = Path(pred_dir)
    all_pred = {}
    for crop in tqdm(crops, desc="Loading predictions"):
        f = pred_dir / f"{crop}.csv"
        if not f.exists():
            logger.warning(f"  {crop}: {f} not found")
            continue
        df = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["time"])
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        all_pred[crop] = df
    logger.info(f"  Loaded {len(all_pred)}/{len(crops)} crops")
    return all_pred


def create_year_nc(year, all_pred, lat, lon, out_dir):
    """Write one multi-crop monthly NetCDF (time=12, lat, lon) for `year`."""
    times = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    lat_to_idx = {round(v, 4): i for i, v in enumerate(lat)}
    lon_to_idx = {round(v, 4): i for i, v in enumerate(lon)}

    data_vars = {}
    for crop in CROPS:
        grid = np.full((12, len(lat), len(lon)), np.nan, dtype=np.float32)
        if crop in all_pred:
            df_year = all_pred[crop][all_pred[crop]["year"] == year]
            for _, row in df_year.iterrows():
                li = lat_to_idx.get(round(row["lat"], 4))
                lo = lon_to_idx.get(round(row["lon"], 4))
                if li is not None and lo is not None:
                    grid[int(row["month"]) - 1, li, lo] = row["y_pred"]
        data_vars[crop] = (["time", "lat", "lon"], grid, {
            "long_name": f"Water Stress - {CROP_NAMES.get(crop, crop)}",
            "units": "1", "valid_min": 0,
        })

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={
            "title": f"Crop Water Stress {year}",
            "institution": "Technical University of Munich",
            "source": "HARM model predictions",
            "created": datetime.now().isoformat(),
            "conventions": "CF-1.8",
            "spatial_resolution": "5 arcminutes (0.0833 degrees, ~9 km at equator)",
            "temporal_resolution": "monthly",
        },
    )
    out_path = Path(out_dir) / f"water_stress_{year}.nc"
    ds.to_netcdf(out_path, engine="scipy")
    ds.close()
    return out_path


def build_yearly_aggregates(monthly_dir, yearly_dir, lat, lon, logger):
    """Regenerate yearly_by_crop + yearly_total from whatever monthly NCs exist."""
    monthly_dir = Path(monthly_dir)
    years = sorted(int(p.stem.split("_")[-1]) for p in monthly_dir.glob("water_stress_*.nc"))
    if not years:
        logger.warning(f"  No monthly NetCDF found in {monthly_dir}")
        return

    n_years = len(years)
    yearly_crop = {c: np.full((n_years, len(lat), len(lon)), np.nan, dtype=np.float32) for c in CROPS}
    yearly_total = np.full((n_years, len(lat), len(lon)), np.nan, dtype=np.float32)

    for i, year in enumerate(tqdm(years, desc="Aggregating yearly")):
        ds = xr.open_dataset(monthly_dir / f"water_stress_{year}.nc")
        total = np.zeros((len(lat), len(lon)), dtype=np.float64)
        has_data = np.zeros((len(lat), len(lon)), dtype=bool)
        for crop in CROPS:
            if crop in ds:
                annual = ds[crop].sum(dim="time", skipna=True).values
                valid = ~np.isnan(annual)
                yearly_crop[crop][i] = annual
                total[valid] += annual[valid]
                has_data |= valid
        total[~has_data] = np.nan
        yearly_total[i] = total.astype(np.float32)
        ds.close()

    yearly_dir = Path(yearly_dir)
    y0, y1 = years[0], years[-1]

    data_vars = {c: (["year", "lat", "lon"], yearly_crop[c], {
        "long_name": f"Yearly Water Stress - {CROP_NAMES.get(c, c)}", "units": "1",
    }) for c in CROPS}
    ds_crop = xr.Dataset(data_vars=data_vars, coords={"year": years, "lat": lat, "lon": lon},
                         attrs={"title": f"Yearly Crop Water Stress by Crop ({y0}-{y1})",
                                "created": datetime.now().isoformat()})
    ds_crop.to_netcdf(yearly_dir / f"yearly_by_crop_{y0}_{y1}.nc", engine="scipy")
    ds_crop.close()
    logger.info(f"  Saved yearly_by_crop_{y0}_{y1}.nc")

    ds_total = xr.Dataset(
        {"total_water_stress": (["year", "lat", "lon"], yearly_total, {
            "long_name": "Total Water Stress (all crops)", "units": "1"})},
        coords={"year": years, "lat": lat, "lon": lon},
        attrs={"title": f"Yearly Total Water Stress ({y0}-{y1})",
               "created": datetime.now().isoformat()})
    ds_total.to_netcdf(yearly_dir / f"yearly_total_{y0}_{y1}.nc", engine="scipy")
    ds_total.close()
    logger.info(f"  Saved yearly_total_{y0}_{y1}.nc")


def build_netcdf(pred_dir, reference_grid_nc, monthly_dir, yearly_dir,
                 year_start, year_end, logger):
    lat, lon = get_reference_grid(reference_grid_nc)
    logger.info(f"Grid: {len(lat)} x {len(lon)}")

    all_pred = load_all_predictions(pred_dir, CROPS, logger)
    years = list(range(year_start, year_end + 1))

    Path(monthly_dir).mkdir(parents=True, exist_ok=True)
    for year in tqdm(years, desc="Monthly NetCDF"):
        create_year_nc(year, all_pred, lat, lon, monthly_dir)
        gc.collect()
    del all_pred
    gc.collect()

    Path(yearly_dir).mkdir(parents=True, exist_ok=True)
    build_yearly_aggregates(monthly_dir, yearly_dir, lat, lon, logger)
    logger.info("Done building NetCDF.")
