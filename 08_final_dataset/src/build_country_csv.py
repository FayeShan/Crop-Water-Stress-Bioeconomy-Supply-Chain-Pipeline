"""Aggregate the monthly NetCDF to country-level CSVs (global totals + per-crop per-country)."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point
from tqdm import tqdm

from src.constants import CROPS


def _years_present(nc_dir):
    return sorted(int(p.stem.split("_")[-1]) for p in Path(nc_dir).glob("water_stress_*.nc"))


def build_country_lookup(nc_dir, years, shapefile, logger):
    """Build a {(lat, lon): country} lookup for the non-NaN grid cells only."""
    nc_dir = Path(nc_dir)
    sample = xr.open_dataset(nc_dir / f"water_stress_{years[0]}.nc")
    lat, lon = sample["lat"].values, sample["lon"].values
    sample.close()

    has_data = np.zeros((len(lat), len(lon)), dtype=bool)
    for year in years:
        ds = xr.open_dataset(nc_dir / f"water_stress_{year}.nc")
        for crop in CROPS:
            if crop in ds:
                has_data |= ~np.isnan(ds[crop].sum(dim="time", skipna=True).values)
        ds.close()

    idx = np.argwhere(has_data)
    pts_lat, pts_lon = lat[idx[:, 0]], lon[idx[:, 1]]
    logger.info(f"  Non-NaN grid cells: {len(pts_lat):,} / {len(lat) * len(lon):,}")

    countries = gpd.read_file(shapefile)
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame({"lat": pts_lat, "lon": pts_lon},
                           geometry=[Point(lo, la) for la, lo in zip(pts_lat, pts_lon)],
                           crs="EPSG:4326")
    joined = gpd.sjoin(gdf, countries[["ADMIN", "geometry"]], how="left", predicate="within")
    joined = joined.groupby(["lat", "lon"]).first()[["ADMIN"]].reset_index()
    n_matched = joined["ADMIN"].notna().sum()
    logger.info(f"  Matched to country: {n_matched:,} / {len(joined):,}")

    lookup = {(round(r.lat, 4), round(r.lon, 4)): r.ADMIN
              for r in joined.itertuples(index=False) if pd.notna(r.ADMIN)}
    del gdf, joined
    gc.collect()
    return lookup


def build_country_csv(nc_dir, csv_dir, shapefile, logger):
    nc_dir = Path(nc_dir)
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    years = _years_present(nc_dir)
    if not years:
        logger.warning(f"  No monthly NetCDF found in {nc_dir}")
        return

    lookup = build_country_lookup(nc_dir, years, shapefile, logger)

    global_rows = []
    country_rows = []
    for year in tqdm(years, desc="Country aggregation"):
        ds = xr.open_dataset(nc_dir / f"water_stress_{year}.nc")
        nc_lat, nc_lon = ds["lat"].values, ds["lon"].values

        year_total = 0.0
        for crop in CROPS:
            if crop not in ds:
                continue
            annual = ds[crop].sum(dim="time", skipna=True).values
            year_total += float(np.nansum(annual))
            country_totals = {}
            for li, lo in np.argwhere(~np.isnan(annual) & (annual != 0)):
                country = lookup.get((round(nc_lat[li], 4), round(nc_lon[lo], 4)))
                if country:
                    country_totals[country] = country_totals.get(country, 0.0) + float(annual[li, lo])
            for country, value in country_totals.items():
                country_rows.append({"year": year, "country": country, "crop": crop, "value": value})

        global_rows.append({"year": year, "total_water_stress": year_total, "source": "predicted"})
        ds.close()
        gc.collect()

    df_global = pd.DataFrame(global_rows).sort_values("year")
    df_global.to_csv(csv_dir / "global_yearly_totals.csv", index=False)
    logger.info(f"  Saved global_yearly_totals.csv ({len(df_global)} rows)")

    df_country = pd.DataFrame(country_rows).sort_values(["year", "country", "crop"])
    df_country.to_csv(csv_dir / "all_crops_country_yearly.csv", index=False)
    logger.info(f"  Saved all_crops_country_yearly.csv ({len(df_country):,} rows)")
