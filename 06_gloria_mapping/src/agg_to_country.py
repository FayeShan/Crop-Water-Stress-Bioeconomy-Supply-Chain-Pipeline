"""Aggregate per-crop HARM grid-level predictions ({pred_dir}/{crop}.csv: time, lat, lon, y_pred) to country level via a spatial join with the NaturalEarth admin-0 shapefile.

Output ``{out_dir}/{crop}_country.csv`` columns: iso3, country, time, year,
month, waterstress.
"""

import gc
import logging
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


def _build_spatial_lookup(
    pred_dir: Path, crops: List[str], shapefile: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Collect all unique (lat, lon) from the crop CSVs and spatially join
    with the country shapefile. Returns (lat, lon, country, iso3) DataFrame.
    """
    logger.info("  building spatial (lat, lon) -> country lookup ...")
    countries = gpd.read_file(shapefile)
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")

    all_coords = set()
    for crop in tqdm(crops, desc="scanning coords", leave=False):
        f = pred_dir / f"{crop}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["lat", "lon"])
        for t in df[["lat", "lon"]].drop_duplicates().itertuples(index=False):
            all_coords.add((t.lat, t.lon))

    coords_df = pd.DataFrame(list(all_coords), columns=["lat", "lon"])
    logger.info(f"  {len(coords_df):,} unique (lat, lon) points")

    geometry = [Point(xy) for xy in zip(coords_df["lon"], coords_df["lat"])]
    gdf = gpd.GeoDataFrame(coords_df, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(
        gdf, countries[["ADMIN", "ISO_A3", "geometry"]],
        how="left", predicate="within",
    )
    lookup = (joined.groupby(["lat", "lon"]).first()[["ADMIN", "ISO_A3"]]
              .reset_index())
    lookup = lookup.rename(columns={"ADMIN": "country", "ISO_A3": "iso3"})
    n_matched = lookup["country"].notna().sum()
    logger.info(f"  matched: {n_matched:,}/{len(lookup):,} "
                f"({n_matched/len(lookup)*100:.1f}%)")

    del gdf, joined, coords_df
    gc.collect()
    return lookup


def _aggregate_crop(
    crop: str, pred_dir: Path, lookup: pd.DataFrame,
    out_dir: Path, y_pred_col: str, logger: logging.Logger,
) -> Optional[str]:
    f = pred_dir / f"{crop}.csv"
    if not f.exists():
        logger.warning(f"  {crop}: {f.name} not found")
        return None

    df = pd.read_csv(f)
    df["time"] = pd.to_datetime(df["time"])
    df = df.merge(lookup, on=["lat", "lon"], how="left")
    df = df.dropna(subset=["country"])

    agg = (df.groupby(["iso3", "country", "time"])[y_pred_col]
           .sum().reset_index())
    agg = agg.rename(columns={y_pred_col: "waterstress"})
    agg["year"] = agg["time"].dt.year
    agg["month"] = agg["time"].dt.month
    agg = agg[["iso3", "country", "time", "year", "month", "waterstress"]]
    agg = agg.sort_values(["country", "time"]).reset_index(drop=True)

    out = out_dir / f"{crop}_country.csv"
    agg.to_csv(out, index=False)
    logger.info(f"  {crop}: {len(agg):,} country-month rows -> {out.name}")
    del df, agg
    gc.collect()
    return str(out)


def run_agg_to_country(
    crops: List[str],
    pred_dir: str,
    shapefile: str,
    out_dir: str,
    y_pred_column: str = "y_pred",
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    logger = logger or logging.getLogger(__name__)
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  Aggregating HARM grid-level predictions to country level")
    logger.info("=" * 70)
    logger.info(f"  pred_dir:  {pred_dir}")
    logger.info(f"  shapefile: {shapefile}")
    logger.info(f"  out_dir:   {out_dir}")
    logger.info(f"  crops:     {len(crops)}")

    lookup = _build_spatial_lookup(pred_dir, crops, Path(shapefile), logger)

    outputs = []
    for crop in tqdm(crops, desc="aggregating"):
        out = _aggregate_crop(crop, pred_dir, lookup, out_dir, y_pred_column, logger)
        if out:
            outputs.append(out)

    logger.info(f"\nSaved {len(outputs)} files to {out_dir}/")
    return outputs
