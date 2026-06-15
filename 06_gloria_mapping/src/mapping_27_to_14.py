"""Apply crop_27 -> sector_14 mapping to a complete 164 regions x 14 sectors x N years water-stress table (0-filled, no NaN).

24 direct crops map 1:1 (ri1 and ri2 both -> "Growing rice", summed); the 3
split crops (oac, pdc, pec) are allocated across sectors using rolling
weights for historical years and recent-period weights for forecast years.
"""

import logging
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import openpyxl
import pandas as pd


DEFAULT_SPLIT_CROPS = ("oac", "pdc", "pec")

# 14 GLORIA crop sectors (drops "Seeds and plant propagation" — no CWS coverage)
SECTORS_14 = [
    "Growing wheat",
    "Growing maize",
    "Growing cereals n.e.c",
    "Growing leguminous crops and oil seeds",
    "Growing rice",
    "Growing vegetables, roots, tubers",
    "Growing sugar beet and cane",
    "Growing tobacco",
    "Growing fibre crops",
    "Growing crops n.e.c.",
    "Growing grapes",
    "Growing fruits and nuts",
    "Growing beverage crops (coffee, tea etc)",
    "Growing spices, aromatic, drug and pharmaceutical crops",
]

# Direct 1:1 mapping for the 24 non-split crops
DIRECT_MAP = {
    "wh":  "Growing wheat",
    "ri1": "Growing rice",
    "ri2": "Growing rice",
    "mai": "Growing maize",
    "sor": "Growing cereals n.e.c",
    "mil": "Growing cereals n.e.c",
    "bar": "Growing cereals n.e.c",
    "soy": "Growing leguminous crops and oil seeds",
    "nut": "Growing leguminous crops and oil seeds",
    "sun": "Growing leguminous crops and oil seeds",
    "rap": "Growing leguminous crops and oil seeds",
    "bea": "Growing leguminous crops and oil seeds",
    "ckp": "Growing leguminous crops and oil seeds",
    "cwp": "Growing leguminous crops and oil seeds",
    "pot": "Growing vegetables, roots, tubers",
    "cas": "Growing vegetables, roots, tubers",
    "vgt": "Growing vegetables, roots, tubers",
    "sgc": "Growing sugar beet and cane",
    "sgb": "Growing sugar beet and cane",
    "cot": "Growing fibre crops",
    "cof": "Growing beverage crops (coffee, tea etc)",
    "coc": "Growing beverage crops (coffee, tea etc)",
    "plm": "Growing fruits and nuts",
    "aff": "Growing crops n.e.c.",
}


def _load_gloria_regions(readme_path: str, n_regions: int = 164) -> List[str]:
    wb = openpyxl.load_workbook(readme_path, data_only=True)
    ws = wb["Regions"]
    regions = []
    for row in ws.iter_rows(min_row=2, max_row=n_regions + 1, values_only=True):
        regions.append(row[1])  # acronym column
    wb.close()
    return regions


def _map_countries(crops_27: pd.DataFrame, country_27: pd.DataFrame,
                   logger: logging.Logger) -> pd.DataFrame:
    df = crops_27.merge(country_27, left_on="country", right_on="country_27", how="left")
    n_unmapped = df["country_14_code"].isna().sum()
    if n_unmapped > 0:
        unmapped = df[df["country_14_code"].isna()]["country"].unique()
        logger.warning(f"  {n_unmapped} rows with unmapped countries; "
                       f"first few: {list(unmapped)[:10]}")
    return df.dropna(subset=["country_14_code"])


def _process_direct(df: pd.DataFrame, split_crops, logger) -> pd.DataFrame:
    direct = df[~df["crop"].isin(split_crops)].copy()
    direct["sector_14"] = direct["crop"].map(DIRECT_MAP)
    n_unmapped = direct["sector_14"].isna().sum()
    if n_unmapped > 0:
        logger.warning(f"  {n_unmapped} rows with unmapped direct crops: "
                       f"{direct[direct['sector_14'].isna()]['crop'].unique()}")
    direct = direct.dropna(subset=["sector_14"])
    direct["allocated"] = direct["value"]
    result = (direct.groupby(["year", "country_14_code", "sector_14"])["allocated"]
              .sum().reset_index())
    logger.info(f"  direct crops: {len(direct)} input -> {len(result)} aggregated rows")
    return result


def _process_split(df, rolling_w, recent_w, split_crops, logger) -> pd.DataFrame:
    split = df[df["crop"].isin(split_crops)].copy()
    if len(split) == 0:
        return pd.DataFrame(columns=["year", "country_14_code", "sector_14", "allocated"])

    historical_years = set(rolling_w["year"].unique())
    hist = split[split["year"].isin(historical_years)].copy()
    fore = split[~split["year"].isin(historical_years)].copy()

    results = []
    if len(hist) > 0:
        merged = hist.merge(
            rolling_w,
            left_on=["country_14_code", "crop", "year"],
            right_on=["country_14_code", "crop_27", "year"],
            how="left",
        )
        merged["weight"] = merged["weight"].fillna(0.0)
        merged["allocated"] = merged["value"] * merged["weight"]
        agg = (merged.groupby(["year", "country_14_code", "sector_14"])["allocated"]
               .sum().reset_index())
        agg = agg.dropna(subset=["sector_14"])
        results.append(agg)
        logger.info(f"  split historical: {len(hist)} input -> {len(agg)} output rows")

    if len(fore) > 0:
        merged = fore.merge(
            recent_w,
            left_on=["country_14_code", "crop"],
            right_on=["country_14_code", "crop_27"],
            how="left",
        )
        merged["weight"] = merged["weight"].fillna(0.0)
        merged["allocated"] = merged["value"] * merged["weight"]
        agg = (merged.groupby(["year", "country_14_code", "sector_14"])["allocated"]
               .sum().reset_index())
        agg = agg.dropna(subset=["sector_14"])
        results.append(agg)
        logger.info(f"  split forecast:   {len(fore)} input -> {len(agg)} output rows")

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=["year", "country_14_code", "sector_14", "allocated"])


def _build_complete_table(direct_result, split_result, regions_164, years, logger):
    direct_result = direct_result.rename(columns={"allocated": "water_stress"})
    split_result = split_result.rename(columns={"allocated": "water_stress"})
    combined = pd.concat([direct_result, split_result], ignore_index=True)
    combined = (combined.groupby(["year", "country_14_code", "sector_14"])["water_stress"]
                .sum().reset_index())
    logger.info(f"  combined: {len(combined)} rows with non-zero values")

    skeleton = pd.DataFrame(
        list(product(years, regions_164, SECTORS_14)),
        columns=["year", "country_14_code", "sector_14"],
    )
    logger.info(f"  skeleton: {len(skeleton)} rows "
                f"({len(years)} x {len(regions_164)} x {len(SECTORS_14)})")

    result = skeleton.merge(combined, on=["year", "country_14_code", "sector_14"], how="left")
    result["water_stress"] = result["water_stress"].fillna(0.0)

    result["country_14_code"] = pd.Categorical(
        result["country_14_code"], categories=regions_164, ordered=True)
    result["sector_14"] = pd.Categorical(
        result["sector_14"], categories=SECTORS_14, ordered=True)
    return result.sort_values(["year", "country_14_code", "sector_14"]).reset_index(drop=True)


def _validate(crops_27_raw, result, country_27, logger):
    raw = crops_27_raw.merge(country_27, left_on="country", right_on="country_27", how="inner")
    original_total = raw["value"].sum()
    result_total = result["water_stress"].sum()
    diff_pct = abs(result_total - original_total) / original_total * 100 if original_total > 0 else 0
    logger.info(f"  [1] total water-stress preservation: "
                f"orig={original_total:,.2f}, result={result_total:,.2f}, "
                f"diff={diff_pct:.4f}%")

    n_nan = result["water_stress"].isna().sum()
    logger.info(f"  [2] NaN count: {n_nan}")
    assert n_nan == 0

    logger.info(f"  [3] dims: years={result['year'].nunique()}, "
                f"regions={result['country_14_code'].nunique()}, "
                f"sectors={result['sector_14'].nunique()}, rows={len(result)}")


def apply_mapping_27_to_14(
    crops_27_csv: str,
    country_mapping_csv: str,
    rolling_csv: str,
    recent_csv: str,
    gloria_readme: str,
    output_csv: str,
    n_regions: int = 164,
    split_crops=DEFAULT_SPLIT_CROPS,
    logger: Optional[logging.Logger] = None,
) -> str:
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("  mapping crops_27 -> 164 regions x 14 sectors")
    logger.info("=" * 70)

    logger.info("[1] loading data...")
    crops_27 = pd.read_csv(crops_27_csv)
    country_27 = pd.read_csv(country_mapping_csv)
    rolling_w = pd.read_csv(rolling_csv)
    recent_w = pd.read_csv(recent_csv)
    regions_164 = _load_gloria_regions(gloria_readme, n_regions=n_regions)
    logger.info(f"  crops_27:        {len(crops_27)} rows, "
                f"years {crops_27['year'].min()}-{crops_27['year'].max()}")
    logger.info(f"  rolling weights: {len(rolling_w)}  recent: {len(recent_w)}")
    logger.info(f"  GLORIA regions:  {len(regions_164)}")

    logger.info("[2] mapping countries to GLORIA regions...")
    df = _map_countries(crops_27, country_27, logger)
    years = sorted(df["year"].unique())
    logger.info(f"  years in data: {int(min(years))}-{int(max(years))} ({len(years)} years)")

    logger.info("[3] processing direct-mapping crops (24 crops)...")
    direct_result = _process_direct(df, split_crops, logger)

    logger.info("[4] processing split crops (oac, pdc, pec)...")
    split_result = _process_split(df, rolling_w, recent_w, split_crops, logger)

    logger.info("[5] building complete 164 x 14 x years table...")
    result = _build_complete_table(direct_result, split_result, regions_164, years, logger)

    logger.info("[6] validation...")
    _validate(crops_27, result, country_27, logger)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    logger.info(f"  saved: {output_csv} ({len(result)} rows)")
    return str(output_csv)
