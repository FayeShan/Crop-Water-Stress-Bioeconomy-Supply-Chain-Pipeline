"""Aggregate the GLORIA MRIO footprint results into a flat dashboard CSV.

Reads the per-year ``target.zarr`` arrays produced by ``run.py`` and an
``aggregation_guide.xlsx`` (sheets ``Sector_Aggregation`` and
``Region_Aggregation``), and collapses the six-dimensional impact array
into a tidy table with columns:
    crop, target_sector, production_country, consumption_country,
    year, impact_type (direct/indirect/total), value
Per-year CSVs are written incrementally and combined into
``aggregated_all_years.csv`` (the supply-chain dashboard dataset).
"""

import argparse
import gc
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import zarr
from tqdm import tqdm

DEFAULT_PATTERN = "Crops_FoodProcessing_AnimalRaising_Textile_BiomassProducts_BioChemicals_Plastics_{year}_ind0"


class CropWaterStressAggregator:

    def __init__(self, results_base_path: str, aggregation_guide_path: str,
                 result_pattern: str = DEFAULT_PATTERN):
        self.base_path = Path(results_base_path)
        self.pattern = result_pattern
        self._load_aggregation_guide(aggregation_guide_path)

    def _load_aggregation_guide(self, path: str):
        print(f"Loading aggregation guide from: {path}")
        sector_df = pd.read_excel(path, sheet_name="Sector_Aggregation")

        crop_mapping = sector_df[sector_df["crop_group"].notna()][["sector_idx", "crop_group"]]
        self.crop_sector_indices = crop_mapping["sector_idx"].tolist()
        self.crop_idx_to_group = dict(zip(crop_mapping["sector_idx"], crop_mapping["crop_group"]))
        self.crop_groups = sorted(sector_df["crop_group"].dropna().unique().tolist())

        target_mapping = sector_df[sector_df["target_group"].notna()][["sector_idx", "target_group"]]
        self.target_sector_indices = target_mapping["sector_idx"].tolist()
        self.target_idx_to_group = dict(zip(target_mapping["sector_idx"], target_mapping["target_group"]))
        self.target_groups = sorted(sector_df["target_group"].dropna().unique().tolist())

        region_df = pd.read_excel(path, sheet_name="Region_Aggregation")
        self.region_idx_to_group = dict(zip(region_df["region_idx"], region_df["region_group"]))
        self.region_groups = sorted(region_df["region_group"].unique().tolist())

        print(f"  Crop groups ({len(self.crop_groups)}): {self.crop_groups}")
        print(f"  Target groups ({len(self.target_groups)}): {self.target_groups}")
        print(f"  Region groups ({len(self.region_groups)}): {self.region_groups}")

    def _get_result_path(self, year: int) -> Path:
        return self.base_path / self.pattern.format(year=year)

    def _check_year_exists(self, year: int) -> bool:
        return (self._get_result_path(year) / "target.zarr").exists()

    def _load_metadata(self, year: int) -> dict:
        with open(self._get_result_path(year) / "metadata.json", "r") as f:
            return json.load(f)

    def aggregate_year(self, year: int, chunk_size: int = 5) -> pd.DataFrame:
        if not self._check_year_exists(year):
            print(f"  Year {year}: no zarr data found, skipping...")
            return pd.DataFrame()

        z_target = zarr.open(self._get_result_path(year) / "target.zarr", mode="r")
        n_preg, n_psec, n_tsec, n_fssec, n_fdreg, n_slot = z_target.shape
        tsec_to_original = self._load_metadata(year)["target_sector_indices"]

        results = {}
        for fd_start in tqdm(range(0, n_fdreg, chunk_size), desc=f"Year {year}", leave=False):
            fd_end = min(fd_start + chunk_size, n_fdreg)
            chunk_sum = z_target[:, :, :, :, fd_start:fd_end, :].sum(axis=3)

            for preg_idx in range(n_preg):
                preg_group = self.region_idx_to_group.get(preg_idx, "Unknown")
                for psec_idx in self.crop_sector_indices:
                    if psec_idx >= n_psec:
                        continue
                    crop_group = self.crop_idx_to_group.get(psec_idx, "Unknown")
                    for tsec_idx in range(n_tsec):
                        target_group = self.target_idx_to_group.get(tsec_to_original[tsec_idx])
                        if target_group is None:
                            continue
                        for i, fdreg_idx in enumerate(range(fd_start, fd_end)):
                            fdreg_group = self.region_idx_to_group.get(fdreg_idx, "Unknown")
                            val_direct = float(chunk_sum[preg_idx, psec_idx, tsec_idx, i, 0])
                            val_indirect = float(chunk_sum[preg_idx, psec_idx, tsec_idx, i, 1])
                            if abs(val_direct) < 1e-10 and abs(val_indirect) < 1e-10:
                                continue
                            key_base = (crop_group, target_group, preg_group, fdreg_group)
                            results[key_base + ("direct",)] = results.get(key_base + ("direct",), 0.0) + val_direct
                            results[key_base + ("indirect",)] = results.get(key_base + ("indirect",), 0.0) + val_indirect
                            results[key_base + ("total",)] = results.get(key_base + ("total",), 0.0) + val_direct + val_indirect
            del chunk_sum
            gc.collect()

        rows = [{
            "crop": k[0], "target_sector": k[1], "production_country": k[2],
            "consumption_country": k[3], "impact_type": k[4], "value": v,
        } for k, v in results.items()]
        df = pd.DataFrame(rows)
        df["year"] = year
        return df[["crop", "target_sector", "production_country", "consumption_country",
                   "year", "impact_type", "value"]]

    def aggregate_years(self, years: List[int], chunk_size: int = 5,
                        output_path: Optional[str] = None, save_yearly: bool = True) -> pd.DataFrame:
        all_dfs = []
        output_dir = Path(output_path) if output_path else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        for year in tqdm(years, desc="Processing years"):
            if output_dir:
                yearly_path = output_dir / f"aggregated_{year}.csv"
                if yearly_path.exists():
                    print(f"  Year {year}: already exists, loading from CSV...")
                    all_dfs.append(pd.read_csv(yearly_path))
                    continue

            df = self.aggregate_year(year, chunk_size)
            if len(df) == 0:
                continue
            if save_yearly and output_dir:
                df.to_csv(output_dir / f"aggregated_{year}.csv", index=False)
                print(f"  Saved: aggregated_{year}.csv ({len(df):,} rows)")
            all_dfs.append(df)
            gc.collect()

        if not all_dfs:
            print("No data found!")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        if output_dir:
            combined.to_csv(output_dir / "aggregated_all_years.csv", index=False)
            print(f"\nSaved combined: {output_dir / 'aggregated_all_years.csv'} ({len(combined):,} rows)")
        return combined


def main():
    parser = argparse.ArgumentParser(description="Aggregate GLORIA MRIO results into the dashboard CSV.")
    parser.add_argument("--results", default="./output",
                        help="Directory with run.py result folders (default: ./output)")
    parser.add_argument("--guide", default="./aggregation_guide.xlsx",
                        help="Aggregation guide Excel (default: ./aggregation_guide.xlsx)")
    parser.add_argument("--output", default="./dashboard_data",
                        help="Output directory (default: ./dashboard_data)")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(1995, 2025)))
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help="Result-folder name pattern (must match the run's target set and order)")
    args = parser.parse_args()

    aggregator = CropWaterStressAggregator(
        results_base_path=args.results,
        aggregation_guide_path=args.guide,
        result_pattern=args.pattern,
    )
    aggregator.aggregate_years(args.years, chunk_size=args.chunk_size, output_path=args.output)


if __name__ == "__main__":
    main()
