"""Orchestrates the full GLORIA MRIO analysis pipeline."""

import os
import gc
from typing import Dict, List, Optional

from .config import Config
from .index_builder import IndexBuilder
from .mrio_loader import MRIOLoader
from .satellite_loader import load_satellite_data
from .calculator import FootprintCalculator


class GloriaMRIO:
    """
    GLORIA MRIO analysis main class
    
    Usage:
        config = Config.from_json("configs/default.json")
        analyzer = GloriaMRIO(config)
        results = analyzer.run()
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.config.validate()
        self.idx = IndexBuilder(config)
    
    def run(self, target_names: Optional[List[str]] = None, 
            years: Optional[List[int]] = None,
            indicators: Optional[List[int]] = None) -> Dict:
        """
        Run full analysis
        
        Args:
            target_names: List of target sector group names, None for all defined in config
            years: List of years, None for config years
            indicators: List of indicators, None for config indicators
        
        Returns:
            Results dictionary {(year, indicator): result_dict}
        """
        if target_names is None:
            target_names = list(self.config.target_sectors.keys())
        
        if years is None:
            years = self.config.years
        
        if indicators is None:
            indicators = self.config.indicators
        
        print("=" * 60)
        print("GLORIA MRIO Analysis")
        print("=" * 60)
        print(f"Years: {years[0]} - {years[-1]} ({len(years)} years)")
        print(f"Target sectors: {target_names}")
        print(f"Indicators: {indicators}")
        print(f"Output: {self.config.output_path}")
        print("=" * 60)
        
        results = {}
        success_count = 0
        skip_count = 0
        error_count = 0
        
        for year in years:
            print(f"\n{'='*60}")
            print(f"Year: {year}")
            print("=" * 60)

            mrio_path = os.path.join(self.config.gloria_mrio_path, f"{year}.zarr")
            if not os.path.exists(mrio_path):
                print(f"  ⚠️ MRIO not found: {mrio_path}")
                skip_count += 1
                continue

            try:
                mrio = MRIOLoader(mrio_path, self.config)
                mrio.compute_A()
                mrio.compute_L()

                try:
                    satellite = load_satellite_data(
                        self.config.satellite_path, year,
                        self.config.n_regions, self.config.n_sectors
                    )
                except FileNotFoundError as e:
                    print(f"  ⚠️ {e}")
                    skip_count += 1
                    del mrio
                    gc.collect()
                    continue

                calc = FootprintCalculator(mrio, self.idx, satellite, self.config)

                for indicator in indicators:
                    if isinstance(indicator, str):
                        print(f"  ⚠️ String indicator not yet supported: {indicator}")
                        continue

                    target_str = "_".join(target_names)
                    output_dir = os.path.join(
                        self.config.output_path,
                        f"{target_str}_{year}_ind{indicator}"
                    )

                    if os.path.exists(os.path.join(output_dir, "metadata.json")):
                        print(f"  ⏭️ Already exists: {output_dir}")
                        skip_count += 1
                        continue
                    
                    result = calc.calculate(indicator, target_names, output_dir)
                    results[(year, indicator)] = result
                    
                    if result['error'] < 0.01:
                        print(f"  ✅ Success: error={result['error']:.4%}")
                        success_count += 1
                    else:
                        print(f"  ⚠️ Warning: error={result['error']:.4%}")
                        success_count += 1

                del mrio, calc
                gc.collect()

            except Exception as e:
                print(f"  ❌ Error: {e}")
                error_count += 1
                gc.collect()

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  ✅ Success: {success_count}")
        print(f"  ⏭️ Skipped: {skip_count}")
        print(f"  ❌ Errors:  {error_count}")
        print("=" * 60)
        
        return results
    
    def run_single(self, year: int, indicator: int, 
                   target_names: Optional[List[str]] = None) -> Dict:
        """
        Run single year single indicator analysis
        
        Args:
            year: Year
            indicator: Indicator index
            target_names: List of target sector group names
        
        Returns:
            Result dictionary
        """
        return self.run(
            target_names=target_names,
            years=[year],
            indicators=[indicator]
        ).get((year, indicator))
