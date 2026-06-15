"""
Configuration management
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Union
from pathlib import Path


# GLORIA constants
N_REGIONS = 164
N_SECTORS = 120


@dataclass
class Config:
    """
    Configuration class for GLORIA MRIO analysis
    
    Attributes:
        gloria_mrio_path: Directory containing GLORIA MRIO zarr files
        satellite_path: Directory containing satellite account data
        output_path: Output directory for results
        years: List of years to process
        target_sectors: Target sector definitions {group_name: [sector_indices]}
        region_grouping: Region group definitions {group_name: [region_indices] or "all"}
        indicators: List of indicators to calculate
        dtype: Data type for calculations
        chunk_size: Zarr chunk size
    """
    gloria_mrio_path: str = ""
    satellite_path: str = ""
    output_path: str = ""

    years: List[int] = field(default_factory=lambda: list(range(1990, 2026)))

    # Target sectors use 0-based indices
    target_sectors: Dict[str, List[int]] = field(default_factory=lambda: {
        'Gold': [31, 79],
    })

    region_grouping: Dict[str, Union[str, List[int]]] = field(default_factory=lambda: {
        'Global': 'all',
    })

    indicators: List[Union[int, str]] = field(default_factory=lambda: [0])

    dtype: str = "float32"
    chunk_size: int = 50

    # GLORIA constants (do not modify)
    n_regions: int = N_REGIONS
    n_sectors: int = N_SECTORS
    
    def __post_init__(self):
        """Process 'all' and 'individual' in region_grouping"""
        if self.region_grouping == "individual":
            # Each region as its own group
            self.region_grouping = {f"R{i}": [i] for i in range(self.n_regions)}
        elif isinstance(self.region_grouping, dict):
            for name, value in self.region_grouping.items():
                if value == 'all':
                    self.region_grouping[name] = list(range(self.n_regions))
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file.

        Any top-level key starting with an underscore (e.g. ``_comments``,
        ``_description``) is treated as documentation metadata and
        stripped before constructing the dataclass.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Strip all underscore-prefixed meta keys
        data = {k: v for k, v in data.items() if not k.startswith('_')}

        return cls(**data)
    
    def to_json(self, path: str):
        """Save configuration to JSON file"""
        data = {
            'gloria_mrio_path': self.gloria_mrio_path,
            'satellite_path': self.satellite_path,
            'output_path': self.output_path,
            'years': self.years,
            'target_sectors': self.target_sectors,
            'region_grouping': self.region_grouping,
            'indicators': self.indicators,
            'dtype': self.dtype,
            'chunk_size': self.chunk_size,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def validate(self):
        """Validate configuration"""
        errors = []
        
        if not self.gloria_mrio_path:
            errors.append("gloria_mrio_path is required")
        
        if not self.satellite_path:
            errors.append("satellite_path is required")
        
        if not self.output_path:
            errors.append("output_path is required")
        
        if not self.years:
            errors.append("years cannot be empty")
        
        if not self.target_sectors:
            errors.append("target_sectors cannot be empty")
        
        for name, indices in self.target_sectors.items():
            for idx in indices:
                if idx < 0 or idx >= self.n_sectors:
                    errors.append(f"Invalid sector index {idx} in {name}")

        for name, indices in self.region_grouping.items():
            if isinstance(indices, list):
                for idx in indices:
                    if idx < 0 or idx >= self.n_regions:
                        errors.append(f"Invalid region index {idx} in {name}")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
