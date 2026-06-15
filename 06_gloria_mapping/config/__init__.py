"""Configuration loader for the gloria_mapping pipeline."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class HarmPredictionsConfig:
    input_dir: str
    country_csv_dir: str
    shapefile: str
    y_pred_column: str


@dataclass
class HistoricalConfig:
    crops_27_csv: str
    crops_175_csv: str
    years: List[int]


@dataclass
class CombinedConfig:
    output_csv: str
    pred_years: List[int]


@dataclass
class WeightsConfig:
    rolling_csv: str
    recent_csv: str
    rolling_window: int
    recent_period: List[int]
    split_crops: List[str]


@dataclass
class ReferenceConfig:
    crop_mapping_175_to_27_to_14: str
    country_mapping_175_to_14: str
    country_mapping_27_to_14: str
    aggregate_region_mapping: str
    gloria_readme: str


@dataclass
class MappingConfig:
    long_csv: str
    n_regions: int
    n_sectors_our: int


@dataclass
class ValidationConfig:
    metrics_csv: str
    composition_csv: str
    start_year: int
    end_year: int


@dataclass
class GloriaRawConfig:
    mrio_zips_dir: str
    sat_zips_dir: str


@dataclass
class GloriaProcessedConfig:
    mrio_unzipped_dir: str
    mrio_zarr_dir: str
    satellite_dir: str


@dataclass
class SatelliteConfig:
    our_dir: str
    hybrid_dir: str
    ag_row_index: int
    non_ag_row_index: int
    n_sectors_gloria: int
    include_non_ag: bool
    unit_scale: float


@dataclass
class DateRangeConfig:
    historical_start: int
    historical_end: int
    predict_start: int
    predict_end: int
    gloria_start: int
    gloria_end: int


@dataclass
class CropsConfig:
    all_27: List[str]


@dataclass
class PipelineConfig:
    log_dir: str
    log_level: str
    harm_predictions: HarmPredictionsConfig
    historical: HistoricalConfig
    combined: CombinedConfig
    weights: WeightsConfig
    reference: ReferenceConfig
    mapping: MappingConfig
    validation: ValidationConfig
    gloria_raw: GloriaRawConfig
    gloria_processed: GloriaProcessedConfig
    satellite: SatelliteConfig
    date_range: DateRangeConfig
    crops: CropsConfig


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return PipelineConfig(
        log_dir=cfg["general"]["log_dir"],
        log_level=cfg["general"]["log_level"],
        harm_predictions=HarmPredictionsConfig(**cfg["harm_predictions"]),
        historical=HistoricalConfig(**cfg["historical"]),
        combined=CombinedConfig(**cfg["combined"]),
        weights=WeightsConfig(**cfg["weights"]),
        reference=ReferenceConfig(**cfg["reference"]),
        mapping=MappingConfig(**cfg["mapping"]),
        validation=ValidationConfig(**cfg["validation"]),
        gloria_raw=GloriaRawConfig(**cfg["gloria_raw"]),
        gloria_processed=GloriaProcessedConfig(**cfg["gloria_processed"]),
        satellite=SatelliteConfig(**cfg["satellite"]),
        date_range=DateRangeConfig(**cfg["date_range"]),
        crops=CropsConfig(**cfg["crops"]),
    )


def create_directories(config: PipelineConfig) -> None:
    for d in [
        config.log_dir,
        config.harm_predictions.input_dir,
        config.harm_predictions.country_csv_dir,
        config.weights.rolling_csv, config.weights.recent_csv,  # parents handled below
        config.gloria_raw.mrio_zips_dir,
        config.gloria_raw.sat_zips_dir,
        config.gloria_processed.mrio_unzipped_dir,
        config.gloria_processed.mrio_zarr_dir,
        config.gloria_processed.satellite_dir,
        config.satellite.our_dir,
        config.satellite.hybrid_dir,
    ]:
        p = Path(d)
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
