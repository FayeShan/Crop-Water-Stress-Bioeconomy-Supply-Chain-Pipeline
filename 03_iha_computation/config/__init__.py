"""Configuration loader for the iHA pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    base_dir: str
    luh2_dir: str
    cropgrids_dir: str
    faostat_change: str
    faostat_change_2024: str


@dataclass
class OutputConfig:
    pipeline_dir: str


@dataclass
class AggregateConfig:
    output_dir: str
    categories: List[str]


@dataclass
class CropCalendarConfig:
    dir: str


@dataclass
class MonthlyConfig:
    output_dir: str
    acea_crops: List[str]


@dataclass
class ValidationConfig:
    spam_dir: str
    output_dir: str


@dataclass
class TargetGridConfig:
    nlat: int
    nlon: int


@dataclass
class PipelineConfig:
    log_dir: str
    log_level: str
    data: DataConfig
    output: OutputConfig
    aggregate: AggregateConfig
    crop_calendar: CropCalendarConfig
    monthly: MonthlyConfig
    validation: ValidationConfig
    target_grid: TargetGridConfig


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return PipelineConfig(
        log_dir=cfg["general"]["log_dir"],
        log_level=cfg["general"]["log_level"],
        data=DataConfig(**cfg["data"]),
        output=OutputConfig(**cfg["output"]),
        aggregate=AggregateConfig(**cfg["aggregate"]),
        crop_calendar=CropCalendarConfig(**cfg["crop_calendar"]),
        monthly=MonthlyConfig(**cfg["monthly"]),
        validation=ValidationConfig(**cfg["validation"]),
        target_grid=TargetGridConfig(**cfg["target_grid"]),
    )


def create_directories(config: PipelineConfig) -> None:
    import os
    for d in [
        config.log_dir,
        config.data.base_dir,
        config.output.pipeline_dir,
        config.aggregate.output_dir,
        config.monthly.output_dir,
        config.validation.output_dir,
    ]:
        os.makedirs(d, exist_ok=True)
