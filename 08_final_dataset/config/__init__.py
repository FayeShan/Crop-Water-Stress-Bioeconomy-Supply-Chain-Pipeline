"""Configuration loader for the final-dataset packaging pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class InputConfig:
    predictions_dir: str
    reference_grid_nc: str
    shapefile: str


@dataclass
class OutputConfig:
    netcdf_monthly_dir: str
    netcdf_yearly_dir: str
    geotiff_monthly_dir: str
    geotiff_yearly_dir: str
    csv_dir: str


@dataclass
class YearsConfig:
    start: int
    end: int


@dataclass
class NetcdfConfig:
    compression_level: int
    recompress_skip_threshold_mb: int


@dataclass
class PipelineConfig:
    log_dir: str
    log_level: str
    input: InputConfig
    output: OutputConfig
    years: YearsConfig
    netcdf: NetcdfConfig


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return PipelineConfig(
        log_dir=cfg["general"]["log_dir"],
        log_level=cfg["general"]["log_level"],
        input=InputConfig(**cfg["input"]),
        output=OutputConfig(**cfg["output"]),
        years=YearsConfig(**cfg["years"]),
        netcdf=NetcdfConfig(**cfg["netcdf"]),
    )


def create_directories(config: PipelineConfig) -> None:
    for d in [
        config.log_dir,
        config.output.netcdf_monthly_dir,
        config.output.netcdf_yearly_dir,
        config.output.geotiff_monthly_dir,
        config.output.geotiff_yearly_dir,
        config.output.csv_dir,
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)
