import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class GeneralConfig:
    log_dir: str
    log_level: str


@dataclass
class WFConfig:
    input_dir: str
    filename_regex: str
    variable: str


@dataclass
class AWAREConfig:
    weights_nc: str
    weights_variable: str
    excel: str
    placemarks_csv: str
    merged_csv: str


@dataclass
class OutputConfig:
    waterstress_dir: str
    compression_level: int


@dataclass
class DateRangeConfig:
    start_year: Optional[int]
    end_year: Optional[int]


@dataclass
class PipelineConfig:
    """Top-level config for the water stress computation pipeline."""
    log_dir: str
    log_level: str
    wf: WFConfig
    aware: AWAREConfig
    output: OutputConfig
    date_range: DateRangeConfig


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load YAML config and return a PipelineConfig dataclass."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return PipelineConfig(
        log_dir=cfg["general"]["log_dir"],
        log_level=cfg["general"]["log_level"],
        wf=WFConfig(
            input_dir=cfg["wf"]["input_dir"],
            filename_regex=cfg["wf"]["filename_regex"],
            variable=cfg["wf"]["variable"],
        ),
        aware=AWAREConfig(
            weights_nc=cfg["aware"]["weights_nc"],
            weights_variable=cfg["aware"]["weights_variable"],
            excel=cfg["aware"]["excel"],
            placemarks_csv=cfg["aware"]["placemarks_csv"],
            merged_csv=cfg["aware"]["merged_csv"],
        ),
        output=OutputConfig(
            waterstress_dir=cfg["output"]["waterstress_dir"],
            compression_level=cfg["output"].get("compression_level", 5),
        ),
        date_range=DateRangeConfig(
            start_year=cfg["date_range"].get("start_year"),
            end_year=cfg["date_range"].get("end_year"),
        ),
    )


def create_directories(config: PipelineConfig) -> None:
    """Create any missing output / log directories."""
    for d in [
        config.log_dir,
        config.wf.input_dir,
        config.output.waterstress_dir,
    ]:
        os.makedirs(d, exist_ok=True)
