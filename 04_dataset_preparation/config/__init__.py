"""Configuration loader for the data_pre pipeline."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class WaterStressConfig:
    ws_dir: str


@dataclass
class PredictorsConfig:
    merged_dir: str


@dataclass
class IHAConfig:
    monthly_dir: str
    annual_dir: str


@dataclass
class ReferenceConfig:
    dir: str
    cropmasks_dir: str
    coarse50km_gridid: str
    region9: str
    template_nc: str


@dataclass
class NOAAConfig:
    dir: str


@dataclass
class DataConfig:
    base_train_dir: str
    base_predict_dir: str
    train_dir: str
    predict_dir: str


@dataclass
class DateRangeConfig:
    train_start: int
    train_end: int
    predict_start: int
    predict_end: int


@dataclass
class CropsConfig:
    all: List[str]
    aggregate: List[str]


@dataclass
class FeatureEngineeringConfig:
    lag_vars: List[str]
    lag_range: List[int]
    roll_vars: List[str]
    roll_windows: List[int]
    prepend_years_from_train: List[int]


@dataclass
class PipelineConfig:
    log_dir: str
    log_level: str
    water_stress: WaterStressConfig
    predictors: PredictorsConfig
    iha: IHAConfig
    reference: ReferenceConfig
    noaa: NOAAConfig
    data: DataConfig
    date_range: DateRangeConfig
    crops: CropsConfig
    predictor_vars: List[str]
    feature_engineering: FeatureEngineeringConfig


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return PipelineConfig(
        log_dir=cfg["general"]["log_dir"],
        log_level=cfg["general"]["log_level"],
        water_stress=WaterStressConfig(ws_dir=cfg["water_stress"]["ws_dir"]),
        predictors=PredictorsConfig(merged_dir=cfg["predictors"]["merged_dir"]),
        iha=IHAConfig(
            monthly_dir=cfg["iha"]["monthly_dir"],
            annual_dir=cfg["iha"]["annual_dir"],
        ),
        reference=ReferenceConfig(
            dir=cfg["reference"]["dir"],
            cropmasks_dir=cfg["reference"]["cropmasks_dir"],
            coarse50km_gridid=cfg["reference"]["coarse50km_gridid"],
            region9=cfg["reference"]["region9"],
            template_nc=cfg["reference"]["template_nc"],
        ),
        noaa=NOAAConfig(dir=cfg["noaa"]["dir"]),
        data=DataConfig(
            base_train_dir=cfg["data"]["base_train_dir"],
            base_predict_dir=cfg["data"]["base_predict_dir"],
            train_dir=cfg["data"]["train_dir"],
            predict_dir=cfg["data"]["predict_dir"],
        ),
        date_range=DateRangeConfig(
            train_start=cfg["date_range"]["train_start"],
            train_end=cfg["date_range"]["train_end"],
            predict_start=cfg["date_range"]["predict_start"],
            predict_end=cfg["date_range"]["predict_end"],
        ),
        crops=CropsConfig(
            all=cfg["crops"]["all"],
            aggregate=cfg["crops"]["aggregate"],
        ),
        predictor_vars=cfg["predictor_vars"],
        feature_engineering=FeatureEngineeringConfig(
            lag_vars=cfg["feature_engineering"]["lag_vars"],
            lag_range=cfg["feature_engineering"]["lag_range"],
            roll_vars=cfg["feature_engineering"]["roll_vars"],
            roll_windows=cfg["feature_engineering"]["roll_windows"],
            prepend_years_from_train=cfg["feature_engineering"]["prepend_years_from_train"],
        ),
    )


def create_directories(config: PipelineConfig) -> None:
    for d in [
        config.log_dir,
        config.reference.dir,
        config.reference.cropmasks_dir,
        config.noaa.dir,
        config.data.base_train_dir,
        config.data.base_predict_dir,
        config.data.train_dir,
        config.data.predict_dir,
    ]:
        os.makedirs(d, exist_ok=True)
