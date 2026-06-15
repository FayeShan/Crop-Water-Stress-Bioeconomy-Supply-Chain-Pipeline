import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ERA5Config:
    output_dir: str
    processed_dir: str
    resampled_dir: str
    variables: list
    months: list


@dataclass
class MODISConfig:
    raw_dir: str
    nc_dir: str
    resampled_dir: str
    start_year: int
    short_name: str
    version: str
    fill_value: int
    var_mapping: dict
    selected_vars: list


@dataclass
class ModisGreenConfig:
    raw_dir: str
    nc_dir: str
    resampled_dir: str
    start_year: int
    keep_hdf: bool


@dataclass
class ModisLandcoverConfig:
    raw_dir: str
    nc_dir: str
    resampled_dir: str
    start_year: int
    keep_hdf: bool


@dataclass
class ModisLanduseConfig:
    raw_dir: str
    nc_dir: str
    resampled_dir: str
    start_year: int
    keep_hdf: bool


@dataclass
class SMIAConfig:
    input_dir: str
    output_dir: str
    resampled_dir: str
    days: list


@dataclass
class TWSAConfig:
    input_dir: str
    output_dir: str
    start_year: int


@dataclass
class ResamplingConfig:
    reference_grid: str
    method: str


@dataclass
class MergeConfig:
    output_dir: str
    compression_level: int
    sources: dict
    extra_files: list


@dataclass
class EncodingConfig:
    dtype: str
    fill_value: float
    zlib: bool
    complevel: int


@dataclass
class PipelineConfig:
    """Main configuration class for the water data pipeline."""
    
    working_dir: str
    log_dir: str
    log_level: str
    train_start: int
    train_end: int
    predict_start: int
    predict_end: int
    era5: ERA5Config
    modis: MODISConfig
    modis_green: Optional[ModisGreenConfig]
    modis_landcover: Optional[ModisLandcoverConfig]
    modis_landuse: Optional[ModisLanduseConfig]
    smia: SMIAConfig
    twsa: TWSAConfig
    resampling: ResamplingConfig
    merge: MergeConfig
    encoding: EncodingConfig
    
    def get_year_range(self, mode: str = "all") -> range:
        """Get year range based on mode."""
        if mode == "train":
            return range(self.train_start, self.train_end + 1)
        elif mode == "predict":
            return range(self.predict_start, self.predict_end + 1)
        else:
            return range(self.train_start, self.predict_end + 1)
    
    def get_encoding(self, variables: list) -> Dict[str, Dict]:
        """Get NetCDF encoding for variables."""
        return {
            var: {
                "zlib": self.encoding.zlib,
                "complevel": self.encoding.complevel,
                "dtype": self.encoding.dtype,
                "_FillValue": self.encoding.fill_value
            }
            for var in variables
        }


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load configuration from YAML file (defaults to config.yaml beside this module)."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    return PipelineConfig(
        working_dir=cfg['general']['working_dir'],
        log_dir=cfg['general']['log_dir'],
        log_level=cfg['general']['log_level'],
        train_start=cfg['date_range']['train_start'],
        train_end=cfg['date_range']['train_end'],
        predict_start=cfg['date_range']['predict_start'],
        predict_end=cfg['date_range']['predict_end'],
        era5=ERA5Config(
            output_dir=cfg['era5']['output_dir'],
            processed_dir=cfg['era5']['processed_dir'],
            resampled_dir=cfg['era5']['resampled_dir'],
            variables=cfg['era5']['variables'],
            months=cfg['era5']['months']
        ),
        modis=MODISConfig(
            raw_dir=cfg['modis']['raw_dir'],
            nc_dir=cfg['modis']['nc_dir'],
            resampled_dir=cfg['modis']['resampled_dir'],
            start_year=cfg['modis']['start_year'],
            short_name=cfg['modis'].get('short_name', 'MOD13C2'),
            version=cfg['modis'].get('version', '061'),
            fill_value=cfg['modis'].get('fill_value', -3000),
            var_mapping=cfg['modis'].get('var_mapping', {}),
            selected_vars=cfg['modis'].get('selected_vars', [])
        ),
        modis_green=ModisGreenConfig(
            raw_dir=cfg['modis_green']['raw_dir'],
            nc_dir=cfg['modis_green']['nc_dir'],
            resampled_dir=cfg['modis_green']['resampled_dir'],
            start_year=cfg['modis_green'].get('start_year', 2000),
            keep_hdf=cfg['modis_green'].get('keep_hdf', False),
        ) if 'modis_green' in cfg else None,
        modis_landcover=ModisLandcoverConfig(
            raw_dir=cfg['modis_landcover']['raw_dir'],
            nc_dir=cfg['modis_landcover']['nc_dir'],
            resampled_dir=cfg['modis_landcover']['resampled_dir'],
            start_year=cfg['modis_landcover'].get('start_year', 2001),
            keep_hdf=cfg['modis_landcover'].get('keep_hdf', False),
        ) if 'modis_landcover' in cfg else None,
        modis_landuse=ModisLanduseConfig(
            raw_dir=cfg['modis_landuse']['raw_dir'],
            nc_dir=cfg['modis_landuse']['nc_dir'],
            resampled_dir=cfg['modis_landuse']['resampled_dir'],
            start_year=cfg['modis_landuse'].get('start_year', 2001),
            keep_hdf=cfg['modis_landuse'].get('keep_hdf', False),
        ) if 'modis_landuse' in cfg else None,
        smia=SMIAConfig(
            input_dir=cfg['smia']['input_dir'],
            output_dir=cfg['smia']['output_dir'],
            resampled_dir=cfg['smia']['resampled_dir'],
            days=cfg['smia']['days']
        ),
        twsa=TWSAConfig(
            input_dir=cfg['twsa']['input_dir'],
            output_dir=cfg['twsa']['output_dir'],
            start_year=cfg['twsa']['start_year']
        ),
        resampling=ResamplingConfig(
            reference_grid=cfg['resampling']['reference_grid'],
            method=cfg['resampling']['method']
        ),
        merge=MergeConfig(
            output_dir=cfg['merge']['output_dir'],
            compression_level=cfg['merge']['compression_level'],
            sources=cfg['merge'].get('sources', {
                'era5': True, 'modis': True, 'smia': True, 'twsa': True
            }),
            extra_files=cfg['merge'].get('extra_files', [])
        ),
        encoding=EncodingConfig(
            dtype=cfg['encoding']['dtype'],
            fill_value=cfg['encoding']['fill_value'],
            zlib=cfg['encoding']['zlib'],
            complevel=cfg['encoding']['complevel']
        )
    )


def create_directories(config: PipelineConfig) -> None:
    """Create all necessary directories from config."""
    dirs = [
        config.working_dir,
        config.log_dir,
        config.era5.output_dir,
        config.era5.processed_dir,
        config.era5.resampled_dir,
        config.modis.raw_dir,
        config.modis.nc_dir,
        config.modis.resampled_dir,
        config.smia.input_dir,
        config.smia.output_dir,
        config.smia.resampled_dir,
        config.twsa.input_dir,
        config.twsa.output_dir,
        config.merge.output_dir,
    ]

    # Optional additional MODIS products
    for opt_cfg in [config.modis_green, config.modis_landcover, config.modis_landuse]:
        if opt_cfg is not None:
            dirs.extend([opt_cfg.raw_dir, opt_cfg.nc_dir, opt_cfg.resampled_dir])

    for d in dirs:
        os.makedirs(d, exist_ok=True)
