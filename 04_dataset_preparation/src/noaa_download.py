"""Download NOAA climate oscillation indices as monthly CSV files.

Five teleconnection indices come from the NOAA CPC FTP (.tim, converted
to CSV); four more (SOI, GISTEMP global mean SST, PDO, Nino) are fetched
via HTTPS. The CSVs are consumed by ``feature_engineering.py``.
"""

import logging
import os
from ftplib import FTP
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


CPC_FTP_HOST = "ftp.cpc.ncep.noaa.gov"
CPC_FTP_INDICES = {
    "wp":   "/wd52dg/data/indices/wp_index.tim",
    "pna":  "/wd52dg/data/indices/pna_index.tim",
    "nao":  "/wd52dg/data/indices/nao_index.tim",
    "ea":   "/wd52dg/data/indices/ea_index.tim",
    "epnp": "/wd52dg/data/indices/epnp_index.tim",
}

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WaterStressPipeline/1.0)",
}


def _download_ftp(
    ftp_host: str, remote_path: str, local_path: Path,
    logger: logging.Logger,
) -> bool:
    try:
        ftp = FTP(ftp_host)
        ftp.login()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            ftp.retrbinary(f"RETR {remote_path}", f.write)
        ftp.quit()
        logger.info(f"  {local_path.name}: downloaded from ftp://{ftp_host}{remote_path}")
        return True
    except Exception as e:
        logger.error(f"  {local_path.name}: FTP download failed — {e}")
        return False


def _convert_tim_to_csv(tim_path: Path) -> Path:
    """Convert a CPC ``.tim`` index file (year + 12 monthly values per row)
    to a CSV with columns: Year, Month_1..Month_12.
    """
    lines = [l.strip() for l in tim_path.read_text().splitlines() if l.strip()]
    # Find first line that starts with a 4-digit year
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 13 and parts[0].isdigit() and len(parts[0]) == 4:
            rows.append(parts[:13])
    columns = ["Year"] + [f"Month_{i}" for i in range(1, 13)]
    df = pd.DataFrame(rows, columns=columns)
    df = df.apply(pd.to_numeric, errors="ignore")
    csv_path = tim_path.with_suffix(".csv").with_name(tim_path.stem + ".csv")
    # e.g. wp_index.tim -> wp_index.csv (matches shipped format)
    df.to_csv(csv_path, index=False)
    return csv_path


def _download_reqsoi(out_dir: Path, logger: logging.Logger) -> Optional[Path]:
    url = "https://www.cpc.ncep.noaa.gov/data/indices/reqsoi.for"
    try:
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        rows = [line.split() for line in lines if line.strip()]
        df = pd.DataFrame(rows, columns=["Year"] + [f"Month_{i}" for i in range(1, 13)])
        out = out_dir / "reqsoi.csv"
        df.to_csv(out, index=False)
        logger.info(f"  reqsoi.csv: downloaded")
        return out
    except Exception as e:
        logger.error(f"  reqsoi.csv: failed — {e}")
        return None


def _download_gmsst(out_dir: Path, logger: logging.Logger) -> Optional[Path]:
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v3/GLB.Ts+dSST.txt"
    try:
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("Year"):
                data_start = i
                break
        data_lines = lines[data_start:]
        rows = [line.split() for line in data_lines if line.strip()]
        columns = [
            "Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec",
            "J-D", "D-N", "DJF", "MAM", "JJA", "SON", "Year_Duplicate",
        ]
        df = pd.DataFrame(rows[1:], columns=columns)
        df = df.apply(pd.to_numeric, errors="ignore")
        df.drop(columns=["Year_Duplicate"], inplace=True, errors="ignore")
        out = out_dir / "gmsst.csv"
        df.to_csv(out, index=False)
        logger.info(f"  gmsst.csv: downloaded")
        return out
    except Exception as e:
        logger.error(f"  gmsst.csv: failed — {e}")
        return None


def _download_pdo(out_dir: Path, logger: logging.Logger) -> Optional[Path]:
    url = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
    try:
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("Year"):
                data_start = i
                break
        data_lines = lines[data_start:]
        rows = [line.split() for line in data_lines if line.strip()]
        columns = ["Year"] + [f"Month_{i}" for i in range(1, 13)]
        df = pd.DataFrame(rows[1:], columns=columns)
        df = df.apply(pd.to_numeric, errors="ignore")
        out = out_dir / "pdo.csv"
        df.to_csv(out, index=False)
        logger.info(f"  pdo.csv: downloaded")
        return out
    except Exception as e:
        logger.error(f"  pdo.csv: failed — {e}")
        return None


def _download_nino(out_dir: Path, logger: logging.Logger) -> Optional[Path]:
    url = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
    try:
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("YR"):
                data_start = i
                break
        data_lines = lines[data_start:]
        rows = [line.split() for line in data_lines if line.strip()]
        columns = ["Year", "Month", "NINO1+2", "ANOM1",
                   "NINO3", "ANOM3", "NINO4", "ANOM4",
                   "NINO3.4", "ANOM3.4"]
        df = pd.DataFrame(rows[1:], columns=columns)
        df = df.apply(pd.to_numeric, errors="ignore")
        out = out_dir / "nino.csv"
        df.to_csv(out, index=False)
        logger.info(f"  nino.csv: downloaded")
        return out
    except Exception as e:
        logger.error(f"  nino.csv: failed — {e}")
        return None


def download_indices(
    output_dir: str,
    logger: Optional[logging.Logger] = None,
    cleanup_tim: bool = True,
) -> None:
    """Download all supported NOAA / climate indices as CSVs into ``output_dir``.

    Args:
        output_dir: Directory to write CSVs into (created if missing).
        logger: Optional logger.
        cleanup_tim: Remove intermediate ``.tim`` files after CSV conversion.
    """
    logger = logger or logging.getLogger(__name__)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  NOAA / climate index downloader")
    logger.info("=" * 60)
    logger.info(f"  Output dir: {out_dir}")

    # 1) CPC FTP .tim indices -> .csv
    for name, remote in CPC_FTP_INDICES.items():
        tim_path = out_dir / f"{name}_index.tim"
        if _download_ftp(CPC_FTP_HOST, remote, tim_path, logger):
            try:
                _convert_tim_to_csv(tim_path)
                logger.info(f"  {name}_index.csv: converted")
            except Exception as e:
                logger.error(f"  {name}_index.csv: conversion failed — {e}")
            if cleanup_tim and tim_path.exists():
                tim_path.unlink()

    # 2) HTTPS indices directly as CSV
    _download_reqsoi(out_dir, logger)
    _download_gmsst(out_dir, logger)
    _download_pdo(out_dir, logger)
    _download_nino(out_dir, logger)

    logger.info("Done.")
