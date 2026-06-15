#!/usr/bin/env python3
"""Download required datasets (LUH2-GCB2025, CROPGRIDS, FAOSTAT QCL) for the pipeline."""

import os
import sys
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path("./data")
LUH2_DIR = BASE_DIR / "luh2"
CROPGRIDS_DIR = BASE_DIR / "cropgrids"
FAOSTAT_DIR = BASE_DIR / "faostat"

# LUH2-GCB2025 on Zenodo
# NOTE: Check https://zenodo.org/records/15557904 for the latest version
# The record contains states.nc, management.nc, transitions.nc, staticData.nc
LUH2_ZENODO_DOI = "10.5281/zenodo.15557904"
LUH2_URLS = {
    # These are the typical file names in LUH2-GCB releases.
    # You may need to update these URLs after checking the Zenodo record page.
    # The key files you need are:
    #   - states.nc    : land-use state fractions (c3ann, c4ann, c3per, c4per, c3nfx, etc.)
    #   - management.nc: irrigation fractions, fertilizer, etc.
    #   - staticData_quarterdeg.nc: grid cell areas and country masks
    #
    # IMPORTANT: LUH2-GCB files are large (states.nc ~10-20 GB for full history).
    # You may want to use CDO or NCO to subset years 2000-2024 after download.
    #
    # Option A: Download from Zenodo (latest GCB2025)
    #   Visit: https://zenodo.org/records/15557904
    #   Download manually or use zenodo_get:
    #     pip install zenodo_get
    #     zenodo_get 15557904
    #
    # Option B: Download LUH2 v2h from the official site (covers 850-2015)
    #   https://luh.umd.edu/data.shtml
    #   states: https://luh.umd.edu/LUH2/LUH2_v2h/states.nc
    #   management: https://luh.umd.edu/LUH2/LUH2_v2h/management.nc
    #   static: https://luh.umd.edu/LUH2/LUH2_v2h/staticData_quarterdeg.nc
    #
    # Option C: Use LUH2-GCB2019 from ORNL DAAC (covers 850-2019, well-documented)
    #   https://doi.org/10.3334/ORNLDAAC/1851
    #   Requires NASA Earthdata login
}

# CROPGRIDS on Figshare
CROPGRIDS_FIGSHARE_DOI = "10.6084/m9.figshare.22491997"
CROPGRIDS_URL = "https://figshare.com/ndownloader/articles/22491997/versions/4"
# NOTE: This downloads a zip file containing NetCDF files for all 173 crops.
# Total size: approximately 2-5 GB compressed.

# FAOSTAT Bulk Download
FAOSTAT_QCL_URL = "https://fenixservices.fao.org/faostat/static/bulkdownloads/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
# Alternative: Use FAOSTAT API for specific crops/countries
FAOSTAT_API_BASE = "https://www.fao.org/faostat/api/v1"


def create_dirs():
    """Create directory structure."""
    for d in [LUH2_DIR, CROPGRIDS_DIR, FAOSTAT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure under {BASE_DIR}/")


def download_luh2():
    """
    Download LUH2-GCB data.
    
    Multiple options are provided because the data is large and hosted on 
    different platforms with different access methods.
    """
    print("=" * 70)
    print("STEP 1: LUH2 - Land Use Harmonization 2")
    print("=" * 70)
    
    readme = LUH2_DIR / "DOWNLOAD_INSTRUCTIONS.txt"
    readme.write_text("""
LUH2 Download Instructions
============================

The LUH2 data files are large (10-20 GB each). Choose ONE of these options:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION A: LUH2-GCB2025 (RECOMMENDED - latest, covers up to ~2025)
──────────────────────────────────────────────────────────────────
    URL: https://zenodo.org/records/15557904
    
    Method 1 - zenodo_get (automated):
        pip install zenodo_get
        cd data/luh2
        zenodo_get 15557904
    
    Method 2 - wget (manual, update filenames from Zenodo page):
        cd data/luh2
        wget https://zenodo.org/records/15557904/files/states.nc
        wget https://zenodo.org/records/15557904/files/management.nc
        wget https://zenodo.org/records/15557904/files/staticData_quarterdeg.nc

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION B: LUH2 v2h (official CMIP6, covers 850-2015)
──────────────────────────────────────────────────────
    URL: https://luh.umd.edu/data.shtml
    
    cd data/luh2
    wget https://luh.umd.edu/LUH2/LUH2_v2h/states.nc
    wget https://luh.umd.edu/LUH2/LUH2_v2h/management.nc
    wget https://luh.umd.edu/LUH2/LUH2_v2h/staticData_quarterdeg.nc
    
    NOTE: Only covers to 2015. You'll need to extend using SSP scenarios
    or FAOSTAT trends for 2016-2024.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION C: LUH2-GCB2019 from ORNL DAAC (covers 850-2019)
────────────────────────────────────────────────────────
    URL: https://doi.org/10.3334/ORNLDAAC/1851
    Requires: NASA Earthdata account (free)
    
    1. Register at https://urs.earthdata.nasa.gov/
    2. Download from ORNL DAAC

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FILES YOU NEED:
    states.nc              - Land-use state fractions per grid cell per year
                             Variables: c3ann, c4ann, c3per, c4per, c3nfx, 
                                        pastr, range, primf, primn, secdf, secdn, urban
    management.nc          - Agricultural management layers
                             Variables: irrig_c3ann, irrig_c4ann, irrig_c3per, 
                                        irrig_c4per, irrig_c3nfx, 
                                        fertl_c3ann, fertl_c4ann, ... (fertilizer)
                                        crpbf_c3ann, ... (biofuel crops)
    staticData_quarterdeg.nc - Grid cell area (carea), country codes, ice/water masks

AFTER DOWNLOAD - Subset to 2000-2024 (saves disk space):
    # Using CDO:
    cdo selyear,2000/2024 states.nc states_2000_2024.nc
    cdo selyear,2000/2024 management.nc management_2000_2024.nc
    
    # Or using NCO:
    ncks -d time,1150,1174 states.nc states_2000_2024.nc  # adjust indices
""")
    
    print(f"  Instructions written to: {readme}")
    print(f"  Please follow the instructions to download LUH2 data.")
    print(f"  Recommended: Option A (LUH2-GCB2025 from Zenodo)")
    
    # Try zenodo_get if available
    try:
        subprocess.run(["zenodo_get", "--help"], capture_output=True, check=True)
        print("\n  zenodo_get is available! You can run:")
        print(f"    cd {LUH2_DIR} && zenodo_get 15557904")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\n  Tip: Install zenodo_get for automated download:")
        print("    pip install zenodo_get")


def download_cropgrids():
    """
    Download CROPGRIDS dataset from Figshare.
    173 crops, 0.05° resolution, circa 2020.
    """
    print("=" * 70)
    print("STEP 2: CROPGRIDS - 173-crop harvested area maps (circa 2020)")
    print("=" * 70)
    
    output_zip = CROPGRIDS_DIR / "cropgrids.zip"
    
    print(f"  Source: Figshare DOI {CROPGRIDS_FIGSHARE_DOI}")
    print(f"  URL: {CROPGRIDS_URL}")
    print(f"  Output: {output_zip}")
    print(f"  Estimated size: ~2-5 GB")
    print()
    
    # Figshare bulk download
    print("  Downloading from Figshare...")
    print("  NOTE: If wget fails, download manually from:")
    print(f"    https://doi.org/10.6084/m9.figshare.22491997")
    print()
    
    cmd = [
        "wget", "-c",
        "--content-disposition",
        "-P", str(CROPGRIDS_DIR),
        CROPGRIDS_URL,
        "-O", str(output_zip)
    ]
    print(f"  Command: {' '.join(cmd)}")
    print()
    
    readme = CROPGRIDS_DIR / "DOWNLOAD_INSTRUCTIONS.txt"
    readme.write_text(f"""
CROPGRIDS Download Instructions
================================

Source: Tang et al. (2024), Scientific Data
DOI: {CROPGRIDS_FIGSHARE_DOI}
URL: https://doi.org/10.6084/m9.figshare.22491997

Method 1 - wget:
    cd {CROPGRIDS_DIR}
    wget -c --content-disposition "{CROPGRIDS_URL}" -O cropgrids.zip
    unzip cropgrids.zip

Method 2 - Manual:
    1. Go to https://figshare.com/articles/dataset/CROPGRIDS/22491997
    2. Click "Download all" 
    3. Extract to {CROPGRIDS_DIR}/

Contents (after extraction):
    - CROPGRIDS_HA_*.nc   : Harvested Area per crop (ha per grid cell)
    - CROPGRIDS_CA_*.nc   : Crop (Physical) Area per crop
    - CODES.zip           : MATLAB code used to create the dataset
    - metadata files

Key variables in NetCDF files:
    - HA: Harvested area in hectares per 0.05° grid cell
    - CA: Crop (physical) area in hectares per 0.05° grid cell
""")
    
    print(f"  Instructions written to: {readme}")

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            print("  Download started successfully!")
        else:
            print("  wget not available or download failed.")
            print("  Please download manually from the URL above.")
    except Exception as e:
        print(f"  Could not start download: {e}")
        print("  Please download manually from the URL above.")


def download_faostat():
    """
    Download FAOSTAT crop production data.
    Country-level, annual, ~300+ crop items.
    """
    print("=" * 70)
    print("STEP 3: FAOSTAT - Country-level crop statistics")
    print("=" * 70)
    
    output_zip = FAOSTAT_DIR / "Production_Crops_Livestock_E_All_Data.zip"
    
    print(f"  Source: FAOSTAT QCL (Crops and livestock products)")
    print(f"  URL: {FAOSTAT_QCL_URL}")
    print(f"  Output: {output_zip}")
    print(f"  Estimated size: ~100-200 MB")
    print()
    
    cmd = [
        "wget", "-c",
        "--no-check-certificate",
        FAOSTAT_QCL_URL,
        "-O", str(output_zip)
    ]
    
    print(f"  Downloading FAOSTAT bulk data...")
    print(f"  Command: {' '.join(cmd)}")
    print()
    
    readme = FAOSTAT_DIR / "DOWNLOAD_INSTRUCTIONS.txt"
    readme.write_text(f"""
FAOSTAT Download Instructions
==============================

Source: FAO - Crops and livestock products (QCL)
URL (bulk): {FAOSTAT_QCL_URL}
Web interface: https://www.fao.org/faostat/en/#data/QCL

Method 1 - wget (automated):
    cd {FAOSTAT_DIR}
    wget -c --no-check-certificate \\
        "{FAOSTAT_QCL_URL}" \\
        -O Production_Crops_Livestock_E_All_Data.zip
    unzip Production_Crops_Livestock_E_All_Data.zip

Method 2 - Manual:
    1. Go to https://www.fao.org/faostat/en/#data/QCL
    2. Click "Bulk Downloads" on the right side
    3. Download "All Data (Normalized)"
    4. Extract to {FAOSTAT_DIR}/

Method 3 - FAOSTAT API (for specific queries):
    import requests
    # Example: Get wheat harvested area for all countries, 2000-2023
    url = "{FAOSTAT_API_BASE}/en/data/QCL"
    params = {{
        "area": ">",           # all countries
        "item": "15",          # wheat
        "element": "2312",     # area harvested
        "year": "2000:2023"
    }}

Contents (after extraction):
    - Production_Crops_Livestock_E_All_Data_(Normalized).csv
    
Key columns:
    - Area Code (M49) : Country code
    - Item Code (CPC) : Crop code  
    - Element Code    : 5312=Area harvested, 5510=Production, 5419=Yield
    - Year            : 1961-2023
    - Value           : The data value
    - Unit            : ha, tonnes, kg/ha

Elements you need:
    5312 = Area harvested (ha)
    5510 = Production (tonnes)  
    5419 = Yield (kg/ha)
""")
    
    print(f"  Instructions written to: {readme}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            print("  Download started successfully!")
        else:
            print("  wget not available or download failed.")
            print("  Please download manually.")
    except Exception as e:
        print(f"  Could not start download: {e}")
        print("  Please download manually.")


def create_crop_cft_mapping():
    """
    Create a mapping table from CROPGRIDS 173 crops to LUH2's 5 CFTs.
    
    LUH2 CFTs:
        c3ann  = C3 annual crops (wheat, rice, barley, potatoes, etc.)
        c4ann  = C4 annual crops (maize, sorghum, millet, sugarcane*)
        c3per  = C3 perennial crops (most fruits, grapes, coffee, cocoa, tea, rubber)
        c4per  = C4 perennial crops (oil palm, some tropical grasses)  
        c3nfx  = C3 nitrogen-fixing crops (soybeans, groundnuts, pulses, beans, lentils)
    
    * Note: Sugarcane is C4 but perennial in practice. LUH2 treats it as C4 annual.
    """
    print("=" * 70)
    print("STEP 4: Creating crop-to-CFT mapping table")
    print("=" * 70)
    
    # This is a partial mapping for the most important crops.
    # A complete mapping for all 173 CROPGRIDS crops needs to be verified
    # against the Monfreda/FAO classification used in LUH2.
    
    mapping_csv = """cropgrids_name,fao_name,cft,photosynthesis,lifecycle,notes
wheat,Wheat,c3ann,C3,annual,
rice,Rice,c3ann,C3,annual,paddy rice
barley,Barley,c3ann,C3,annual,
rye,Rye,c3ann,C3,annual,
oats,Oats,c3ann,C3,annual,
potatoes,Potatoes,c3ann,C3,annual,
sweet_potatoes,Sweet potatoes,c3ann,C3,annual,tuber crop
cassava,Cassava,c3ann,C3,annual,treated as annual in LUH2
sugar_beet,Sugar beet,c3ann,C3,annual,
rapeseed,Rapeseed,c3ann,C3,annual,canola
sunflower,Sunflower seed,c3ann,C3,annual,
cotton,Cotton,c3ann,C3,annual,
tobacco,Tobacco,c3ann,C3,annual,
flax,Flax,c3ann,C3,annual,
sesame,Sesame,c3ann,C3,annual,
maize,Maize,c4ann,C4,annual,
sorghum,Sorghum,c4ann,C4,annual,
millet,Millet,c4ann,C4,annual,pearl and finger millet
sugarcane,Sugarcane,c4ann,C4,annual*,perennial but C4 annual in LUH2
soybeans,Soybeans,c3nfx,C3,annual,nitrogen-fixing
groundnuts,Groundnuts,c3nfx,C3,annual,nitrogen-fixing
beans,Beans dry,c3nfx,C3,annual,nitrogen-fixing
peas,Peas dry,c3nfx,C3,annual,nitrogen-fixing
lentils,Lentils,c3nfx,C3,annual,nitrogen-fixing
chickpeas,Chick peas,c3nfx,C3,annual,nitrogen-fixing
cowpeas,Cow peas dry,c3nfx,C3,annual,nitrogen-fixing
pigeon_peas,Pigeon peas,c3nfx,C3,annual,nitrogen-fixing
broad_beans,Broad beans,c3nfx,C3,annual,nitrogen-fixing
lupins,Lupins,c3nfx,C3,annual,nitrogen-fixing
other_pulses,Other pulses,c3nfx,C3,annual,nitrogen-fixing
coffee,Coffee,c3per,C3,perennial,
cocoa,Cocoa,c3per,C3,perennial,
tea,Tea,c3per,C3,perennial,
grapes,Grapes,c3per,C3,perennial,
apples,Apples,c3per,C3,perennial,
bananas,Bananas,c3per,C3,perennial,
oranges,Oranges,c3per,C3,perennial,
lemons,Lemons and limes,c3per,C3,perennial,
mangoes,Mangoes,c3per,C3,perennial,
pineapples,Pineapples,c3per,C3,perennial,actually CAM but treated as C3per
avocados,Avocados,c3per,C3,perennial,
coconuts,Coconuts,c3per,C3,perennial,
olives,Olives,c3per,C3,perennial,
rubber,Natural rubber,c3per,C3,perennial,
oil_palm,Oil palm fruit,c4per,C4,perennial,one of few C4 perennials
"""
    
    output = FAOSTAT_DIR / "crop_to_cft_mapping.csv"
    output.write_text(mapping_csv)
    
    print(f"  Mapping table written to: {output}")
    print(f"  Contains {len(mapping_csv.strip().split(chr(10)))-1} crop entries")
    print()
    print("  IMPORTANT: This is a partial mapping covering ~45 major crops.")
    print("  You need to complete this for all 173 CROPGRIDS crops.")
    print("  Key rule: check photosynthesis pathway (C3/C4) and lifecycle (annual/perennial)")
    print("  For N-fixing crops (legumes), use c3nfx regardless of annual/perennial.")
    print()
    print("  Reference for complete classification:")
    print("  - LUH2 paper (Hurtt et al. 2020), supplementary materials")
    print("  - Monfreda et al. (2008) crop type classification")
    print("  - CROPGRIDS metadata files")


def main():
    parser = argparse.ArgumentParser(
        description="Download data for gridded crop harvested area pipeline"
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--luh2", action="store_true", help="Download LUH2")
    parser.add_argument("--cropgrids", action="store_true", help="Download CROPGRIDS")
    parser.add_argument("--faostat", action="store_true", help="Download FAOSTAT")
    parser.add_argument("--mapping", action="store_true", help="Create crop-CFT mapping")
    
    args = parser.parse_args()
    
    if not any([args.all, args.luh2, args.cropgrids, args.faostat, args.mapping]):
        parser.print_help()
        print("\nExample: python download_data.py --all")
        return
    
    create_dirs()
    print()
    
    if args.all or args.luh2:
        download_luh2()
        print()
    
    if args.all or args.cropgrids:
        download_cropgrids()
        print()
    
    if args.all or args.faostat:
        download_faostat()
        print()
    
    if args.all or args.mapping:
        create_crop_cft_mapping()
        print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Data directory: {BASE_DIR.resolve()}")
    print(f"  LUH2:      {LUH2_DIR}/      (check DOWNLOAD_INSTRUCTIONS.txt)")
    print(f"  CROPGRIDS: {CROPGRIDS_DIR}/  (check DOWNLOAD_INSTRUCTIONS.txt)")  
    print(f"  FAOSTAT:   {FAOSTAT_DIR}/    (check DOWNLOAD_INSTRUCTIONS.txt)")
    print()
    print("  Next step: Run process_data.py after all data is downloaded.")


if __name__ == "__main__":
    main()
