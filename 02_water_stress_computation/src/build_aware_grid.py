"""
One-time construction of the AWARE gridded monthly weights NC.

This module documents how ``aware/aware_gridcell.nc`` was built from the
original AWARE tables. The file is shipped with this repository, so the
code below is kept commented for reference — uncomment if you need to
regenerate the weights (e.g. for a different target grid).

Inputs (see ``aware/`` subfolder):
  - AWARE.xlsx            (raw AWARE CFs, sheet index 1)
  - Placemarks_Data.csv   (Longitude, Latitude for each AWARE point)
  - AWARE_with_geo.csv    (produced below: AWARE CFs + lat/lon)

Output:
  - aware_gridcell.nc     (monthly_weight, dims: month=12, lat, lon)

The nearest-neighbour assignment uses ``scipy.spatial.cKDTree`` on (lat,
lon) pairs from an example monthly-WFP NC, matched against AWARE points.
"""

# import numpy as np
# import pandas as pd
# import xarray as xr
# from scipy.spatial import cKDTree


# def build_aware_with_geo(
#     excel_path: str,
#     placemarks_csv: str,
#     out_csv: str,
# ) -> str:
#     """Merge AWARE CFs with their lon/lat and save as CSV."""
#     wateraware = pd.read_excel(excel_path, sheet_name=1)
#     place_geoinfo = pd.read_csv(placemarks_csv)
#     place_geoinfo_selected = place_geoinfo.iloc[:, [1, 2]]  # Longitude, Latitude
#     merged = pd.concat([place_geoinfo_selected, wateraware], axis=1)
#     merged.to_csv(out_csv)
#     return out_csv


# def build_aware_gridcell_nc(
#     merged_csv: str,
#     template_nc: str,
#     out_nc: str,
# ) -> str:
#     """Assign AWARE CFs to a monthly (month, lat, lon) grid via nearest
#     neighbour and save as NetCDF."""
#     wateraware_global = pd.read_csv(merged_csv)
#     wateraware_global.replace(-99999.0, 0, inplace=True)
#
#     ds = xr.open_dataset(template_nc)
#     df_coords = wateraware_global[["Latitude", "Longitude"]].values
#     tree = cKDTree(df_coords)
#
#     lon_vals = ds["lon"].values
#     lat_vals = ds["lat"].values
#     lat_lon_pairs = np.array([(lat, lon) for lat in lat_vals for lon in lon_vals])
#     _, indices = tree.query(lat_lon_pairs)
#     indices_grid = indices.reshape(len(lat_vals), len(lon_vals))
#
#     months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
#               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
#     monthly_weights = np.zeros((12, len(lat_vals), len(lon_vals)))
#
#     for m, month in enumerate(months):
#         for i in range(len(lat_vals)):
#             for j in range(len(lon_vals)):
#                 df_idx = indices_grid[i, j]
#                 value = wateraware_global.iloc[df_idx][month]
#                 monthly_weights[m, i, j] = value if value != -99999 else np.nan
#
#     weights_ds = xr.Dataset(
#         {"monthly_weight": (("month", "lat", "lon"), monthly_weights)},
#         coords={
#             "month": np.arange(1, 13),
#             "lat": lat_vals,
#             "lon": lon_vals,
#         },
#     )
#     weights_ds.to_netcdf(out_nc)
#     return out_nc
