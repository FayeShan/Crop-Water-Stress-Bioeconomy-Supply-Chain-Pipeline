# Input: Monthly Water Footprint NC files

Place ACEA blue-water irrigated footprint NC files here. The expected
filename pattern is:

```
acea_5arc_{crop}_wfp_blue_ir_global_monthly_{START_YEAR}_{END_YEAR}.nc
```

Example filenames:

```
acea_5arc_wh_wfp_blue_ir_global_monthly_1990_2018.nc
acea_5arc_mai_wfp_blue_ir_global_monthly_1990_2018.nc
acea_5arc_ri1_wfp_blue_ir_global_monthly_1990_2018.nc
```

Each file must contain the variable `wfp_blue_ir` with dimensions
`(time, lat, lon)` on the 5-arcmin global grid (2160 × 4320).

**Source**: ACEA (Aqueduct Crop Evapotranspiration Analysis) —
https://data.4tu.nl/ (search "acea water footprint")
