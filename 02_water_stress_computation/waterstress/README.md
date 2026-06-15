# Output: Per-crop, per-year Water Stress NC files

This folder is populated by `main.py compute-ws`. Structure:

```
waterstress/
├── wh/
│   ├── 1990.nc
│   ├── 1991.nc
│   └── ...
├── mai/
│   └── ...
└── ...
```

Each `{year}.nc` contains a single variable `waterstress`
with dimensions `(time=12, lat, lon)`, computed as:

```
waterstress(t, lat, lon) = wfp_blue_ir(t, lat, lon) * aware_weight(month, lat, lon)
```
