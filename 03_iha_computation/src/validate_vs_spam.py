#!/usr/bin/env python3
"""Compare pipeline irrigated harvested area (2020) against SPAM2020 _I.tif files."""

import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# CROP MAPPING: pipeline crop name → SPAM code(s)
CROP_MAPPING = {
    # pipeline_crop: (acea_code, [SPAM_codes])
    'wheat':     ('wh',  ['WHEA']),
    'rice':      ('ri1', ['RICE']),
    'cotton':    ('cot', ['COTT']),
    'sugarcane': ('sgc', ['SUGC']),
    'maize':     ('mai', ['MAIZ']),
    'rapeseed':  ('rap', ['RAPE']),
    'sorghum':   ('sor', ['SORG']),
    'potato':    ('pot', ['POTA']),
    'groundnut': ('nut', ['GROU']),
    'sugarbeet': ('sgb', ['SUGB']),
    'barley':    ('bar', ['BARL']),
    'chickpea':  ('ckp', ['CHIC']),
    'bean':      ('bea', ['BEAN']),
    'sunflower': ('sun', ['SUNF']),
    'soybean':   ('soy', ['SOYB']),
    'millet':    ('mil', ['MILL', 'PMIL']),
    'coffee':    ('cof', ['COFF', 'RCOF']),
    'cocoa':     ('coc', ['COCO']),
    'cowpea':    ('cwp', ['COWP']),
    'cassava':   ('cas', ['CASS']),
    'oilpalm':   ('plm', ['OILP']),
    # Vegetables: aggregate multiple SPAM crops
    # We compare sum of pipeline veg crops vs SPAM TOMA+ONIO+VEGE
}

# For vegetables, map individual pipeline crops to SPAM
VEG_PIPELINE_CROPS = [
    'tomato', 'onion', 'cabbage', 'lettuce', 'cucumberetc', 'eggplant',
    'spinach', 'garlic', 'greenonion', 'okra', 'cauliflower', 'carrot',
    'artichoke', 'asparagus', 'pumpkinetc', 'chilleetc', 'watermelon', 'melonetc'
]
VEG_SPAM_CODES = ['TOMA', 'ONIO', 'VEGE']


def load_pipeline_irrigated_2020(pipeline_dir, crop_name):
    """Load irrigated HA for year 2020 from pipeline NetCDF output."""
    nc_path = Path(pipeline_dir) / f"{crop_name}.nc"
    if not nc_path.exists():
        return None, None, None
    
    ds = xr.open_dataset(nc_path)

    times = ds['time'].values
    idx_2020 = None
    for i, t in enumerate(times):
        if hasattr(t, 'year'):
            if t.year == 2020: idx_2020 = i; break
        elif int(t) == 2020:
            idx_2020 = i; break
    
    if idx_2020 is None:
        ds.close()
        return None, None, None
    
    ha_irr = ds['harvested_area_irrigated'].isel(time=idx_2020).values
    lat = ds['lat'].values
    lon = ds['lon'].values
    ds.close()

    ha_irr = np.where(np.isfinite(ha_irr) & (ha_irr > 0), ha_irr, 0).astype(np.float32)
    return ha_irr, lat, lon


def load_spam_irrigated(spam_dir, spam_codes):
    """Load and sum SPAM irrigated TIF files.
    
    Supports naming patterns:
      - spam2020_V2r0_global_H_{CODE}_I.tif  (SPAM2020 V2r0)
      - 2020_{CODE}_I.tif                     (simplified naming)
    """
    import rioxarray
    
    total = None
    for code in spam_codes:
        # Try different naming patterns
        candidates = [
            Path(spam_dir) / f"spam2020_V2r0_global_H_{code}_I.tif",
            Path(spam_dir) / f"spam2020_V2r0_global_H_{code.upper()}_I.tif",
            Path(spam_dir) / f"spam2020_V2r0_global_H_{code.lower()}_I.tif",
            Path(spam_dir) / f"2020_{code}_I.tif",
        ]
        
        tif_path = None
        for c in candidates:
            if c.exists():
                tif_path = c
                break
        
        if tif_path is None:
            print(f"    WARNING: no file found for {code} in {spam_dir}")
            continue
        
        da = rioxarray.open_rasterio(tif_path).squeeze("band", drop=True)
        if not da.rio.crs:
            da = da.rio.write_crs("EPSG:4326")

        if 'y' in da.dims:
            da = da.rename({'y': 'lat', 'x': 'lon'})

        da = da.where(da > 0, 0)
        
        if total is None:
            total = da
        else:
            total = total + da
    
    return total


def aggregate_to_coarse(fine_data, fine_lat, fine_lon, factor=10):
    """
    Aggregate fine grid (0.05°) to coarser grid for spatial comparison.
    factor=10 → 0.5° grid (same as SPAM)
    
    Ensures output lat is always ascending (S→N).
    """
    # Ensure lat is ascending before aggregation
    if len(fine_lat) > 1 and fine_lat[0] > fine_lat[-1]:
        fine_data = fine_data[::-1, :]
        fine_lat = fine_lat[::-1]
    
    nlat, nlon = fine_data.shape
    # Trim to be divisible by factor
    nlat_trim = (nlat // factor) * factor
    nlon_trim = (nlon // factor) * factor
    trimmed = fine_data[:nlat_trim, :nlon_trim]

    coarse = trimmed.reshape(nlat_trim // factor, factor,
                              nlon_trim // factor, factor).sum(axis=(1, 3))
    
    # Coarse lat/lon (centers)
    coarse_lat = fine_lat[:nlat_trim].reshape(-1, factor).mean(axis=1)
    coarse_lon = fine_lon[:nlon_trim].reshape(-1, factor).mean(axis=1)
    
    return coarse, coarse_lat, coarse_lon


def compare_one_crop(crop_name, spam_codes, pipeline_dir, spam_dir):
    """Compare one crop: pipeline vs SPAM."""
    result = {
        'crop': crop_name,
        'spam_codes': '+'.join(spam_codes),
    }
    
    ha_pipe, lat_p, lon_p = load_pipeline_irrigated_2020(pipeline_dir, crop_name)
    if ha_pipe is None:
        result['status'] = 'pipeline_missing'
        return result, None
    
    pipe_total = ha_pipe.sum()
    result['pipeline_total_Mha'] = pipe_total / 1e6
    result['pipeline_nonzero_cells'] = int((ha_pipe > 0).sum())

    try:
        spam_da = load_spam_irrigated(spam_dir, spam_codes)
    except Exception as e:
        result['status'] = f'spam_error: {e}'
        return result, None
    
    if spam_da is None:
        result['status'] = 'spam_missing'
        return result, None
    
    spam_data = spam_da.values
    spam_data = np.where(np.isfinite(spam_data) & (spam_data > 0), spam_data, 0)
    spam_total = spam_data.sum()
    
    result['spam_total_Mha'] = spam_total / 1e6
    result['spam_nonzero_cells'] = int((spam_data > 0).sum())
    
    if spam_total == 0:
        result['ratio_pipe_over_spam'] = np.inf
        result['pearson_r'] = np.nan
        result['spearman_r'] = np.nan
        result['status'] = 'spam_zero'
        return result, None
    
    result['ratio_pipe_over_spam'] = pipe_total / spam_total
    result['status'] = 'ok'
    
    # Aggregate pipeline to SPAM resolution for spatial comparison
    # SPAM is typically 0.0833° (5 arcmin), pipeline is 0.05° (3 arcmin)
    # Let's aggregate both to ~0.5° for comparison
    spam_lat = spam_da['lat'].values
    spam_lon = spam_da['lon'].values
    
    # Ensure SPAM lat is ascending (SPAM TIF is typically N→S)
    if len(spam_lat) > 1 and spam_lat[0] > spam_lat[-1]:
        spam_data = spam_data[::-1, :]
        spam_lat = spam_lat[::-1]
    
    # Aggregate pipeline: 0.05° → 0.5° (factor=10)
    pipe_coarse, lat_c_p, lon_c_p = aggregate_to_coarse(ha_pipe, lat_p, lon_p, factor=10)
    
    # Aggregate SPAM: need to figure out SPAM resolution first
    spam_step = abs(spam_lat[1] - spam_lat[0]) if len(spam_lat) > 1 else 0.0833
    spam_factor = max(1, int(round(0.5 / spam_step)))
    spam_coarse, lat_c_s, lon_c_s = aggregate_to_coarse(spam_data, spam_lat, spam_lon, factor=spam_factor)
    
    # Match grids for correlation (use nearest neighbor on the coarser grid)
    # Simple approach: both are now ~0.5°, just compare if shapes match
    min_lat = max(len(lat_c_p), len(lat_c_s))
    min_lon = max(len(lon_c_p), len(lon_c_s))
    
    # For correlation, just use the common shape
    h = min(pipe_coarse.shape[0], spam_coarse.shape[0])
    w = min(pipe_coarse.shape[1], spam_coarse.shape[1])
    
    p_flat = pipe_coarse[:h, :w].ravel()
    s_flat = spam_coarse[:h, :w].ravel()
    
    # Only compare where at least one has data
    mask = (p_flat > 0) | (s_flat > 0)
    if mask.sum() > 10:
        from scipy.stats import pearsonr, spearmanr
        r_pearson, _ = pearsonr(p_flat[mask], s_flat[mask])
        r_spearman, _ = spearmanr(p_flat[mask], s_flat[mask])
        result['pearson_r'] = r_pearson
        result['spearman_r'] = r_spearman
    else:
        result['pearson_r'] = np.nan
        result['spearman_r'] = np.nan
    
    return result, {
        'pipe_coarse': pipe_coarse[:h, :w],
        'spam_coarse': spam_coarse[:h, :w],
        'lat': lat_c_p[:h],
        'lon': lon_c_p[:w],
    }


def plot_comparison(crop_name, spatial_data, output_dir):
    """Plot side-by-side maps + scatter plot."""
    if spatial_data is None:
        return None
    
    pipe = spatial_data['pipe_coarse']
    spam = spatial_data['spam_coarse']
    lat = spatial_data['lat']
    lon = spatial_data['lon']
    
    # Safety: skip if arrays are empty
    if pipe.size == 0 or spam.size == 0 or len(lat) == 0 or len(lon) == 0:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    pipe = spatial_data['pipe_coarse']
    spam = spatial_data['spam_coarse']
    lat = spatial_data['lat']
    lon = spatial_data['lon']
    
    vmax = max(np.percentile(pipe[pipe > 0], 99) if (pipe > 0).any() else 1,
               np.percentile(spam[spam > 0], 99) if (spam > 0).any() else 1)

    im1 = axes[0].imshow(pipe[::-1], extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                          aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    axes[0].set_title(f'Pipeline Irrigated HA\n({crop_name}, 2020)')
    plt.colorbar(im1, ax=axes[0], label='ha', shrink=0.7)

    im2 = axes[1].imshow(spam[::-1], extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                          aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    axes[1].set_title(f'SPAM2020 Irrigated HA\n({crop_name})')
    plt.colorbar(im2, ax=axes[1], label='ha', shrink=0.7)

    mask = (pipe > 0) | (spam > 0)
    if mask.sum() > 0:
        axes[2].scatter(spam[mask].ravel(), pipe[mask].ravel(), alpha=0.3, s=5, c='steelblue')
        max_val = max(spam[mask].max(), pipe[mask].max())
        axes[2].plot([0, max_val], [0, max_val], 'r--', lw=1, label='1:1 line')
        axes[2].set_xlabel('SPAM2020 Irrigated HA (ha)')
        axes[2].set_ylabel('Pipeline Irrigated HA (ha)')
        axes[2].set_title(f'Scatter (0.5° cells)\n{crop_name}')
        axes[2].legend()
        axes[2].set_aspect('equal')
    
    plt.tight_layout()
    out_path = Path(output_dir) / f'validation_{crop_name}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline irrigated HA vs SPAM2020")
    parser.add_argument("--pipeline_dir", required=True, help="Directory with pipeline .nc outputs")
    parser.add_argument("--spam_dir", required=True, help="Directory with SPAM 2020_*_I.tif files")
    parser.add_argument("--output_dir", default="./validation", help="Output directory for plots and summary")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  Validation: Pipeline Irrigated HA vs SPAM2020")
    print("=" * 70)
    
    all_results = []
    
    for crop_name, (acea, spam_codes) in CROP_MAPPING.items():
        print(f"\n  [{crop_name}] SPAM: {'+'.join(spam_codes)}")
        result, spatial = compare_one_crop(crop_name, spam_codes, args.pipeline_dir, args.spam_dir)
        
        if result['status'] == 'ok':
            ratio = result['ratio_pipe_over_spam']
            flag = "✅" if 0.3 < ratio < 3.0 else "⚠️" if 0.1 < ratio < 10.0 else "❌"
            print(f"    Pipeline: {result['pipeline_total_Mha']:.2f} M ha")
            print(f"    SPAM:     {result['spam_total_Mha']:.2f} M ha")
            print(f"    Ratio:    {ratio:.2f} {flag}")
            print(f"    Pearson r: {result.get('pearson_r', 'N/A'):.3f}" if isinstance(result.get('pearson_r'), float) else "")
            
            if not args.no_plots:
                plot_path = plot_comparison(crop_name, spatial, output_dir)
                if plot_path:
                    print(f"    Plot: {plot_path}")
        else:
            print(f"    Status: {result['status']}")
        
        all_results.append(result)
    
    print(f"\n  [vegetables] SPAM: {'+'.join(VEG_SPAM_CODES)}")
    # Vegetables would require summing multiple pipeline crops; not yet implemented.

    df_results = pd.DataFrame(all_results)
    summary_path = output_dir / "validation_summary.csv"
    df_results.to_csv(summary_path, index=False)
    
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    
    ok_results = df_results[df_results['status'] == 'ok']
    if len(ok_results) > 0:
        print(f"\n  {'Crop':15s} {'Pipeline':>10s} {'SPAM':>10s} {'Ratio':>8s} {'Pearson':>8s} {'Spearman':>8s}")
        print(f"  {'-'*65}")
        for _, r in ok_results.iterrows():
            ratio = r['ratio_pipe_over_spam']
            flag = "✅" if 0.3 < ratio < 3.0 else "⚠️"
            pr = f"{r['pearson_r']:.3f}" if pd.notna(r.get('pearson_r')) else "N/A"
            sr = f"{r['spearman_r']:.3f}" if pd.notna(r.get('spearman_r')) else "N/A"
            print(f"  {r['crop']:15s} {r['pipeline_total_Mha']:10.2f} {r['spam_total_Mha']:10.2f} "
                  f"{ratio:8.2f} {pr:>8s} {sr:>8s} {flag}")
        
        print(f"\n  Global total:")
        print(f"    Pipeline: {ok_results['pipeline_total_Mha'].sum():.1f} M ha")
        print(f"    SPAM:     {ok_results['spam_total_Mha'].sum():.1f} M ha")

        valid_r = ok_results['pearson_r'].dropna()
        if len(valid_r) > 0:
            print(f"\n  Spatial correlation (Pearson r):")
            print(f"    Mean: {valid_r.mean():.3f}")
            print(f"    Min:  {valid_r.min():.3f} ({ok_results.loc[valid_r.idxmin(), 'crop']})")
            print(f"    Max:  {valid_r.max():.3f} ({ok_results.loc[valid_r.idxmax(), 'crop']})")
    
    print(f"\n  Summary saved: {summary_path}")
    print(f"  Plots saved:   {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
