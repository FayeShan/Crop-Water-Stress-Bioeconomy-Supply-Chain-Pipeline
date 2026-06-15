"""Validate the reconstructed 14-sector CWS satellite against the GLORIA agricultural water-stress satellite (row 390 of the TQ table) over overlapping years.

Reads the GLORIA TQ row as ``row 390 -> (n_regions, n_sectors)`` via the
supply-col index formula ``s + n_sectors * c * 2`` (same logic as
``build_hybrid_satellite.py``).
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def _supply_cols(n_regions: int, n_sectors: int) -> np.ndarray:
    """Supply-side column indices in the GLORIA SUT row."""
    return np.array([s + n_sectors * c * 2
                     for c in range(n_regions)
                     for s in range(n_sectors)])


def _load_our(year: int, our_dir: Path, unit_scale: float,
              n_regions: int, n_sectors: int) -> np.ndarray:
    """Load ``satellite_{year}.npy`` and convert to million m^3."""
    arr = np.load(our_dir / f"satellite_{year}.npy")
    arr = arr.reshape(n_regions, n_sectors) / unit_scale
    return arr


def _load_gloria_ag(year: int, gloria_dir: Path, ag_row_index: int,
                    supply_cols: np.ndarray,
                    n_regions: int, n_sectors: int) -> np.ndarray:
    """Read the agriculture water-stress row from the GLORIA TQ CSV and
    reshape to ``(n_regions, n_sectors)``.
    """
    path = gloria_dir / str(year) / f"TQ_{year}.csv"
    row = pd.read_csv(path, header=None,
                      skiprows=lambda x: x != ag_row_index)
    return row.iloc[0, supply_cols].values.astype(float).reshape(n_regions, n_sectors)


def _metrics_14(ours: np.ndarray, glo: np.ndarray) -> dict:
    """Compute agreement metrics between two (n_regions, 14) matrices."""
    our14 = ours[:, :14]
    glo14 = glo[:, :14]
    our_tot = float(our14.sum())
    glo_tot = float(glo14.sum())

    our_c = our14.sum(axis=1)
    glo_c = glo14.sum(axis=1)
    both_pos = (our_c > 0) & (glo_c > 0)
    ratios = our_c[both_pos] / glo_c[both_pos]

    o = our14.flatten()
    g = glo14.flatten()
    cell_mask = (o > 0) | (g > 0)

    def _safe_pearson(x, y):
        return float(stats.pearsonr(x, y)[0]) if len(x) > 2 else float("nan")

    def _safe_spearman(x, y):
        return float(stats.spearmanr(x, y)[0]) if len(x) > 2 else float("nan")

    country_mask = (our_c > 0) | (glo_c > 0)

    rmse = float(np.sqrt(((o - g) ** 2).mean()))
    denom = g > 0
    mape = (float(np.abs((o[denom] - g[denom]) / g[denom]).mean() * 100)
            if denom.any() else float("nan"))

    return {
        "our_total_Mm3": our_tot,
        "gloria_total_Mm3": glo_tot,
        "ratio_our_over_gloria_total": our_tot / glo_tot if glo_tot > 0 else float("nan"),
        "country_ratio_n_total": int(both_pos.sum()),
        "country_ratio_median": float(np.median(ratios)) if ratios.size else float("nan"),
        "country_ratio_p25":    float(np.percentile(ratios, 25)) if ratios.size else float("nan"),
        "country_ratio_p75":    float(np.percentile(ratios, 75)) if ratios.size else float("nan"),
        "country_ratio_n_within_2x": int(((ratios > 0.5) & (ratios < 2.0)).sum()),
        "pearson_cell": _safe_pearson(o[cell_mask], g[cell_mask]),
        "pearson_country":       _safe_pearson(our_c[country_mask], glo_c[country_mask]),
        "pearson_country_log1p": _safe_pearson(np.log1p(our_c[country_mask]),
                                               np.log1p(glo_c[country_mask])),
        "spearman_cell":    _safe_spearman(o[cell_mask], g[cell_mask]),
        "spearman_country": _safe_spearman(our_c[country_mask], glo_c[country_mask]),
        "rmse_cell_Mm3": rmse,
        "mape_cell_pct": mape,
    }


def _composition_14_20(ours: np.ndarray, glo: np.ndarray) -> dict:
    """Decompose the hybrid 20-sector agriculture total into the OUR-14
    block and the GLORIA-retained 15-20 block, and report the pure-GLORIA
    14/20 share for reference.
    """
    our_14 = float(ours[:, :14].sum())
    glo_15_20 = float(glo[:, 14:20].sum())
    hybrid_20 = our_14 + glo_15_20

    glo_14 = float(glo[:, :14].sum())
    glo_20 = float(glo[:, :20].sum())

    return {
        "our_14_Mm3":              our_14,
        "gloria_15_20_Mm3":        glo_15_20,
        "hybrid_total_20_Mm3":     hybrid_20,
        "pct_14_in_hybrid_20":     (our_14 / hybrid_20 * 100) if hybrid_20 > 0 else float("nan"),
        "gloria_14_Mm3":           glo_14,
        "gloria_1_20_Mm3":         glo_20,
        "pct_gloria_14_in_gloria_20": (glo_14 / glo_20 * 100) if glo_20 > 0 else float("nan"),
    }


def validate_vs_gloria(
    our_satellite_dir: str,
    gloria_satellite_dir: str,
    output_metrics_csv: str,
    output_composition_csv: str,
    start_year: int,
    end_year: int,
    ag_row_index: int = 390,
    n_regions: int = 164,
    n_sectors: int = 120,
    unit_scale: float = 1e6,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Run the OUR-vs-GLORIA comparison over ``start_year..end_year`` and
    write the two summary CSVs.
    """
    logger = logger or logging.getLogger(__name__)
    our_dir = Path(our_satellite_dir)
    gloria_dir = Path(gloria_satellite_dir)

    logger.info("=" * 70)
    logger.info("  Validate reconstructed 14-sector CWS vs GLORIA agriculture")
    logger.info("=" * 70)
    logger.info(f"  our:    {our_dir}")
    logger.info(f"  gloria: {gloria_dir}")
    logger.info(f"  years:  {start_year}-{end_year}")

    supply_cols = _supply_cols(n_regions, n_sectors)

    metric_rows = []
    comp_rows = []
    for year in range(start_year, end_year + 1):
        our_path = our_dir / f"satellite_{year}.npy"
        gloria_path = gloria_dir / str(year) / f"TQ_{year}.csv"
        if not our_path.exists() or not gloria_path.exists():
            logger.info(f"  {year}: skip (missing inputs)")
            continue

        ours = _load_our(year, our_dir, unit_scale, n_regions, n_sectors)
        glo = _load_gloria_ag(year, gloria_dir, ag_row_index, supply_cols,
                              n_regions, n_sectors)

        m = _metrics_14(ours, glo)
        m["year"] = year
        metric_rows.append(m)

        c = _composition_14_20(ours, glo)
        c["year"] = year
        comp_rows.append(c)

        logger.info(
            f"  {year}: r_country={m['pearson_country']:.3f}  "
            f"r_cell={m['pearson_cell']:.3f}  "
            f"ratio={m['ratio_our_over_gloria_total']:.3f}  "
            f"pct_14_in_20={c['pct_14_in_hybrid_20']:.1f}%")

    if not metric_rows:
        logger.warning("  no overlap years found")
        return []

    df_m = pd.DataFrame(metric_rows)
    df_c = pd.DataFrame(comp_rows)

    df_m = df_m[["year"] + [c for c in df_m.columns if c != "year"]]
    df_c = df_c[["year"] + [c for c in df_c.columns if c != "year"]]

    Path(output_metrics_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_composition_csv).parent.mkdir(parents=True, exist_ok=True)
    df_m.to_csv(output_metrics_csv, index=False)
    df_c.to_csv(output_composition_csv, index=False)

    logger.info(f"  saved: {output_metrics_csv}  ({len(df_m)} years)")
    logger.info(f"  saved: {output_composition_csv}  ({len(df_c)} years)")

    logger.info("")
    logger.info("  === Multi-year summary ===")
    logger.info(f"  OUR total (median):       {df_m['our_total_Mm3'].median():.2f} million m^3")
    logger.info(f"  GLORIA total (median):    {df_m['gloria_total_Mm3'].median():.2f} million m^3")
    logger.info(f"  ratio (median):           {df_m['ratio_our_over_gloria_total'].median():.3f}")
    logger.info(f"  country Pearson r (median): {df_m['pearson_country'].median():.3f}")
    logger.info(f"  cell Pearson r (median):    {df_m['pearson_cell'].median():.3f}")
    logger.info(f"  hybrid 14/20 share (median): {df_c['pct_14_in_hybrid_20'].median():.1f}%")

    return [str(output_metrics_csv), str(output_composition_csv)]
