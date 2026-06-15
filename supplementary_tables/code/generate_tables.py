"""Generate the paper's main and supplementary tables from the pipeline outputs.

Each table is produced by an independent function that is guarded on its
inputs, so a missing input skips that table rather than aborting the run.
Edit the PATHS block below to point at your own outputs. Inputs map to:

  EVAL_DIR        Stage 05 eval-phase results (per-crop summary.json,
                  feature_importance.csv, country_eval_*.csv)
  PRED_CSV_DIR    Stage 05 exported predictions ({crop}.csv)
  PRED_COUNTRY    per-crop country-month predictions ({crop}_country.csv,
                  Stage 06 `agg-to-country` output)
  CROPS_27_CSV    Stage 06 crops_27_1990_2024.csv
  BASE_AGG_DIR    historical country-aggregated reference
                  ({year}/{crop}_country_{year}.csv) — analysis intermediate
  VALIDATION_DIR  validation diagnostics (dir1a/dir1b/dir2a/dir2b/dir3 .csv)
                  — analysis intermediate
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Inputs (edit to taste) ──
EVAL_DIR = REPO_ROOT / "05_harm_model" / "results" / "eval"
PRED_CSV_DIR = REPO_ROOT / "05_harm_model" / "results" / "final" / "predictions_csv"
PRED_COUNTRY = REPO_ROOT / "06_gloria_mapping" / "data" / "predictions_country"
CROPS_27_CSV = REPO_ROOT / "06_gloria_mapping" / "data" / "crops_27_1990_2024.csv"
BASE_AGG_DIR = Path("./inputs/base_ws_countryagg")   # analysis intermediate — supply your own
VALIDATION_DIR = Path("./inputs/validation")         # analysis intermediate — supply your own

OUT_DIR = Path("./tables")

ALL_CROPS = [
    "aff", "bar", "bea", "cas", "ckp", "coc", "cof", "cot", "cwp",
    "mai", "mil", "nut", "oac", "pdc", "pec", "plm", "pot", "rap",
    "ri1", "ri2", "sgb", "sgc", "sor", "soy", "sun", "vgt", "wh",
]
UNWEIGHTED = {"soy", "sor", "sgb", "rap"}


def _tier(rho):
    if rho >= 0.5:
        return "Strong"
    if rho >= 0.3:
        return "Moderate"
    return "Weak"


def table_s1_grid_metrics(out):
    """Table S1 — grid-level metrics, Global vs Global-weighted."""
    rows = []
    for crop in ALL_CROPS:
        for variant, label in [("global", "G"), ("global_weighted", "GW")]:
            sp = EVAL_DIR / crop / variant / "summary.json"
            if not sp.exists():
                continue
            m = json.load(open(sp))["metrics"]["mean"]
            rows.append({"crop": crop, "model": label, "r2": m.get("r2"),
                         "rmse": m.get("rmse"), "nrmse": m.get("nrmse"),
                         "hit_rate_p90": m.get("hit_rate_p90"), "picp_80": m.get("picp_80")})
    if not rows:
        print("S1: no eval summaries found, skipping")
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="crop", columns="model",
                     values=["r2", "rmse", "nrmse", "hit_rate_p90", "picp_80"])
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index().sort_values("r2_GW", ascending=False, na_position="last")
    pivot.to_csv(out / "table_s1_full_metrics.csv", index=False)
    print(f"S1: saved ({len(pivot)} crops)")


def table_s2_adoption(out):
    """Table S2 — regional/extreme adoption decisions."""
    rows = []
    for crop in ALL_CROPS:
        row = {"crop": crop}
        for subdir, prefix in [("regional", "R"), ("regional_weighted", "RW"),
                               ("extreme", "E"), ("extreme_weighted", "EW")]:
            sp = EVAL_DIR / crop / subdir / "summary.json"
            if sp.exists():
                s = json.load(open(sp))
                trained = s.get("trained_regions", [])
                row[f"{prefix}_adopted"] = len(trained)
                row[f"{prefix}_total"] = s.get("n_regions", len(trained) + len(s.get("skipped_regions", [])))
                row[f"{prefix}_regions"] = ", ".join(trained)
            else:
                row[f"{prefix}_adopted"] = 0
                row[f"{prefix}_total"] = 0
                row[f"{prefix}_regions"] = ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_s2_regional_adoption.csv", index=False)
    print(f"S2: saved ({len(df)} crops)")


def table_s3_weighted_comparison(out):
    """Table S3 — weighted vs unweighted country-level comparison."""
    unw_path = EVAL_DIR / "country_eval_summary_global.csv"
    wt_path = EVAL_DIR / "country_eval_comparison_weighted.csv"
    if not (unw_path.exists() and wt_path.exists()):
        print("S3: country_eval CSVs missing, skipping")
        return
    df_unw = pd.read_csv(unw_path)
    df_wt = pd.read_csv(wt_path)
    merged = df_unw[["crop", "grid_r2", "country_year_r2", "correlation", "n_countries"]].rename(
        columns={"grid_r2": "grid_GR", "country_year_r2": "ctry_GR", "correlation": "corr_GR"})
    if "grid_GWR" in df_wt.columns:
        merged = merged.merge(df_wt[["crop", "grid_GWR", "ctry_GWR"]], on="crop", how="outer")
    merged["selected"] = merged["crop"].apply(lambda c: "G+R" if c in UNWEIGHTED else "GW+RW")
    if "ctry_GWR" in merged.columns:
        merged["ctry_diff"] = merged["ctry_GWR"] - merged["ctry_GR"]
    merged = merged.sort_values("ctry_GR", ascending=False, na_position="last")
    merged.to_csv(out / "table_s3_weighted_comparison.csv", index=False)
    print(f"S3: saved ({len(merged)} crops)")


def table_s6_feature_importance(out):
    """Table S6 — feature importance averaged across crops."""
    all_fi = []
    for crop in ALL_CROPS:
        for subdir in ["global_weighted", "global"]:
            fi_path = EVAL_DIR / crop / subdir / "feature_importance.csv"
            if fi_path.exists():
                df = pd.read_csv(fi_path)
                if "importance" in df.columns and "feature" in df.columns:
                    total = df["importance"].sum()
                    df["importance_norm"] = df["importance"] / total if total > 0 else 0
                    df["crop"] = crop
                    all_fi.append(df[["crop", "feature", "importance_norm"]])
                break
    if not all_fi:
        print("S6: no feature_importance.csv found, skipping")
        return
    df_fi = pd.concat(all_fi, ignore_index=True)
    avg = df_fi.groupby("feature")["importance_norm"].agg(["mean", "std", "count"]).reset_index()
    avg.columns = ["feature", "mean_importance", "std_importance", "n_crops"]
    avg = avg.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    avg.index += 1
    avg.index.name = "rank"
    avg.to_csv(out / "table_s6_feature_importance.csv")
    print(f"S6: saved ({len(avg)} features)")


def table_s7_koppen(out):
    """Table S7 — Köppen (30) vs THZ×MST (9) region comparison for wheat/maize."""
    rows = []
    for crop in ["wh", "mai"]:
        for subdir in ["regional", "regional_weighted"]:
            sp = EVAL_DIR / crop / subdir / "summary.json"
            if not sp.exists():
                continue
            s = json.load(open(sp))
            regime = "30-Koppen" if s.get("n_regions", 0) > 9 else "9-THZxMST"
            for r_str in s.get("trained_regions", []):
                rp = EVAL_DIR / crop / subdir / r_str / "region_summary.json"
                if rp.exists():
                    rr = json.load(open(rp))
                    rows.append({"crop": crop,
                                 "pipeline": subdir.replace("regional", "R").replace("_weighted", "W"),
                                 "regime": regime, "region": r_str,
                                 "n_samples": rr.get("n_samples", 0),
                                 "global_r2": rr.get("global_r2") or rr.get("global_nrmse"),
                                 "combined_r2": rr.get("combined_r2") or rr.get("combined_r2_mean")})
    if not rows:
        print("S7: no wh/mai regional summaries found, skipping")
        return
    pd.DataFrame(rows).to_csv(out / "table_s7_koppen_comparison.csv", index=False)
    print(f"S7: saved ({len(rows)} regions)")


def table_s8_pred_vs_hist(out):
    """Table S8 — per-crop predicted (2019-2024) vs historical (2014-2018) mean."""
    if not BASE_AGG_DIR.exists():
        print("S8: BASE_AGG_DIR missing, skipping")
        return
    rows = []
    for crop in ALL_CROPS:
        hist_vals = [pd.read_csv(f)["waterstress"].sum()
                     for y in range(2014, 2019)
                     if (f := BASE_AGG_DIR / str(y) / f"{crop}_country_{y}.csv").exists()]
        hist_mean = sum(hist_vals) / len(hist_vals) if hist_vals else 0
        pred_f = PRED_CSV_DIR / f"{crop}.csv"
        pred_mean = pd.read_csv(pred_f).groupby("year")["y_pred"].sum().mean() if pred_f.exists() else 0
        rows.append({"crop": crop,
                     "pipeline": "G+R+E" if crop in UNWEIGHTED else "GW+RW+E",
                     "hist_mean_14_18": hist_mean, "pred_mean_19_24": pred_mean,
                     "ratio": pred_mean / hist_mean if hist_mean > 0 else 0})
    pd.DataFrame(rows).sort_values("hist_mean_14_18", ascending=False).to_csv(
        out / "table_s8_prediction_vs_historical.csv", index=False)
    print(f"S8: saved ({len(rows)} crops)")


def tables_validation(out):
    """Tables 5, 6, S9, S10 — ranking, climate events, temporal trend, short-window."""
    if not VALIDATION_DIR.exists():
        print("Tables 5/6/S9/S10: VALIDATION_DIR missing, skipping")
        return
    v = VALIDATION_DIR

    # Table 5 — country ranking (hist vs pred)
    if (v / "dir1a_hist_ranking.csv").exists() and (v / "dir1b_pred_ranking.csv").exists():
        dir1a = pd.read_csv(v / "dir1a_hist_ranking.csv").rename(
            columns={"rho": "hist_rho", "r": "hist_r", "n": "hist_n"})
        dir1b = pd.read_csv(v / "dir1b_pred_ranking.csv", index_col=0)
        t5 = dir1a[["crop", "hist_rho", "hist_n"]].copy()
        pred_map = dir1b.to_dict() if isinstance(dir1b, pd.Series) else dir1b.iloc[:, 0].to_dict()
        t5["pred_rho"] = t5["crop"].map(pred_map)
        t5["diff"] = t5["pred_rho"] - t5["hist_rho"]
        t5["hist_tier"] = t5["hist_rho"].apply(_tier)
        t5["pred_tier"] = t5["pred_rho"].apply(_tier)
        t5.sort_values("hist_rho", ascending=False).to_csv(out / "table5_country_ranking.csv", index=False)
        print("Table 5: saved")

    # Table 6 — climate event detection
    if (v / "dir3_climate_events.csv").exists():
        dir3 = pd.read_csv(v / "dir3_climate_events.csv")
        indirect = ["2020 East Africa Locusts", "2019 Indian Monsoon Delay",
                    "2022 China Heat Wave", "2023 El Nino (India)"]
        dir3["event_type"] = dir3["event"].apply(
            lambda x: "Indirect/Averaged" if x in indirect else "Direct Water Stress")
        dir3.to_csv(out / "table6_climate_events.csv", index=False)
        print("Table 6: saved")

    # Table S9 — global temporal trend
    if (v / "dir2a_temporal_full.csv").exists():
        pd.read_csv(v / "dir2a_temporal_full.csv").sort_values("rho", ascending=False).to_csv(
            out / "table_s9_temporal_trend.csv", index=False)
        print("Table S9: saved")

    # Table S10 — short-window baseline
    if (v / "dir2b_short_window.csv").exists():
        pd.read_csv(v / "dir2b_short_window.csv").to_csv(out / "table_s10_short_window.csv", index=False)
        print("Table S10: saved")


def tables_perf_pred(out):
    """Tables 2, 3, 4, S4, S5 — grid/country performance, prediction summary, per-country, transition."""
    unw_path = EVAL_DIR / "country_eval_summary_global.csv"
    wt_path = EVAL_DIR / "country_eval_comparison_weighted.csv"
    if not (unw_path.exists() and wt_path.exists()):
        print("Tables 2/3: country_eval CSVs missing, skipping perf tables")
    else:
        df_unw = pd.read_csv(unw_path).set_index("crop")
        df_wt = pd.read_csv(wt_path).set_index("crop")

        t2 = []
        for crop in sorted(df_unw.index):
            row = {"crop": crop, "G_r2": df_unw.loc[crop, "grid_r2"]}
            if crop in df_wt.index:
                row["GW_r2"] = df_wt.loc[crop].get("grid_GW")
                row["GW_RW_r2"] = df_wt.loc[crop].get("grid_GWR")
            if crop in UNWEIGHTED:
                row["selected"], row["best_grid_r2"] = "G+R", row.get("G_r2")
            else:
                row["selected"], row["best_grid_r2"] = "GW+RW", row.get("GW_RW_r2")
            t2.append(row)
        pd.DataFrame(t2).sort_values("best_grid_r2", ascending=False, na_position="last").to_csv(
            out / "table2_grid_performance.csv", index=False)
        print("Table 2: saved")

        t3 = []
        for crop in sorted(df_unw.index):
            row = {"crop": crop, "ctry_GR": df_unw.loc[crop, "country_year_r2"],
                   "corr": df_unw.loc[crop, "correlation"], "n_countries": df_unw.loc[crop, "n_countries"]}
            if crop in df_wt.index:
                row["ctry_GW"] = df_wt.loc[crop].get("ctry_GW")
                row["ctry_GWR"] = df_wt.loc[crop].get("ctry_GWR")
            if crop in UNWEIGHTED:
                row["selected"], row["best_ctry_r2"] = "G+R", row["ctry_GR"]
            else:
                row["selected"], row["best_ctry_r2"] = "GW+RW", row.get("ctry_GWR", row["ctry_GR"])
            t3.append(row)
        pd.DataFrame(t3).sort_values("best_ctry_r2", ascending=False, na_position="last").to_csv(
            out / "table3_country_performance.csv", index=False)
        print("Table 3: saved")

    if not CROPS_27_CSV.exists():
        print("Tables 4/S4/S5: crops_27_1990_2024.csv missing, skipping")
        return
    df_cwss = pd.read_csv(CROPS_27_CSV)

    # Table 4 — prediction summary (historical from BASE_AGG if available)
    yearly_hist = {}
    if BASE_AGG_DIR.exists():
        for year in range(2014, 2019):
            total = sum(pd.read_csv(f)["waterstress"].sum()
                        for f in BASE_AGG_DIR.glob(f"{year}/*_country_{year}.csv"))
            yearly_hist[year] = total
    yearly_pred = df_cwss[df_cwss["year"].between(2019, 2024)].groupby("year")["value"].sum().to_dict()
    all_yearly = {**yearly_hist, **yearly_pred}
    base_2018 = yearly_hist.get(2018, 1) or 1
    pd.DataFrame([{"year": y, "global_total": all_yearly[y],
                   "vs_2018_pct": (all_yearly[y] - base_2018) / base_2018 * 100,
                   "source": "historical" if y <= 2018 else "predicted"}
                  for y in sorted(all_yearly)]).to_csv(out / "table4_prediction_summary.csv", index=False)
    print("Table 4: saved")

    # Table S4 — per-country predictions (top 30)
    pred = df_cwss[df_cwss["year"].between(2019, 2024)]
    country_yearly = pred.groupby(["country", "year"])["value"].sum().unstack(fill_value=0)
    country_mean = country_yearly.mean(axis=1).sort_values(ascending=False)
    top30 = country_mean.head(30).index.tolist()
    s4 = country_yearly.loc[top30].copy()
    s4["mean"] = s4.mean(axis=1)
    s4["pct_global"] = s4["mean"] / country_mean.sum() * 100
    s4["top_crops"] = pd.Series({
        c: ", ".join(pred[pred["country"] == c].groupby("crop")["value"].sum()
                     .sort_values(ascending=False).head(3).index.tolist()) for c in top30})
    s4.sort_values("mean", ascending=False).to_csv(out / "table_s4_country_predictions.csv")
    print("Table S4: saved")

    # Table S5 — 2018->2019 transition (crop + country)
    crop_2018 = df_cwss[df_cwss["year"] == 2018].groupby("crop")["value"].sum()
    crop_2019 = df_cwss[df_cwss["year"] == 2019].groupby("crop")["value"].sum()
    crop_change = ((crop_2019 - crop_2018) / crop_2018 * 100).dropna().sort_values(ascending=False)
    pd.DataFrame([{"crop": c, "val_2018": crop_2018.get(c, 0), "val_2019": crop_2019.get(c, 0),
                   "change_pct": crop_change[c]} for c in crop_change.index]).to_csv(
        out / "table_s5a_crop_transition.csv", index=False)
    c_2018 = df_cwss[df_cwss["year"] == 2018].groupby("country")["value"].sum()
    c_2019 = df_cwss[df_cwss["year"] == 2019].groupby("country")["value"].sum()
    c_abs = (c_2019 - c_2018).dropna().sort_values(ascending=False)
    c_pct = ((c_2019 - c_2018) / c_2018 * 100).dropna()
    s5b = ([{"country": c, "abs_change": c_abs[c], "pct_change": c_pct.get(c, 0), "direction": "increase"}
            for c in c_abs.head(10).index]
           + [{"country": c, "abs_change": c_abs[c], "pct_change": c_pct.get(c, 0), "direction": "decrease"}
              for c in c_abs.tail(10).index])
    pd.DataFrame(s5b).to_csv(out / "table_s5b_country_transition.csv", index=False)
    print("Table S5: saved")


def table_s12_seasonality(out):
    """Table S12 — monthly seasonality, historical vs predicted (per crop)."""
    from scipy import stats
    if not (BASE_AGG_DIR.exists() and PRED_COUNTRY.exists()):
        print("S12: BASE_AGG_DIR or PRED_COUNTRY missing, skipping")
        return

    hist_monthly = []
    for year in range(2014, 2019):
        for crop in ALL_CROPS:
            f = BASE_AGG_DIR / str(year) / f"{crop}_country_{year}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            if "time" not in df.columns:
                continue
            df["month"] = pd.to_datetime(df["time"]).dt.month
            m = df.groupby("month")["waterstress"].sum().reset_index()
            m["crop"] = crop
            hist_monthly.append(m)
    pred_monthly = []
    for crop in ALL_CROPS:
        f = PRED_COUNTRY / f"{crop}_country.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["month"] = pd.to_datetime(df["time"]).dt.month
        m = df.groupby("month")["waterstress"].sum().reset_index()
        m["crop"] = crop
        pred_monthly.append(m)
    if not hist_monthly or not pred_monthly:
        print("S12: no monthly data, skipping")
        return

    df_h = pd.concat(hist_monthly, ignore_index=True)
    df_p = pd.concat(pred_monthly, ignore_index=True)
    rows = []
    for crop in ALL_CROPS:
        h = df_h[df_h["crop"] == crop].groupby("month")["waterstress"].mean()
        p = df_p[df_p["crop"] == crop].groupby("month")["waterstress"].mean()
        common = sorted(set(h.index) & set(p.index))
        if len(common) < 4:
            continue
        rho_c, _ = stats.spearmanr([h[m] for m in common], [p[m] for m in common])
        peak_h, peak_p = (h.idxmax() if len(h) else 0), (p.idxmax() if len(p) else 0)
        rows.append({"crop": crop, "seasonal_rho": rho_c, "peak_month_hist": peak_h,
                     "peak_month_pred": peak_p, "peak_match": "yes" if abs(peak_h - peak_p) <= 1 else "no"})
    pd.DataFrame(rows).sort_values("seasonal_rho", ascending=False).to_csv(
        out / "table_s12_monthly_seasonality.csv", index=False)
    print(f"S12: saved ({len(rows)} crops)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = [
        table_s1_grid_metrics, table_s2_adoption, table_s3_weighted_comparison,
        table_s6_feature_importance, table_s7_koppen, table_s8_pred_vs_hist,
        tables_validation, tables_perf_pred, table_s12_seasonality,
    ]
    for fn in tables:
        try:
            fn(OUT_DIR)
        except Exception as e:
            print(f"{fn.__name__}: FAILED -- {e}")
    print(f"\nTables written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
