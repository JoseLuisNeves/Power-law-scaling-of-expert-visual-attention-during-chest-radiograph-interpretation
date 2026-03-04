#!/usr/bin/env python3
"""
Experiment 4: Geometry-corrected allocation residuals versus AOI magnitude
              and latency baselines for within-case diagnostic difficulty
              discrimination.

Primary claim:
  Within a case, ellipses that receive more attention than predicted by the
  population area scaling law are enriched for diagnostically difficult
  findings (as defined by documented high miss rates in the chest radiography
  malpractice literature), beyond what is captured by fixation magnitude or
  detection latency baselines.

Difficulty operationalisation:
  Findings are classified as difficult if they appear among the most frequent
  causes of missed diagnosis and malpractice claims in chest radiography,
  as documented in Gefter et al. (CHEST 2023) and Baker et al. (J Thorac
  Imaging 2013). These sources identify missed nodules/lung cancer,
  pneumothorax, mediastinal abnormalities, thoracic fractures, and pleural
  effusion as the predominant error categories. All remaining labeled
  findings form the easier comparison group by exclusion.

Design:
  1. Leak-free 50/50 split at case level.
     Gamma fitted on fit split using Negative Binomial count model (same
     model as exp1a) via fit_negative_binomial_to_density().
  2. Downstream dataset: all labeled ellipses with a known difficulty label.
     Unvisited ellipses included for n_fix / dwell / TTFF baselines.
     Allocation residual delta defined only for n_fix >= DELTA_MIN_FIX.
  3. All metrics compared under the same within-case ranking task (AUC).
  4. Validity: within-case n_fix permutation + label-shuffle control.

Dependencies:
  exp1a_fit_scaling_laws.fit_negative_binomial_to_density
  reflacxloader.ReflacxLoader
  statsmodels, numpy, pandas, sklearn, scipy, matplotlib
"""

import os
import json
import warnings
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

from reflacxloader import ReflacxLoader
from exp1a_fit_scaling_laws import fit_negative_binomial_to_density

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "axes.linewidth": 1.2, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_TEAL  = "#AFEEEE"; C_PLUM = "#DC92EF"
C_ROSE  = "#c76076"; C_DUSK = "#93D8D8"; C_BLACK = "#111111"

# ── Config ─────────────────────────────────────────────────────────────────────
PERIOD                = "pre-reporting"
SEED                  = 42
SCALING_FIT_FRAC      = 0.50
MIN_ELLIPSES          = 2        # min labeled ellipses per case
MIN_FIX_FIT           = 5        # min fixations for scaling-law fit ellipses
DELTA_MIN_FIX_PRIMARY = 5        # min fixations to compute delta (primary)
DELTA_MIN_FIX_SENS    = (3, 5)   # sensitivity sweep
EPS                   = 1e-6
N_BOOT                = 2000
N_PERM                = 1000
N_LABEL_SHUFFLE       = 500
OUT_DIR               = "./results/exp4"

# ── Difficulty classification ──────────────────────────────────────────────────
# Findings with documented high miss rates and malpractice claims in chest
# radiography, per Gefter et al. (CHEST 2023) and Baker et al. (J Thorac
# Imaging 2013). Cited sources identify these as the predominant categories
# of missed diagnosis: nodules/lung cancer (43% of chest imaging malpractice
# claims), pneumothorax, mediastinal abnormalities, thoracic fractures
DIFFICULT = {
    "lung nodule or mass", "nodule", "mass",
    "groundglass opacity",
    "pneumothorax",
    "interstitial lung disease",
    "acute fracture", "fracture",
    "abnormal mediastinal contour", "wide mediastinum",
    "fibrosis",
    "pleural thickening",
    "emphysema",
}

EASIER = {
    "enlarged cardiac silhouette", "pulmonary edema",
    "hiatal hernia", "pleural effusion",
    "atelectasis", 
    "consolidation",
    "pleural abnormality", "high lung volume / emphysema",
    "enlarged hilum", "airway wall thickening",
    "other",
}

def ellipse_difficulty(labels):
    """
    Returns 1 if any label is in the difficult set, 0 if any label is in
    the easier set, and None if no label maps to either group.
    An ellipse with at least one difficult label is classified as difficult.
    """
    if isinstance(labels, str):
        labels = [labels]
    normalised = [str(l).strip().lower() for l in labels]
    if any(l in DIFFICULT for l in normalised):
        return 1
    if any(l in EASIER for l in normalised):
        return 0
    return None


# ── Scaling-law fit (NB via exp1a) ────────────────────────────────────────────
def collect_fit_ellipses(loader, fit_cases: List[Tuple], min_fix_fit: int):
    """
    Collect EllipseAttention objects from fit_cases only, filtered by
    min_fix_fit. Passed directly to fit_negative_binomial_to_density().
    """
    out = []
    for pid, sid in fit_cases:
        try:
            ells = loader.get_ellipses_attention(
                pid, sid, period=PERIOD, relate_to_mention=None)
        except Exception:
            continue
        out.extend(
            e for e in ells
            if len(e.fixations) >= min_fix_fit and e.ellipse.labels
        )
    return out


# ── Data collection ───────────────────────────────────────────────────────────
def collect_downstream_ellipses(loader, downstream_cases: Set[Tuple]):
    """
    All labeled ellipses in downstream cases regardless of fixation count.
    Unvisited ellipses: dwell=0, ttff=case_duration (right-censored).
    delta_centered will be NaN for these rows.
    Only ellipses whose labels map to either DIFFICULT or EASIER are retained.
    """
    rows = []
    for pid, studies in loader.transcripts_dict.items():
        for sid in studies:
            if (pid, sid) not in downstream_cases:
                continue
            try:
                ells    = loader.get_ellipses_attention(
                    pid, sid, period=PERIOD, relate_to_mention=None)
                all_fix = loader.get_study_fixations(pid, sid, period=PERIOD)
            except Exception:
                continue

            labeled = [e for e in ells if e.ellipse.labels]
            if len(labeled) < MIN_ELLIPSES:
                continue

            if all_fix:
                case_t0  = float(min(f.start_time for f in all_fix))
                case_t1  = float(max(f.end_time   for f in all_fix))
                case_dur = max(0.0, case_t1 - case_t0)
            else:
                case_t0 = case_dur = np.nan

            for e in labeled:
                difficulty = ellipse_difficulty(e.ellipse.labels)
                if difficulty is None:
                    continue
                a = float(e.ellipse.area)
                if a <= 0:
                    continue

                n_fix = int(len(e.fixations))
                dwell = (float(sum(f.duration for f in e.fixations))
                         if n_fix > 0 else 0.0)

                if n_fix > 0 and np.isfinite(case_t0):
                    ttff = max(0.0,
                               float(min(f.start_time for f in e.fixations))
                               - case_t0)
                elif np.isfinite(case_dur):
                    ttff = float(case_dur)
                else:
                    ttff = np.nan

                rows.append({
                    "case_id":      f"{pid}__{sid}",
                    "is_difficult": int(difficulty),
                    "area":         a,
                    "n_fix":        n_fix,
                    "dwell":        dwell,
                    "ttff":         ttff,
                    "log_nfix":     float(np.log(n_fix + 1)),
                    "log_dwell":    float(np.log(dwell + 1)),
                    "log_ttff":     (float(np.log(ttff + 1))
                                     if ttff is not None and np.isfinite(ttff)
                                     else np.nan),
                })

    return pd.DataFrame(rows)


# ── Allocation score ───────────────────────────────────────────────────────────
def add_delta_centered(df, gamma: float, delta_min_fix: int):
    out   = df.copy()
    alpha = gamma + 1.0

    out["w_exp"]  = out["area"] ** alpha
    out["W_case"] = out.groupby("case_id")["w_exp"].transform("sum")
    out["N_case"] = out.groupby("case_id")["n_fix"].transform("sum")

    mask  = out["n_fix"] >= int(delta_min_fix)
    f_obs = np.where(mask, out["n_fix"] / out["N_case"].clip(lower=1), np.nan)
    f_exp = out["w_exp"] / out["W_case"].replace(0, np.nan)

    out["delta"] = np.where(
        mask,
        np.log((f_obs + EPS) / (f_exp.values + EPS)),
        np.nan)
    out["delta_centered"] = out.groupby("case_id")["delta"].transform(
        lambda x: x - np.nanmedian(x.values))
    return out


# ── Within-case standardisation ───────────────────────────────────────────────
def zscore_within_case(df, cols):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = out.groupby("case_id")[c].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12))
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────
def bootstrap_mean_ci(vals, seed=SEED, n_boot=N_BOOT):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    rng  = np.random.RandomState(seed)
    boot = [rng.choice(vals, size=len(vals), replace=True).mean()
            for _ in range(n_boot)]
    boot = np.asarray(boot)
    return {"mean":    float(vals.mean()),
            "ci_low":  float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "n":       int(len(vals))}

def case_auc_series(df, score_col, target_col="is_difficult"):
    rows = []
    for cid, g in df.groupby("case_id"):
        g2 = g.dropna(subset=[score_col, target_col]).copy()
        if len(g2) < 2:
            continue
        y = g2[target_col].values.astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        s = g2[score_col].values.astype(float)
        if not np.isfinite(s).all():
            continue
        try:
            rows.append({"case_id": cid, "auc": roc_auc_score(y, s)})
        except Exception:
            continue
    return pd.DataFrame(rows)


# ── Validity tests ─────────────────────────────────────────────────────────────
def permute_nfix_within_case(df_raw, gamma: float, delta_min_fix: int):
    rng = np.random.RandomState(SEED)
    df0 = add_delta_centered(df_raw, gamma, delta_min_fix)
    df0 = zscore_within_case(df0, ["delta_centered"])
    obs = case_auc_series(df0, "delta_centered")["auc"].mean()

    case_ids    = df_raw["case_id"].values
    idx_by_case = {c: np.where(case_ids == c)[0] for c in np.unique(case_ids)}

    null = []
    for _ in range(N_PERM):
        dfp  = df_raw.copy()
        nfix = dfp["n_fix"].values.copy()
        for idx in idx_by_case.values():
            if len(idx) > 1:
                nfix[idx] = nfix[rng.permutation(idx)]
        dfp["n_fix"] = nfix
        dfp = add_delta_centered(dfp, gamma, delta_min_fix)
        dfp = zscore_within_case(dfp, ["delta_centered"])
        a   = case_auc_series(dfp, "delta_centered")
        if len(a) > 0 and a["auc"].notna().any():
            null.append(float(a["auc"].mean()))

    null = np.asarray(null, dtype=float)
    p    = float((null >= obs).mean()) if len(null) else np.nan
    return {"obs":               float(obs),
            "p":                 p,
            "null_mean":         float(np.nanmean(null)) if len(null) else np.nan,
            "null_std":          float(np.nanstd(null))  if len(null) else np.nan,
            "null":              null.tolist(),
            "n_perm_effective":  int(len(null))}

def label_shuffle_control(df, score_col, n_perm=N_LABEL_SHUFFLE):
    rng  = np.random.RandomState(SEED + 1)
    base = df.dropna(subset=[score_col, "is_difficult"]).copy()
    if len(base) == 0:
        return {"null_mean": np.nan, "null_std": np.nan, "null_p95": np.nan, "n": 0}
    null = []
    for _ in range(n_perm):
        dfp = base.copy()
        dfp["is_difficult"] = dfp.groupby("case_id")["is_difficult"].transform(
            lambda x: rng.permutation(x.values))
        a = case_auc_series(dfp, score_col)
        if len(a) > 0:
            null.append(float(a["auc"].mean()))
    null = np.asarray(null, dtype=float)
    return {"null_mean": float(np.nanmean(null)) if len(null) else np.nan,
            "null_std":  float(np.nanstd(null))  if len(null) else np.nan,
            "null_p95":  float(np.nanpercentile(null, 95)) if len(null) else np.nan,
            "n":         int(len(null))}


# ── B. OOF grouped logistic ───────────────────────────────────────────────────
def oof_logit(df, feature_cols, seed=SEED):
    sub = df.dropna(subset=feature_cols + ["is_difficult"]).copy()
    sub = sub.dropna(subset=feature_cols)
    if len(sub) == 0:
        return {"auc_oof": np.nan, "n": 0}
    y      = sub["is_difficult"].values
    groups = sub["case_id"].values
    if y.sum() == 0 or y.sum() == len(y):
        return {"auc_oof": np.nan, "n": len(sub)}
    gkf   = GroupKFold(n_splits=5)
    probs = np.zeros(len(sub))
    for tri, tei in gkf.split(sub, y, groups):
        if y[tri].sum() == 0 or y[tri].sum() == len(y[tri]):
            continue
        clf = LogisticRegression(class_weight="balanced",
                                  max_iter=500, random_state=seed)
        clf.fit(sub.iloc[tri][feature_cols].values, y[tri])
        probs[tei] = clf.predict_proba(
            sub.iloc[tei][feature_cols].values)[:, 1]
    mask = probs > 0
    auc  = float(roc_auc_score(y[mask], probs[mask])) if mask.sum() > 0 else np.nan
    return {"auc_oof": auc, "n": int(mask.sum())}


# ── C. Incremental value of delta beyond dwell+ttff ──────────────────────────
def incremental_delta(df, seed=SEED):
    """In-sample per-case logistic AUC for baseline vs full model."""
    base_cols = ["log_dwell", "log_ttff"]
    full_cols  = ["delta_centered", "log_dwell", "log_ttff"]
    sub = df.dropna(subset=full_cols + ["is_difficult"]).copy()

    def _per_case_auc(sub, cols):
        rows = []
        for cid, g in sub.groupby("case_id"):
            y = g["is_difficult"].values
            if y.sum() == 0 or y.sum() == len(y):
                continue
            X = g[cols].values
            if not np.isfinite(X).all():
                continue
            try:
                clf = LogisticRegression(class_weight="balanced",
                                          max_iter=200, random_state=seed)
                clf.fit(X, y)
                rows.append({"case_id": cid,
                              "auc": roc_auc_score(y, clf.predict_proba(X)[:, 1])})
            except Exception:
                continue
        return pd.DataFrame(rows)

    ab   = _per_case_auc(sub, base_cols).set_index("case_id")["auc"]
    af   = _per_case_auc(sub, full_cols).set_index("case_id")["auc"]
    comm = ab.index.intersection(af.index)
    b, f = ab.loc[comm].values, af.loc[comm].values
    diff = f - b

    stat, p = wilcoxon(f, b, alternative="greater")
    return {"n_cases":       len(comm),
            "baseline_auc":  bootstrap_mean_ci(b,    seed=seed),
            "full_auc":      bootstrap_mean_ci(f,    seed=seed + 1),
            "delta_auc":     bootstrap_mean_ci(diff, seed=seed + 2),
            "wilcoxon_stat": float(stat),
            "wilcoxon_p":    float(p)}


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_auc_bar(auc_summary, out_png):
    order  = ["delta_centered", "log_dwell", "log_nfix", "log_ttff"]
    labels = {"delta_centered": "Allocation residual",
              "log_dwell":      "Dwell time",
              "log_nfix":       "Fixation count",
              "log_ttff":       "Time to first fixation"}
    colors = {"delta_centered": C_PLUM, "log_dwell": C_DUSK,
              "log_nfix": C_TEAL, "log_ttff": C_ROSE}
    names  = [m for m in order if m in auc_summary]
    means  = [auc_summary[m]["mean"]    for m in names]
    lo     = [auc_summary[m]["ci_low"]  for m in names]
    hi     = [auc_summary[m]["ci_high"] for m in names]
    xerr   = [[m - l for m, l in zip(means, lo)],
               [h - m for m, h in zip(means, hi)]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([labels[m] for m in names], means, xerr=xerr,
            color=[colors[m] for m in names],
            alpha=0.85, capsize=5, edgecolor="white", height=0.65)
    ax.axvline(0.5, color=C_BLACK, linestyle="--", linewidth=1.2, alpha=0.6)
    for y_pos, (m, h) in enumerate(zip(means, hi)):
        ax.text(h + 0.005, y_pos, f"{m:.3f}", va="center", fontsize=11)
    ax.set_xlabel("Mean within-case AUC (difficult vs easier)")
    ax.set_title("Diagnostic Difficulty Discrimination Within Case")
    ax.set_xlim(left=0.3)
    ax.grid(axis="x", alpha=0.15)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_perm_hist(null, obs, p, out_png):
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=40, alpha=0.85, color=C_TEAL, edgecolor="white")
    ax.axvline(obs, color=C_ROSE, linewidth=2,
               label=f"Obs={obs:.4f}  p={p:.4f}")
    ax.set_xlabel("Mean within-case AUC under permutation")
    ax.set_ylabel("Count")
    ax.set_title("Permutation null: shuffle n_fix within case")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_incremental(incr, out_png):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    means = [incr["baseline_auc"]["mean"], incr["full_auc"]["mean"]]
    lo    = [incr["baseline_auc"]["ci_low"],  incr["full_auc"]["ci_low"]]
    hi    = [incr["baseline_auc"]["ci_high"], incr["full_auc"]["ci_high"]]
    yerr  = [[m - l for m, l in zip(means, lo)],
             [h - m for m, h in zip(means, hi)]]
    ax.bar(["dwell+TTFF", "delta+dwell+TTFF"], means, yerr=yerr,
           color=[C_TEAL, C_PLUM], alpha=0.82, capsize=6, edgecolor="white")
    ax.axhline(0.5, color=C_BLACK, linestyle="--", linewidth=1.2, alpha=0.5)
    ax.set_ylabel("Mean per-case AUC (difficult vs easier)")
    d = incr["delta_auc"]
    ax.set_title(f"Incremental value of delta  (n={incr['n_cases']} cases)\n"
                 f"ΔAUC={d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]"
                 f"  Wilcoxon p={incr['wilcoxon_p']:.4f}")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Single run ─────────────────────────────────────────────────────────────────
def run_one(delta_min_fix: int, make_plots: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Leak-free 50/50 split ──────────────────────────────────────────
    loader = ReflacxLoader()
    loader.load_jsons()
    eligible = []
    for pid, studies in loader.transcripts_dict.items():
        for sid in studies:
            try:
                n = sum(1 for e in loader.get_ellipses_attention(
                    pid, sid, period=PERIOD, relate_to_mention=None)
                    if e.ellipse.labels)
                if n >= MIN_ELLIPSES:
                    eligible.append((pid, sid))
            except Exception:
                continue

    rng       = np.random.RandomState(SEED + 1000 * int(delta_min_fix))
    rng.shuffle(eligible)
    n_fit      = int(np.floor(SCALING_FIT_FRAC * len(eligible)))
    fit_cases  = eligible[:n_fit]
    downstream = set(eligible[n_fit:])

    # ── 2. NB scaling-law fit on fit split ───────────────────────────────
    fit_ellipses = collect_fit_ellipses(loader, fit_cases, MIN_FIX_FIT)
    if len(fit_ellipses) < 50:
        raise RuntimeError(f"Too few fit ellipses: n={len(fit_ellipses)}")
    nb_params  = fit_negative_binomial_to_density(fit_ellipses)
    gamma      = nb_params.slope
    intercept  = nb_params.intercept

    # ── 3. Collect + score downstream ellipses ───────────────────────────
    df_raw    = collect_downstream_ellipses(loader, downstream)
    df_scored = add_delta_centered(df_raw, gamma, delta_min_fix)
    df_z      = zscore_within_case(df_scored,
                    ["delta_centered", "log_nfix", "log_dwell", "log_ttff"])

    n_difficult = int((df_z["is_difficult"] == 1).sum())
    n_easier    = int((df_z["is_difficult"] == 0).sum())
    unvis       = int((df_z["n_fix"] == 0).sum())

    print(f"\n{'─'*70}")
    print(f"  delta_min_fix={delta_min_fix}  gamma={gamma:.4f}  "
          f"intercept={intercept:.4f}")
    print(f"  NB fit: n_ellipses={len(fit_ellipses)}  "
          f"gamma_CI=[{nb_params.gamma_ci[0]:.4f},{nb_params.gamma_ci[1]:.4f}]  "
          f"R2={nb_params.r2:.3f}")
    print(f"  downstream: ellipses={len(df_z)}  "
          f"cases={df_z['case_id'].nunique()}")
    print(f"    Difficult: {n_difficult} ({100*n_difficult/len(df_z):.1f}%)")
    print(f"    Easier:    {n_easier} ({100*n_easier/len(df_z):.1f}%)")
    print(f"  Unvisited: {unvis} ({100*unvis/len(df_z):.1f}%)")

    # ── A. Within-case AUC ───────────────────────────────────────────────
    print("\nA. Within-case AUC (difficult vs easier):")
    metrics     = ["delta_centered", "log_nfix", "log_dwell", "log_ttff"]
    auc_summary = {}
    for m in metrics:
        auc_df = case_auc_series(df_z, m)
        boot   = bootstrap_mean_ci(auc_df["auc"].values, seed=SEED + 10)
        auc_summary[m] = {**boot, "n_cases": int(len(auc_df))}
        sig = "✓" if boot["ci_low"] > 0.5 else "~"
        print(f"  {m:<22} AUC={boot['mean']:.4f} "
              f"[{boot['ci_low']:.4f},{boot['ci_high']:.4f}]  "
              f"n={len(auc_df)}  {sig}")

    perm = permute_nfix_within_case(df_raw, gamma, delta_min_fix)
    print(f"  Perm (delta): obs={perm['obs']:.4f}  p={perm['p']:.4f}  "
          f"null mu={perm['null_mean']:.4f}+/-{perm['null_std']:.4f}")

    lsc = label_shuffle_control(df_z, "delta_centered")
    print(f"  Label-shuffle: null mu={lsc['null_mean']:.4f}  "
          f"p95={lsc['null_p95']:.4f}")

    # ── B. OOF grouped logistic ──────────────────────────────────────────
    print("\nB. OOF grouped logistic (secondary):")
    oof_specs = [
        ("delta",                 ["delta_centered"]),
        ("nfix",                  ["log_nfix"]),
        ("dwell",                 ["log_dwell"]),
        ("ttff",                  ["log_ttff"]),
        ("delta+dwell",           ["delta_centered", "log_dwell"]),
        ("delta+ttff",            ["delta_centered", "log_ttff"]),
        ("delta+nfix",            ["delta_centered", "log_nfix"]),
        ("delta+dwell+ttff",      ["delta_centered", "log_dwell", "log_ttff"]),
        ("delta+nfix+dwell+ttff", ["delta_centered", "log_nfix",
                                    "log_dwell", "log_ttff"]),
    ]
    oof_res = {}
    for name, feats in oof_specs:
        r = oof_logit(df_z, feats)
        auc_str = f"{r['auc_oof']:.4f}" if r["auc_oof"] is not None \
                  and np.isfinite(r["auc_oof"]) else "  nan "
        print(f"  {name:<28} AUC_oof={auc_str}  n={r['n']}")
        oof_res[name] = r

    # ── C. Incremental value ─────────────────────────────────────────────
    print("\nC. Incremental value of delta beyond dwell+TTFF (in-sample per-case):")
    incr = incremental_delta(df_z)
    d    = incr["delta_auc"]
    print(f"  baseline (dwell+ttff): {incr['baseline_auc']['mean']:.4f} "
          f"[{incr['baseline_auc']['ci_low']:.4f},"
          f"{incr['baseline_auc']['ci_high']:.4f}]")
    print(f"  full (delta+dwell+ttff): {incr['full_auc']['mean']:.4f} "
          f"[{incr['full_auc']['ci_low']:.4f},"
          f"{incr['full_auc']['ci_high']:.4f}]")
    print(f"  ΔAUC={d['mean']:+.4f} [{d['ci_low']:+.4f},{d['ci_high']:+.4f}]  "
          f"Wilcoxon p={incr['wilcoxon_p']:.4f}  n={incr['n_cases']}")

    # ── Plots ────────────────────────────────────────────────────────────
    out_dir = os.path.join(OUT_DIR, f"dminfix{delta_min_fix}")
    os.makedirs(out_dir, exist_ok=True)
    if make_plots:
        plot_auc_bar(auc_summary, os.path.join(out_dir, "auc_bar.png"))
        plot_perm_hist(perm["null"], perm["obs"], perm["p"],
                       os.path.join(out_dir, "perm_hist.png"))
        plot_incremental(incr, os.path.join(out_dir, "incremental.png"))
        np.save(os.path.join(out_dir, "perm_null.npy"),
                np.asarray(perm["null"], dtype=float))

    # ── Save ─────────────────────────────────────────────────────────────
    df_scored.to_csv(os.path.join(out_dir, "ellipse_features.csv"), index=False)

    payload = {
        "config": {
            "period":           PERIOD,
            "seed":             SEED,
            "scaling_fit_frac": SCALING_FIT_FRAC,
            "min_fix_fit":      MIN_FIX_FIT,
            "delta_min_fix":    int(delta_min_fix),
            "min_ellipses":     MIN_ELLIPSES,
            "difficult_labels": sorted(DIFFICULT),
            "easier_labels":    sorted(EASIER),
        },
        "scaling_fit": {
            "method":         "negative_binomial",
            "gamma":          gamma,
            "intercept":      intercept,
            "gamma_ci":       list(nb_params.gamma_ci),
            "r2":             nb_params.r2,
            "n_fit_ellipses": len(fit_ellipses),
            "n_fit_cases":    len(fit_cases),
        },
        "downstream": {
            "n_ellipses":   int(len(df_z)),
            "n_cases":      int(df_z["case_id"].nunique()),
            "n_difficult":  n_difficult,
            "n_easier":     n_easier,
            "unvisited_n":  unvis,
        },
        "A_within_case_auc":     auc_summary,
        "A_permutation_delta":   {k: v for k, v in perm.items() if k != "null"},
        "A_label_shuffle_delta": lsc,
        "B_oof":                 oof_res,
        "C_incremental":         incr,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    return payload


# ── Sensitivity sweep ──────────────────────────────────────────────────────────
def run_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for dmf in DELTA_MIN_FIX_SENS:
        res = run_one(delta_min_fix=dmf,
                      make_plots=(int(dmf) == int(DELTA_MIN_FIX_PRIMARY)))
        a = res["A_within_case_auc"]
        b = res["B_oof"]
        c = res["C_incremental"]
        rows.append({
            "delta_min_fix":        int(dmf),
            "gamma":                float(res["scaling_fit"]["gamma"]),
            "gamma_ci_lo":          float(res["scaling_fit"]["gamma_ci"][0]),
            "gamma_ci_hi":          float(res["scaling_fit"]["gamma_ci"][1]),
            "n_cases":              int(res["downstream"]["n_cases"]),
            "n_ellipses":           int(res["downstream"]["n_ellipses"]),
            "n_difficult":          int(res["downstream"]["n_difficult"]),
            "n_easier":             int(res["downstream"]["n_easier"]),
            "auc_delta":            float(a["delta_centered"]["mean"]),
            "auc_dwell":            float(a["log_dwell"]["mean"]),
            "auc_nfix":             float(a["log_nfix"]["mean"]),
            "auc_ttff":             float(a["log_ttff"]["mean"]),
            "perm_p":               float(res["A_permutation_delta"]["p"]),
            "lsc_null_mean":        float(res["A_label_shuffle_delta"]["null_mean"]),
            "oof_delta":            float(b.get("delta",            {}).get("auc_oof", np.nan)),
            "oof_dwell":            float(b.get("dwell",            {}).get("auc_oof", np.nan)),
            "oof_ttff":             float(b.get("ttff",             {}).get("auc_oof", np.nan)),
            "oof_delta_dwell_ttff": float(b.get("delta+dwell+ttff", {}).get("auc_oof", np.nan)),
            "incr_delta_auc":       float(c["delta_auc"]["mean"]),
            "incr_ci_low":          float(c["delta_auc"]["ci_low"]),
            "incr_ci_high":         float(c["delta_auc"]["ci_high"]),
            "incr_wilcoxon_p":      float(c["wilcoxon_p"]),
        })

    sens = pd.DataFrame(rows).sort_values("delta_min_fix").reset_index(drop=True)
    path = os.path.join(OUT_DIR, "sensitivity_summary.csv")
    sens.to_csv(path, index=False)
    print(f"\nSensitivity summary -> {path}")
    print(sens.to_string(index=False))


if __name__ == "__main__":
    run_all()