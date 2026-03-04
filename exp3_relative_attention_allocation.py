import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, wilcoxon
from scipy.stats import false_discovery_control

from reflacxloader import ReflacxLoader
from local_annotations import EllipseAttention
from exp1a_fit_scaling_laws import fit_negative_binomial_to_density

# ─── Configuration ────────────────────────────────────────────────────────────

RUN_PARAMS = {
    "min_ellipses_per_study":   2,
    "k_folds":                  5,
    "random_seed":              42,
    "min_ellipses_for_study_r2": 3,
    "fdr_alpha":                0.05,
}

EXPERIMENTS = [
    ("pre-reporting", None),
    ("all_time",      "pre-mention"),
    ("post-reporting","post-mention"),
]

PHASE_LABELS = {
    "pre-reporting": "pre_reporting",
    "pre-mention":   "pre_mention",
    "post-mention":  "post_mention",
}

MODEL_PAIRS = [
    ("power", "uniform"),
    ("power", "area"),
]

METRICS = ("r2", "mae")


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_phase_studies(
    period: str,
    relate_to_mention: Optional[str],
    min_fix: int,
) -> Dict[str, List[EllipseAttention]]:
    loader = ReflacxLoader()
    loader.load_jsons()

    study_dict: Dict[str, List[EllipseAttention]] = {}

    for patient_id, studies in tqdm(
        loader.transcripts_dict.items(), desc=f"Loading {period}", leave=False
    ):
        for study_id in studies:
            try:
                ellipses = loader.get_ellipses_attention(
                    patient_id, study_id,
                    period=period,
                    relate_to_mention=relate_to_mention,
                )
            except Exception:
                continue

            qualifying = [e for e in ellipses if len(e.fixations) >= min_fix]

            if len(qualifying) < RUN_PARAMS["min_ellipses_per_study"]:
                continue

            key = f"{patient_id}|{study_id}"
            study_dict[key] = qualifying

    return study_dict


# ─── Alpha estimation ─────────────────────────────────────────────────────────

def fit_alpha_from_studies(
    study_dict: Dict[str, List[EllipseAttention]],
    train_ids: List[str],
) -> float:
    train_ellipses: List[EllipseAttention] = []
    for sid in train_ids:
        train_ellipses.extend(study_dict[sid])
    params = fit_negative_binomial_to_density(train_ellipses)
    return params.slope + 1.0


# ─── Prediction models ────────────────────────────────────────────────────────

def predict_power(ellipses: List[EllipseAttention], alpha: float) -> np.ndarray:
    weights = np.array([e.ellipse.area ** alpha for e in ellipses], dtype=float)
    return weights / weights.sum()


def predict_uniform(ellipses: List[EllipseAttention]) -> np.ndarray:
    return np.full(len(ellipses), 1.0 / len(ellipses))


def predict_area(ellipses: List[EllipseAttention]) -> np.ndarray:
    areas = np.array([max(e.ellipse.area, 1e-12) for e in ellipses], dtype=float)
    return areas / areas.sum()


def observed_fractions(ellipses: List[EllipseAttention]) -> np.ndarray:
    counts = np.array([len(e.fixations) for e in ellipses], dtype=float)
    total  = counts.sum()
    return counts / total if total > 0 else np.zeros(len(ellipses))


# ─── Pooled metrics ───────────────────────────────────────────────────────────

def compute_pooled_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {"pearson": float("nan"), "r2": float("nan"), "mae": float("nan")}
    return {
        "pearson": float(pearsonr(yt, yp)[0]),
        "r2":      float(r2_score(yt, yp)),
        "mae":     float(np.mean(np.abs(yt - yp))),
    }


def aggregate_pooled_metrics(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    y_true = np.array([r["f_obs"] for r in rows], dtype=float)
    return {
        model: compute_pooled_metrics(
            y_true,
            np.array([r[f"f_pred_{model}"] for r in rows], dtype=float),
        )
        for model in ("power", "uniform", "area")
    }


# ─── Per-study metrics ────────────────────────────────────────────────────────

def compute_per_study_metrics(
    rows: List[Dict[str, Any]],
    model: str,
    min_ellipses: int,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-study R² and MAE for a given model.
    Studies with fewer than min_ellipses are excluded from R² but not MAE.
    """
    df = pd.DataFrame(rows)
    study_metrics: Dict[str, Dict[str, float]] = {}

    for case_id, grp in df.groupby("case_id"):
        y_true = grp["f_obs"].values
        y_pred = grp[f"f_pred_{model}"].values
        n      = len(y_true)

        mae = float(np.mean(np.abs(y_true - y_pred)))

        if n >= min_ellipses:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        else:
            r2 = float("nan")

        study_metrics[case_id] = {"r2": r2, "mae": mae, "n_ellipses": n}

    return study_metrics


# ─── Wilcoxon significance testing ────────────────────────────────────────────

def run_wilcoxon_paired(
    study_metrics_a: Dict[str, Dict[str, float]],
    study_metrics_b: Dict[str, Dict[str, float]],
    metric: str,
) -> Dict[str, float]:
    """
    One-sided paired Wilcoxon signed-rank test.
    Tests whether model_a (power law) is significantly better than model_b.
    For R²: higher is better. For MAE: lower is better.
    Returns statistic, p_value, n_pairs, and median_difference (a minus b).
    """
    common_ids = set(study_metrics_a.keys()) & set(study_metrics_b.keys())
    diffs = np.array([
        study_metrics_a[sid][metric] - study_metrics_b[sid][metric]
        for sid in common_ids
        if np.isfinite(study_metrics_a[sid][metric])
        and np.isfinite(study_metrics_b[sid][metric])
    ])

    if len(diffs) < 10:
        return {
            "statistic":         float("nan"),
            "p_value":           float("nan"),
            "n_pairs":           int(len(diffs)),
            "median_difference": float("nan"),
        }

    # Flip MAE diffs so positive always means power law is better
    test_diffs = -diffs if metric == "mae" else diffs
    stat, pval = wilcoxon(test_diffs, alternative="greater")

    return {
        "statistic":         float(stat),
        "p_value":           float(pval),
        "n_pairs":           int(len(diffs)),
        "median_difference": float(np.median(diffs)),  # a minus b, raw
    }


def apply_bh_correction(
    raw_pvalues: Dict[str, float],
) -> Dict[str, float]:
    """
    Benjamini-Hochberg FDR correction applied globally across all comparisons.
    """
    labels = list(raw_pvalues.keys())
    pvals  = np.array([raw_pvalues[k] for k in labels])

    finite_mask = np.isfinite(pvals)
    corrected   = np.full(len(pvals), float("nan"))

    if finite_mask.sum() > 0:
        corrected[finite_mask] = false_discovery_control(
            pvals[finite_mask], method="bh"
        )

    return {labels[i]: float(corrected[i]) for i in range(len(labels))}


def run_all_significance_tests(
    rows: List[Dict[str, Any]],
    phase_label: str,
    min_ellipses: int,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Runs all paired Wilcoxon tests for one phase.
    Returns (structured results dict, raw p-values dict for global BH correction).
    """
    per_study = {
        model: compute_per_study_metrics(rows, model, min_ellipses)
        for model in ("power", "uniform", "area")
    }

    raw_pvalues:      Dict[str, float] = {}
    wilcoxon_results: Dict[str, Any]   = {}

    for model_a, model_b in MODEL_PAIRS:
        pair_key = f"{model_a}_vs_{model_b}"
        wilcoxon_results[pair_key] = {}

        for metric in METRICS:
            comp_key = f"{phase_label}__{pair_key}__{metric}"
            result   = run_wilcoxon_paired(
                per_study[model_a], per_study[model_b], metric
            )
            wilcoxon_results[pair_key][metric] = result
            raw_pvalues[comp_key] = result["p_value"]

    return wilcoxon_results, raw_pvalues


# ─── K-fold evaluation ────────────────────────────────────────────────────────

def run_kfold(
    study_dict: Dict[str, List[EllipseAttention]],
    k: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    study_ids = np.array(list(study_dict.keys()))
    kf        = KFold(n_splits=k, shuffle=True, random_state=seed)

    all_rows:   List[Dict[str, Any]] = []
    fold_stats: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(study_ids)):
        train_ids = study_ids[train_idx].tolist()
        test_ids  = study_ids[test_idx].tolist()

        alpha = fit_alpha_from_studies(study_dict, train_ids)

        fold_rows: List[Dict[str, Any]] = []

        for sid in test_ids:
            ellipses  = study_dict[sid]
            f_obs     = observed_fractions(ellipses)
            f_power   = predict_power(ellipses, alpha)
            f_uniform = predict_uniform(ellipses)
            f_area    = predict_area(ellipses)

            patient_id, study_id = sid.split("|", 1)

            for i, e in enumerate(ellipses):
                fold_rows.append({
                    "fold":           fold_idx,
                    "case_id":        sid,
                    "patient_id":     patient_id,
                    "study_id":       study_id,
                    "ellipse_idx":    i,
                    "k_ellipses":     len(ellipses),
                    "abnormality":    e.ellipse.labels[0] if e.ellipse.labels else "unknown",
                    "rho":            float(e.ellipse.area),
                    "n_fix":          len(e.fixations),
                    "alpha_used":     alpha,
                    "f_obs":          float(f_obs[i]),
                    "f_pred_power":   float(f_power[i]),
                    "f_pred_uniform": float(f_uniform[i]),
                    "f_pred_area":    float(f_area[i]),
                })

        fold_metrics = aggregate_pooled_metrics(fold_rows)
        fold_stats.append({
            "fold":            fold_idx,
            "alpha":           alpha,
            "n_test_studies":  len(test_ids),
            "n_test_ellipses": len(fold_rows),
            "metrics":         fold_metrics,
        })

        all_rows.extend(fold_rows)

    return all_rows, fold_stats


# ─── Summary construction ─────────────────────────────────────────────────────

def build_summary(
    pooled_metrics:   Dict[str, Dict[str, float]],
    fold_stats:       List[Dict[str, Any]],
    alpha_values:     List[float],
    wilcoxon_results: Dict[str, Any],
) -> Dict[str, Any]:
    models       = ("power", "uniform", "area")
    metric_names = ("pearson", "r2", "mae")

    fold_metrics_by_model: Dict[str, Dict[str, List[float]]] = {
        m: {mn: [] for mn in metric_names} for m in models
    }
    for fs in fold_stats:
        for m in models:
            for mn in metric_names:
                val = fs["metrics"][m][mn]
                if np.isfinite(val):
                    fold_metrics_by_model[m][mn].append(val)

    summary: Dict[str, Any] = {
        "alpha": {
            "mean":     float(np.mean(alpha_values)),
            "std":      float(np.std(alpha_values)),
            "per_fold": [float(a) for a in alpha_values],
        },
        "pooled_held_out": pooled_metrics,
        "fold_mean_std":   {},
        "wilcoxon":        wilcoxon_results,
    }

    for m in models:
        summary["fold_mean_std"][m] = {}
        for mn in metric_names:
            vals = fold_metrics_by_model[m][mn]
            summary["fold_mean_std"][m][mn] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std":  float(np.std(vals))  if vals else float("nan"),
            }

    return summary


# ─── Phase runner ─────────────────────────────────────────────────────────────

def run_phase(
    period:            str,
    relate_to_mention: Optional[str],
    min_fix:           int,
    out_dir:           str,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Full k-fold pipeline for one phase.
    Returns (summary dict, raw_pvalues dict for global BH correction).
    """
    os.makedirs(out_dir, exist_ok=True)

    phase_key   = relate_to_mention if relate_to_mention else period
    phase_label = PHASE_LABELS.get(phase_key, phase_key)

    print(f"\n{'='*60}")
    print(f"Phase: {phase_label}  "
          f"(period={period}, relate={relate_to_mention}, min_fix={min_fix})")
    print(f"{'='*60}")

    study_dict = load_phase_studies(period, relate_to_mention, min_fix)
    n_studies  = len(study_dict)
    print(f"Qualifying studies: {n_studies}")

    if n_studies < RUN_PARAMS["k_folds"]:
        raise ValueError(
            f"Not enough studies ({n_studies}) for "
            f"{RUN_PARAMS['k_folds']}-fold CV."
        )

    all_rows, fold_stats = run_kfold(
        study_dict,
        k=RUN_PARAMS["k_folds"],
        seed=RUN_PARAMS["random_seed"],
    )

    pooled_metrics = aggregate_pooled_metrics(all_rows)

    print(f"\nPooled held-out metrics:")
    print(f"{'Model':<16} {'Pearson':>8} {'R²':>8} {'MAE':>8}")
    for model in ("power", "uniform", "area"):
        m = pooled_metrics[model]
        print(f"{model:<16} {m['pearson']:>8.3f} {m['r2']:>8.3f} {m['mae']:>8.4f}")

    wilcoxon_results, raw_pvalues = run_all_significance_tests(
        all_rows,
        phase_label=phase_label,
        min_ellipses=RUN_PARAMS["min_ellipses_for_study_r2"],
    )

    print(f"\nRaw Wilcoxon p-values (one-sided, power law > baseline):")
    for pair_key, pair_results in wilcoxon_results.items():
        for metric, result in pair_results.items():
            print(
                f"  {pair_key} | {metric}: "
                f"p = {result['p_value']:.4f}  "
                f"(n_pairs = {result['n_pairs']}, "
                f"median_diff = {result['median_difference']:.4f})"
            )

    alpha_values = [fs["alpha"] for fs in fold_stats]
    print(f"\nMean alpha: {np.mean(alpha_values):.4f} ± {np.std(alpha_values):.4f}")

    summary = build_summary(
        pooled_metrics, fold_stats, alpha_values, wilcoxon_results
    )
    summary.update({
        "phase":             phase_label,
        "period":            period,
        "relate_to_mention": relate_to_mention,
        "min_fix":           min_fix,
        "n_studies":         n_studies,
        "n_folds":           RUN_PARAMS["k_folds"],
    })

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "fold_stats.json"), "w") as f:
        json.dump(fold_stats, f, indent=2)
    pd.DataFrame(all_rows).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False
    )

    return summary, raw_pvalues


# ─── Table builder ────────────────────────────────────────────────────────────

def build_paper_table(
    phase_summaries: Dict[str, Dict[str, Any]],
    all_raw_pvalues: Dict[str, float],
    out_path:        str,
) -> None:
    corrected_pvalues = apply_bh_correction(all_raw_pvalues)

    table: Dict[str, Any] = {
        "models":          {},
        "alpha_per_phase": {},
        "significance": {
            "raw_pvalues":       all_raw_pvalues,
            "bh_corrected":      corrected_pvalues,
            "correction_method": "Benjamini-Hochberg FDR",
            "fdr_alpha":         RUN_PARAMS["fdr_alpha"],
            "n_comparisons":     len(all_raw_pvalues),
            "test": (
                "Wilcoxon signed-rank, one-sided (power law > baseline), "
                "paired at study level"
            ),
        },
    }

    for phase_key, summary in phase_summaries.items():
        col = PHASE_LABELS.get(phase_key, phase_key)
        table["alpha_per_phase"][col] = summary["alpha"]
        for model in ("power", "uniform", "area"):
            if model not in table["models"]:
                table["models"][model] = {}
            table["models"][model][col] = {
                **summary["pooled_held_out"][model],
                "wilcoxon_vs_power": None,  # power law is the reference
            }

    # Attach BH-corrected p-values into each model/phase cell
    for comp_key, pval_corr in corrected_pvalues.items():
        parts = comp_key.split("__")     # phase__pair__metric
        phase_label, pair, metric = parts[0], parts[1], parts[2]
        _, model_b = pair.split("_vs_")  # "power_vs_uniform" -> baseline is uniform
        col = phase_label
        if model_b in table["models"] and col in table["models"][model_b]:
            table["models"][model_b][col][f"p_bh_{metric}"] = pval_corr
            table["models"][model_b][col][f"p_raw_{metric}"] = all_raw_pvalues[comp_key]

    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)

    print(f"\nPaper table saved to: {out_path}")
    print(f"\nBH-corrected p-values ({len(all_raw_pvalues)} comparisons total):")
    for comp_key, pval_corr in corrected_pvalues.items():
        pval_raw = all_raw_pvalues[comp_key]
        sig = (
            "***" if pval_corr < 0.001 else
            "**"  if pval_corr < 0.01  else
            "*"   if pval_corr < 0.05  else
            "ns"
        )
        print(
            f"  {comp_key}: "
            f"raw p = {pval_raw:.4f}, "
            f"BH p = {pval_corr:.4f}  {sig}"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MIN_FIX  = 5
    BASE_OUT = "./results/exp3_kfold"

    phase_summaries: Dict[str, Dict[str, Any]] = {}
    all_raw_pvalues: Dict[str, float]           = {}

    for period, relate_to_mention in EXPERIMENTS:
        phase_key  = relate_to_mention if relate_to_mention else period
        sub_folder = relate_to_mention if relate_to_mention else "all"
        out_dir    = os.path.join(BASE_OUT, period, str(MIN_FIX), sub_folder)

        summary, raw_pvalues = run_phase(
            period=period,
            relate_to_mention=relate_to_mention,
            min_fix=MIN_FIX,
            out_dir=out_dir,
        )

        phase_summaries[phase_key] = summary
        all_raw_pvalues.update(raw_pvalues)

    table_path = os.path.join(BASE_OUT, "paper_table.json")
    build_paper_table(phase_summaries, all_raw_pvalues, table_path)