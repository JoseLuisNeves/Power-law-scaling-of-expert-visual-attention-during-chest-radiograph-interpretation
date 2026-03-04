import os
import warnings
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr, ttest_ind
from reflacxloader import ReflacxLoader
from stats_utils import cohens_d
from local_annotations import EllipseAttention
from exp1a_fit_scaling_laws import fit_negative_binomial_to_density
warnings.filterwarnings('ignore')
rng = np.random.RandomState(42)
N_STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "axes.linewidth": 1.0, "grid.linewidth": 0.5, "lines.linewidth": 1.5, "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600, "figure.autolayout": True}
plt.rcParams.update(N_STYLE)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#F0E442", "#CC79A7", "#56B4E9"]
RUN_PARAMS = {"WINDOW_WIDTH": 3, "WINDOW_STEP": 0.1, "TIMES": [-20.0, 5.0], "THRESHOLD": 5, "SUBSET_SIZE": 100, "N_BOOTSTRAP": 100}
WINDOWS = [(round(x, 2), round(x + RUN_PARAMS["WINDOW_WIDTH"], 2)) for x in np.arange(RUN_PARAMS["TIMES"][0], RUN_PARAMS["TIMES"][1], RUN_PARAMS["WINDOW_STEP"])]
def filter_to_window(ellipses: List[EllipseAttention], w_start: float, w_end: float, threshold: int = 1):
    filtered = []
    for e in ellipses:
        window_fixations = [f for f, t in zip(e.fixations, e.fixation_times_relative_to_reporting) if w_start <= t < w_end]
        if len(window_fixations) >= threshold:
            filtered.append(EllipseAttention(e.patient_id, e.study_id, e.ellipse, window_fixations, e.mention_time, e.start_reporting_time))
    return filtered
def bootstrap_fit(ellipses: List[EllipseAttention], subset_size: int, n_bootstrap: int):
    if len(ellipses) < subset_size: return None
    gammas, r2s, areas = [], [], []
    for _ in range(n_bootstrap):
        try:
            sample = rng.choice(ellipses, size=subset_size, replace=True)
            params = fit_negative_binomial_to_density(sample)
            gammas.append(params.slope)
            r2s.append(params.r2)
            areas.append(np.median([e.ellipse.area for e in sample]))
        except Exception: continue
    return {"gamma_mean": np.mean(gammas), "gamma_std": np.std(gammas), "r2_mean": np.mean(r2s), "area_median": np.median(areas)}
def compute_statistics(df: pd.DataFrame, group_name: str) -> dict:
    time, gamma, area = df['window_mid'].values, df['gamma'].values, df['area_median'].values
    rho_spearman, p_spearman = spearmanr(time, gamma)
    X = sm.add_constant(np.column_stack([time, area])) # OLS: Gamma ~ Time + Area
    model = sm.OLS(gamma, X).fit()
    beta_time, p_time = model.params[1], model.pvalues[1]
    ci_time = model.conf_int()[1]
    gamma_residuals = sm.OLS(gamma, sm.add_constant(area)).fit().resid # Residuals: partial out area, regress residual vs time
    resid_model = sm.OLS(gamma_residuals, sm.add_constant(time)).fit()
    beta_resid, p_resid = resid_model.params[1], resid_model.pvalues[1]
    ci_resid = resid_model.conf_int()[1]
    q1_idx = df['window_mid'] <= df['window_mid'].quantile(0.25) # T-test: Q1 vs Q4
    q4_idx = df['window_mid'] >= df['window_mid'].quantile(0.75)
    gamma_early, gamma_late = df.loc[q1_idx, 'gamma'].values, df.loc[q4_idx, 'gamma'].values
    effect_size = cohens_d(gamma_late, gamma_early)
    _, p_ttest = ttest_ind(gamma_late, gamma_early)
    return {'group': group_name, 'rho_spearman': rho_spearman, 'p_spearman': p_spearman, 'beta_time': beta_time, 'beta_time_ci_lower': ci_time[0], 'beta_time_ci_upper': ci_time[1], 'p_time': p_time, 'beta_resid': beta_resid, 'beta_resid_ci_lower': ci_resid[0], 'beta_resid_ci_upper': ci_resid[1], 'p_resid': p_resid, 'cohens_d': effect_size, 'p_ttest_early_vs_late': p_ttest, 'n_windows': len(df)}
def plot_scaling_law_evolution(results_df: pd.DataFrame, out_dir: str):
    metrics = [('gamma', r'Scaling Exponent ($\gamma$)', 'gamma_std'), ('r2', r'Model Fit ($R^2$)', None), ('area_median', r'Median Area ($px^2$)', None)]
    for metric_key, ylabel, std_key in metrics:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        df = results_df.sort_values('window_mid')
        ax.plot(df['window_mid'], df[metric_key], color=COLORS[0], marker='o', markersize=4, markeredgewidth=0.5, markeredgecolor='white')
        if std_key and std_key in df.columns:
            ax.fill_between(df['window_mid'], df[metric_key] - df[std_key], df[metric_key] + df[std_key], color=COLORS[0], alpha=0.15, lw=0)
        ax.axvline(0, color='#333333', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel(ylabel, fontweight='medium')
        ax.set_xlabel('Time Relative to Mention (s)', fontweight='medium')
        ax.legend(frameon=False, loc='best')
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.2)
        plt.savefig(os.path.join(out_dir, f'{metric_key}_evolution.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f'{metric_key}_evolution.png'), bbox_inches='tight')
        plt.close(fig)
def run_experiment(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'gamma_evolution.csv')
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
    else:
        loader = ReflacxLoader()
        loader.load_jsons()
        all_ellipses = []
        for patient_id, studies in loader.transcripts_dict.items():
            for study_id in studies:
                for relate, period in [("pre-mention", "pre-reporting"), ("post-mention", "reporting")]:
                    try:
                        all_ellipses.extend(loader.get_ellipses_attention(patient_id, study_id, period=period, relate_to_mention=relate))
                    except Exception: continue
        significant_ellipses = [e for e in all_ellipses if len(e.fixations) >= RUN_PARAMS["THRESHOLD"]]
        results = []
        for w_start, w_end in WINDOWS:
            windowed = filter_to_window(significant_ellipses, w_start, w_end, threshold=1)
            fit = bootstrap_fit(windowed, RUN_PARAMS["SUBSET_SIZE"], RUN_PARAMS["N_BOOTSTRAP"])
            if fit:
                results.append({'threshold': RUN_PARAMS["THRESHOLD"], 'window_mid': (w_start + w_end) / 2, 'gamma': fit['gamma_mean'], 'gamma_std': fit['gamma_std'], 'r2': fit['r2_mean'], 'area_median': fit['area_median']})
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_path, index=False)
    results_df = results_df.sort_values('window_mid')
    all_stats = []
    pre_df = results_df[results_df['window_mid'] < 0]
    if not pre_df.empty:  all_stats.append(compute_statistics(pre_df, "Pre-Mention"))
    post_df = results_df[results_df['window_mid'] >= 0]
    if not post_df.empty:
        peak_time = post_df.loc[post_df['gamma'].idxmax(), 'window_mid']
        post_decay_df = post_df[post_df['window_mid'] > peak_time]
        if len(post_decay_df) >= 3:
            all_stats.append(compute_statistics(post_decay_df, "Post-Peak-Decay"))
    pd.DataFrame(all_stats).to_csv(os.path.join(out_dir, 'stats_gamma_evolution.csv'), index=False)
    plot_scaling_law_evolution(results_df, out_dir)

if __name__ == "__main__":
    run_experiment("./results/exp2/pre-mention/")