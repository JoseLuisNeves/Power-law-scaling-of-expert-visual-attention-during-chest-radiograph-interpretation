import os 
import json 
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
from scaling_law import ScalingLawParams
import statsmodels.api as sm
from stats_utils import freedman_diaconis_bins
from statsmodels.discrete.discrete_model import NegativeBinomial
from local_annotations import EllipseAttention
from reflacxloader import ReflacxLoader

plt.rcParams.update({"font.family": "sans-serif", "font.size": 13, "axes.labelsize": 14, 
                     "axes.titlesize": 14, "axes.linewidth": 1.2, "xtick.direction": "in", 
                     "ytick.direction": "in", "axes.spines.top": False, "axes.spines.right": False})

# Target abnormality types for paper table
TARGET_ABNORMALITIES = ['consolidation', 'atelectasis', 'pleural abnormality', 
                        'pulmonary edema', 'groundglass opacity']

def load_data(min_fix: int, period: str, relate_to_mention: Optional[str] = None) -> List[EllipseAttention]:
    loader = ReflacxLoader()
    loader.load_jsons()
    records = []
    for patient_id in loader.transcripts_dict.keys():
        for study_id in loader.transcripts_dict[patient_id].keys():
            try:
                ellipses_attention = loader.get_ellipses_attention(patient_id, study_id, period, relate_to_mention)
                records.extend(ellipses_attention)
            except Exception as e:
                continue
    return [e for e in records if len(e.fixations) >= min_fix]

def fit_negative_binomial_to_density(ellipses_attention: List[EllipseAttention]) -> ScalingLawParams:
    areas = np.array([e.ellipse.area for e in ellipses_attention])
    counts = np.array([len(e.fixations) for e in ellipses_attention])
    densities = np.array([e.density for e in ellipses_attention])
    
    logA = np.log(areas)
    logD = np.log(densities)
    
    X = sm.add_constant(logA)
    model = NegativeBinomial(counts, X).fit(disp=False)
    
    intercept = float(model.params[0])
    gamma = float(model.params[1]) - 1.0
    conf = model.conf_int()
    gamma_ci = (float(conf[1][0]) - 1.0, float(conf[1][1]) - 1.0)
    
    # Freedman-Diaconis rule for bin count
    n_bins = freedman_diaconis_bins(logA)
    
    # Create bins for structural R² calculation
    dt_temp = pd.DataFrame({"logA": logA, "logD": logD})
    dt_temp['area_bin'] = pd.qcut(dt_temp['logA'], q=n_bins, duplicates='drop')
    binned = dt_temp.groupby('area_bin', observed=True).agg({'logA': 'mean', 'logD': 'mean'}).reset_index(drop=True)
    
    # Structural R² calculation
    binned_pred = intercept + gamma * binned['logA']
    ss_res = np.sum((binned['logD'] - binned_pred)**2)
    ss_tot = np.sum((binned['logD'] - binned['logD'].mean())**2)
    binned_r2 = 1 - (ss_res / ss_tot)
    
    return ScalingLawParams(slope=gamma, intercept=intercept, r2=binned_r2, gamma_ci=gamma_ci)

def plot_binned_density(ellipses_attention: List[EllipseAttention], scaling_params: ScalingLawParams, 
                        label: Optional[str] = None, out_path=None):
    areas = np.array([e.ellipse.area for e in ellipses_attention])
    densities = np.array([e.density for e in ellipses_attention])
    logA = np.log(areas)
    logD = np.log(densities)
    
    n_bins = freedman_diaconis_bins(logA)
    df_temp = pd.DataFrame({"logA": logA, "logD": logD})
    df_temp['bin'] = pd.qcut(df_temp['logA'], q=n_bins, duplicates='drop')
    binned = df_temp.groupby('bin', observed=True).agg({'logA': 'mean', 'logD': 'mean'})
    
    gamma = scaling_params.slope
    intercept = scaling_params.intercept
    
    x_line = np.linspace(logA.min(), logA.max(), 100)
    y_line = intercept + gamma * x_line
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(logA, logD, s=15, alpha=0.1, color="gray", label="Observations")
    ax.scatter(binned['logA'], binned['logD'], s=70, color="crimson", 
              edgecolors="black", zorder=3, label="Binned Means")
    ax.plot(x_line, y_line, color="black", linewidth=2, linestyle="--", label=f"γ={gamma:.2f}")
    ax.set_xlabel("log(Area)")
    ax.set_ylabel("log(Density)")
    ax.legend(frameon=False, loc='lower left')
    plt.tight_layout()
    
    os.makedirs(out_path, exist_ok=True)
    suffix = f"_{label.replace(' ', '_').replace('/', '_')}" if label else ""
    plt.savefig(os.path.join(out_path, f"binned_density{suffix}.png"), dpi=200)
    plt.close(fig)

def plot_count_histogram(ellipses_attention: List[EllipseAttention], label: Optional[str] = None, out_path=None):
    counts = np.array([len(e.fixations) for e in ellipses_attention])
    mean, median, std = np.mean(counts), np.median(counts), np.std(counts)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(counts, bins='auto', color="skyblue", edgecolor="black", alpha=0.7)
    stats_text = f"Mean: {mean:.2f}\nMedian: {median:.1f}\nStd: {std:.2f}\nN: {len(counts)}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', 
           horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.set_xlabel("Fixation Count")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Fixation Counts: {label if label else 'Overall'}")
    plt.tight_layout()
    
    os.makedirs(out_path, exist_ok=True)
    suffix = f"_{label.replace(' ', '_').replace('/', '_')}" if label else ""
    plt.savefig(os.path.join(out_path, f"counts_histogram{suffix}.png"), dpi=200)
    plt.close(fig)

def run_experiment(out_dir: str, min_fix: int, label_min_n: int, period: str, 
                   relate_to_mention: Optional[str] = None) -> Dict:
    """Run experiment and return results dict for sensitivity analysis."""
    os.makedirs(out_dir, exist_ok=True)
    
    filtered = load_data(min_fix, period, relate_to_mention=relate_to_mention)
    overall_params = fit_negative_binomial_to_density(filtered)
    
    plot_binned_density(filtered, overall_params, label="Overall", out_path=out_dir)
    plot_count_histogram(filtered, label="Overall", out_path=out_dir)
    
    # Build summary with CI
    summary_rows = [{
        "Period": period,
        "Relate_to_mention": relate_to_mention if relate_to_mention else "None",
        "Min_fixations": min_fix,
        "N": len(filtered),
        "Gamma": overall_params.slope,
        "Gamma_CI_lower": overall_params.gamma_ci[0],
        "Gamma_CI_upper": overall_params.gamma_ci[1],
        "Intercept": overall_params.intercept,
        "R2": overall_params.r2
    }]
    
    # Group by label
    label_map = {}
    for ea in filtered:
        for lbl in ea.ellipse.labels:
            lbl_lower = lbl.lower()
            if lbl_lower not in label_map:
                label_map[lbl_lower] = []
            label_map[lbl_lower].append(ea)
    
    per_label_params = {}
    for label, ellipses_attention in label_map.items():
        if len(ellipses_attention) >= label_min_n:
            params = fit_negative_binomial_to_density(ellipses_attention)
            per_label_params[label] = params
            
            plot_binned_density(ellipses_attention, params, label=label, out_path=out_dir)
            plot_count_histogram(ellipses_attention, label=label, out_path=out_dir)
            
            summary_rows.append({
                "Period": period,
                "Relate_to_mention": relate_to_mention if relate_to_mention else "None",
                "Min_fixations": min_fix,
                "Label": label,
                "N": len(ellipses_attention),
                "Gamma": params.slope,
                "Gamma_CI_lower": params.gamma_ci[0],
                "Gamma_CI_upper": params.gamma_ci[1],
                "Intercept": params.intercept,
                "R2": params.r2
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, f"scaling_law_summary_{period}{'_' + relate_to_mention if relate_to_mention else ''}.csv"), index=False)
    
    params_dict = {
        "overall": overall_params.to_dict(), 
        "per_region": {label: params.to_dict() for label, params in per_label_params.items()}
    }
    with open(os.path.join(out_dir, f"scaling_law_params_{period}{'_' + relate_to_mention if relate_to_mention else ''}.json"), "w") as f:
        json.dump(params_dict, f, indent=4)
    
    return {
        "overall": overall_params,
        "per_label": per_label_params,
        "n_total": len(filtered),
        "summary_df": summary_df
    }

def print_paper_table(results_dict: Dict[str, Dict], min_fix: int):
    """Print formatted table for paper with target abnormalities."""
    print("\n" + "="*100)
    print(f"PAPER TABLE: Scaling Laws Across Abnormality Types (min_fix={min_fix})")
    print("="*100)
    
    phases = [
        ("pre-reporting", None, "Silent"),
        ("all_time", "pre-mention", "Pre-mention"),
        ("post-reporting", "post-mention", "Post-mention")
    ]
    
    print(f"\n{'Abnormality Type':<25} {'N (S/Pre/Post)':<20} {'Silent':<25} {'Pre-mention':<25} {'Post-mention':<25}")
    print(f"{'':25} {'':20} {'γ (95% CI)':<12} {'R²':<12} {'γ (95% CI)':<12} {'R²':<12} {'γ (95% CI)':<12} {'R²':<12}")
    print("-"*100)
    
    # Overall row
    overall_row = ["Overall", ""]
    for period, relate, _ in phases:
        key = f"{period}_{relate if relate else 'None'}"
        if key in results_dict:
            params = results_dict[key]['overall']
            gamma_str = f"{params.slope:.2f} [{params.gamma_ci[0]:.2f}, {params.gamma_ci[1]:.2f}]"
            r2_str = f"{params.r2:.2f}"
            overall_row.extend([gamma_str, r2_str])
        else:
            overall_row.extend(["--", "--"])
    
    print(f"{'Overall':<25} {'--':<20} {overall_row[2]:<12} {overall_row[3]:<12} {overall_row[4]:<12} {overall_row[5]:<12} {overall_row[6]:<12} {overall_row[7]:<12}")
    print("-"*100)
    
    # Target abnormalities
    for abn in TARGET_ABNORMALITIES:
        # Get sample sizes across phases
        n_strs = []
        for period, relate, _ in phases:
            key = f"{period}_{relate if relate else 'None'}"
            if key in results_dict and abn in results_dict[key]['per_label']:
                df = results_dict[key]['summary_df']
                n_row = df[df['Label'] == abn]
                if not n_row.empty:
                    n_strs.append(str(int(n_row.iloc[0]['N'])))
                else:
                    n_strs.append("--")
            else:
                n_strs.append("--")
        n_str = "/".join(n_strs)
        
        row_data = [abn.title(), n_str]
        for period, relate, _ in phases:
            key = f"{period}_{relate if relate else 'None'}"
            if key in results_dict and abn in results_dict[key]['per_label']:
                params = results_dict[key]['per_label'][abn]
                # UPDATED: Include confidence intervals for abnormalities
                gamma_str = f"{params.slope:.2f} [{params.gamma_ci[0]:.2f}, {params.gamma_ci[1]:.2f}]"
                r2_str = f"{params.r2:.2f}"
                row_data.extend([gamma_str, r2_str])
            else:
                row_data.extend(["--", "--"])
        
        print(f"{row_data[0]:<25} {row_data[1]:<20} {row_data[2]:<12} {row_data[3]:<12} {row_data[4]:<12} {row_data[5]:<12} {row_data[6]:<12} {row_data[7]:<12}")
    
    print("="*100 + "\n")

def print_sensitivity_analysis(sensitivity_results: Dict[int, Dict]):
    """Print sensitivity analysis for min_fix thresholds."""
    print("\n" + "="*100)
    print("SENSITIVITY ANALYSIS: Effect of Minimum Fixation Threshold")
    print("="*100)
    
    phases = [
        ("pre-reporting", None, "Silent"),
        ("all_time", "pre-mention", "Pre-mention"),
        ("post-reporting", "post-mention", "Post-mention")
    ]
    
    for period, relate, phase_name in phases:
        print(f"\n{phase_name} Phase:")
        print(f"{'Min Fix':<10} {'N':<10} {'γ':<15} {'95% CI':<25} {'R²':<10}")
        print("-"*70)
        
        for min_fix in sorted(sensitivity_results.keys()):
            key = f"{period}_{relate if relate else 'None'}"
            if key in sensitivity_results[min_fix]:
                params = sensitivity_results[min_fix][key]['overall']
                n = sensitivity_results[min_fix][key]['n_total']
                ci_str = f"[{params.gamma_ci[0]:.3f}, {params.gamma_ci[1]:.3f}]"
                print(f"{min_fix:<10} {n:<10} {params.slope:<15.3f} {ci_str:<25} {params.r2:<10.3f}")
        print()
    
    print("="*100 + "\n")

def print_exploratory_abnormality_analysis(results_dict: Dict[str, Dict], min_fix: int, label_min_n: int):
    """Print exploratory analysis of all abnormality types to inform table selection."""
    print("\n" + "="*100)
    print(f"EXPLORATORY ANALYSIS: All Abnormality Types (min_fix={min_fix}, min_N={label_min_n})")
    print("="*100)
    
    phases = [
        ("pre-reporting", None, "Silent"),
        ("all_time", "pre-mention", "Pre-mention"),
        ("post-reporting", "post-mention", "Post-mention")
    ]
    
    # Collect all abnormality types and their sample sizes
    all_abnormalities = set()
    for period, relate, _ in phases:
        key = f"{period}_{relate if relate else 'None'}"
        if key in results_dict and 'per_label' in results_dict[key]:
            all_abnormalities.update(results_dict[key]['per_label'].keys())
    
    # Build data for each abnormality
    abnormality_data = []
    for abn in sorted(all_abnormalities):
        n_counts = []
        r2_values = []
        gamma_values = []
        ci_widths = []
        
        for period, relate, _ in phases:
            key = f"{period}_{relate if relate else 'None'}"
            if key in results_dict and abn in results_dict[key]['per_label']:
                params = results_dict[key]['per_label'][abn]
                df = results_dict[key]['summary_df']
                n_row = df[df['Label'] == abn]
                if not n_row.empty:
                    n = int(n_row.iloc[0]['N'])
                    n_counts.append(n)
                    r2_values.append(params.r2)
                    gamma_values.append(params.slope)
                    ci_width = params.gamma_ci[1] - params.gamma_ci[0]
                    ci_widths.append(ci_width)
        
        if n_counts:  # Only include if present in at least one phase
            abnormality_data.append({
                'abnormality': abn,
                'total_n': sum(n_counts),
                'min_n': min(n_counts),
                'max_n': max(n_counts),
                'mean_r2': np.mean(r2_values),
                'min_r2': min(r2_values),
                'mean_gamma': np.mean(gamma_values),
                'mean_ci_width': np.mean(ci_widths),
                'n_phases': len(n_counts)
            })
    
    # Sort by total sample size
    abnormality_data.sort(key=lambda x: x['total_n'], reverse=True)
    
    print("\nAbnormality types ranked by total sample size across phases:")
    print(f"\n{'Abnormality':<30} {'Total N':<10} {'Min N':<10} {'Max N':<10} {'Phases':<10} {'Mean R²':<10} {'Min R²':<10} {'Mean γ':<10} {'Mean CI Width':<15}")
    print("-"*130)
    
    for data in abnormality_data:
        print(f"{data['abnormality']:<30} {data['total_n']:<10} {data['min_n']:<10} {data['max_n']:<10} "
              f"{data['n_phases']:<10} {data['mean_r2']:<10.2f} {data['min_r2']:<10.2f} "
              f"{data['mean_gamma']:<10.2f} {data['mean_ci_width']:<15.3f}")
    
    print("\n" + "="*100)
    print("SELECTION CRITERIA:")
    print("- High total N (>200 preferred for statistical power)")
    print("- Present in all 3 phases (n_phases = 3)")
    print("- Good fit quality (Mean R² > 0.75)")
    print("- Narrow confidence intervals (Mean CI Width < 0.3)")
    print("- Clinical relevance and diversity from current selection")
    print("="*100)
    
    # Highlight candidates not in TARGET_ABNORMALITIES
    candidates = [d for d in abnormality_data 
                  if d['abnormality'] not in TARGET_ABNORMALITIES 
                  and d['total_n'] > 200 
                  and d['n_phases'] == 3 
                  and d['mean_r2'] > 0.75]
    
    if candidates:
        print(f"\nTOP CANDIDATES (not currently in table, N>200, all phases, R²>0.75):")
        print(f"{'Abnormality':<30} {'Total N':<10} {'Mean R²':<10} {'Mean γ':<10} {'Mean CI Width':<15}")
        print("-"*75)
        for data in candidates[:5]:  # Top 5 candidates
            print(f"{data['abnormality']:<30} {data['total_n']:<10} {data['mean_r2']:<10.2f} "
                  f"{data['mean_gamma']:<10.2f} {data['mean_ci_width']:<15.3f}")
    
    print("\n")

if __name__ == "__main__":
    label_min_ellipses = 30
    min_fixations = [1, 3, 5]  # Sensitivity analysis thresholds
    experiments = [
        ("pre-reporting", [None]), 
        ("all_time", ["pre-mention"]), 
        ("post-reporting", ["post-mention"])
    ]
    
    # Store all results for sensitivity analysis
    sensitivity_results = {mf: {} for mf in min_fixations}
    
    # Run all experiments
    for min_fix in min_fixations:
        print(f"\n{'='*100}")
        print(f"Running experiments with min_fixations = {min_fix}")
        print(f"{'='*100}\n")
        
        for period, relates in experiments:
            for relate_to_mention in relates:
                output_directory = f"./results/exp1a/{period}/{min_fix}/{'all' if not relate_to_mention else relate_to_mention}/"
                
                results = run_experiment(
                    output_directory, 
                    min_fix, 
                    label_min_ellipses, 
                    period, 
                    relate_to_mention
                )
                
                # Store for sensitivity analysis
                key = f"{period}_{relate_to_mention if relate_to_mention else 'None'}"
                sensitivity_results[min_fix][key] = results
    
    # Print comprehensive results
    print("\n" + "="*100)
    print("EXPERIMENT 1: SCALING LAW ANALYSIS - COMPREHENSIVE RESULTS")
    print("="*100)
    
    # Print table for min_fix=5 (main results)
    print_paper_table(sensitivity_results[5], min_fix=5)
    
    # Print sensitivity analysis
    print_sensitivity_analysis(sensitivity_results)
    
    # Print exploratory analysis of all abnormality types
    print_exploratory_abnormality_analysis(sensitivity_results[5], min_fix=5, label_min_n=label_min_ellipses)
    
    print("\n" + "="*100)
    print("All results saved to ./results/exp1a/")
    print("="*100 + "\n")
