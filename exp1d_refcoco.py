import os
import warnings
import numpy as np
import pandas as pd
from typing import List
from refcocoloader import RefCocoLoader
from exp1a_fit_scaling_laws import (fit_negative_binomial_to_density, plot_binned_density, plot_count_histogram)
warnings.filterwarnings("ignore")
def run_permutation_test(group_a, group_b, observed_delta, n_permutations=1000):
    combined = group_a + group_b
    n_a = len(group_a)
    null_deltas = []
    for i in range(n_permutations):
        np.random.shuffle(combined)
        fake_a = combined[:n_a]
        fake_b = combined[n_a:]
        try:
            params_a = fit_negative_binomial_to_density(fake_a)
            params_b = fit_negative_binomial_to_density(fake_b)
            null_deltas.append(params_b.slope - params_a.slope)
        except: continue
        if (i + 1) % 200 == 0:
            print(f"  Completed {i + 1}/{n_permutations}...")
    null_deltas = np.array(null_deltas)
    p_value = np.mean(np.abs(null_deltas) >= np.abs(observed_delta))
    return p_value, null_deltas
def run_experiment():
    loader = RefCocoLoader()
    out_dir = "results/exp1d"
    os.makedirs(out_dir, exist_ok=True)
    periods = ["pretarget", "posttarget"]
    datasets = {p: loader.get_ellipses_attention(p, min_fix=1) for p in periods}
    results = []
    for p in periods:
        data = datasets[p]
        print(f"\n📊 Analyzing {p.upper()} (N={len(data)} trials)")
        counts = np.array([len(e.fixations) for e in data])
        mean_val, med_val, std_val = np.mean(counts), np.median(counts), np.std(counts)
        print(f"  Fixation Stats: Mean={mean_val:.2f}, Median={med_val:.1f}, Std={std_val:.2f}")
        plot_count_histogram(data, label=p, out_path=out_dir)
        params = fit_negative_binomial_to_density(data)
        plot_binned_density(data, params, label=p, out_path=out_dir)
        results.append({"period": p,"gamma": params.slope, "gamma_low": params.gamma_ci[0],"gamma_high": params.gamma_ci[1], "r2": params.r2, "n_samples": len(data)})
    df_res = pd.DataFrame(results).set_index("period")
    obs_pre = df_res.loc["pretarget", "gamma"]
    obs_post = df_res.loc["posttarget", "gamma"]
    delta_gamma = obs_post - obs_pre
    print(f"\nRunning Permutation Test for Δγ...")
    p_val, null_dist = run_permutation_test(datasets["pretarget"], datasets["posttarget"], delta_gamma)
    print("\n" + "="*40)
    print(f"FINAL FINDINGS:")
    print(f"Pre-target γ:  {obs_pre:.3f}")
    print(f"Post-target γ: {obs_post:.3f}")
    print(f"Δγ (Shift):    {delta_gamma:+.3f}")
    print(f"P-Value:       {p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not Significant'})")
    print("="*40)
    df_res["p_value"] = p_val
    df_res.to_csv(os.path.join(out_dir, "scaling_comparison_with_stats.csv"))
    np.save(os.path.join(out_dir, "null_distribution.npy"), null_dist)

if __name__ == "__main__":
    run_experiment()