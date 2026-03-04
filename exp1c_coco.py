import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from cocoloader import CocoLoader
from exp1a_fit_scaling_laws import fit_negative_binomial_to_density, plot_binned_density
plt.rcParams.update({"font.family": "sans-serif", "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14, "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in", "axes.spines.top": False, "axes.spines.right": False})
NUM_BINS = 15
def run_experiment(out_dir: str, min_fix: int, label_min_n: int = 30):
    os.makedirs(out_dir, exist_ok=True)
    loader = CocoLoader()
    attentions, correctness = loader.get_ellipses_attention(min_fix)
    groups = {"Overall": attentions}
    groups["Correct"] = [a for a, c in zip(attentions, correctness) if c == 1]
    groups["Incorrect"] = [a for a, c in zip(attentions, correctness) if c == 0]
    for ea in attentions:
        task = ea.ellipse.labels[0] 
        task_label = f"task_{task}"
        if task_label not in groups:
            groups[task_label] = []
        groups[task_label].append(ea)     
    summary_rows = []
    for name, group_att in groups.items():
        if len(group_att) >= label_min_n:
            params = fit_negative_binomial_to_density(group_att)
            plot_binned_density(group_att, params, label=name, out_path=out_dir)
            summary_rows.append({"Min_fixations": min_fix, "Label": name, "Count": len(group_att), "Slope": params.slope, "Intercept": params.intercept, "R2": params.r2})
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "coco_scaling_summary.csv"), index=False)
if __name__ == "__main__":
    min_fixations = [1, 3]
    for m_fix in min_fixations:
        output_directory = f"./results/exp1c/min_fix_{m_fix}/"
        run_experiment(output_directory, m_fix)