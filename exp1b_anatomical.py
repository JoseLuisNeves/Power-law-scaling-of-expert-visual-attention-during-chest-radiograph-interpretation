import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scaling_law import ScalingLawParams
import statsmodels.api as sm
from stats_utils import freedman_diaconis_bins
from statsmodels.discrete.discrete_model import NegativeBinomial
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict
from reflacxloader import ReflacxLoader
from local_annotations import AnatomicalRegion, EllipseAnnotation
from fixationbuilder import Fixation
plt.rcParams.update({"font.family": "sans-serif", "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14, "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in", "axes.spines.top": False, "axes.spines.right": False})
def load_yolo_anat_regions(patient_id: str, threshold: float = 0.9) -> Dict[str, AnatomicalRegion]:
    path = Path(f"./dataset/reflacx_anat_region_boxe/{patient_id}")
    if not path.exists(): return {}
    try:
        subfolder = next(path.iterdir())
        df = pd.read_csv(subfolder / "boxes.csv")
        regions = {}
        for _, row in df[df["confidence"] >= threshold].iterrows():
            label = str(row["label"]).lower().replace(" ", "_")
            if label in ["left_lung", "right_lung"]:
                coords = (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))
                regions[label] = AnatomicalRegion(coords=coords, label=label)
        return regions
    except Exception: return {}
def get_shuffled_fixations(fixs: List[Fixation], chest: AnatomicalRegion) -> List[Fixation]:
    xmin, ymin, xmax, ymax = chest.coords
    shuffled = []
    for f in fixs:
        new_x = np.random.uniform(xmin, xmax)
        new_y = np.random.uniform(ymin, ymax)
        shuffled.append(Fixation(x=new_x, y=new_y, start_time=f.start_time, end_time=f.end_time))
    return shuffled
def fit_negative_binomial_to_anat_baseline(areas: np.ndarray, counts: np.ndarray) -> ScalingLawParams:
    logA, logD = np.log(areas), np.log(counts / areas)
    X = sm.add_constant(logA)
    model = NegativeBinomial(counts, X).fit(disp=False)
    intercept, gamma, conf = float(model.params[0]), float(model.params[1])-1.0, model.conf_int()
    gamma_ci = (float(conf[1][0]) - 1.0, float(conf[1][1]) - 1.0)
    n_bins = freedman_diaconis_bins(logA)
    dt_temp = pd.DataFrame({"logA": logA, "logD": logD})
    dt_temp['area_bin'] = pd.qcut(dt_temp['logA'], q=n_bins, duplicates='drop')
    binned = dt_temp.groupby('area_bin', observed=True).agg({'logA': 'mean', 'logD': 'mean'}).reset_index(drop=True)
    binned_pred = intercept + gamma * binned['logA']
    ss_res = np.sum((binned['logD'] - binned_pred)**2)
    ss_tot = np.sum((binned['logD'] - binned['logD'].mean())**2)
    binned_r2 = 1 - (ss_res / ss_tot)
    return ScalingLawParams(slope=gamma, intercept=intercept, r2=binned_r2, gamma_ci=gamma_ci) 
def plot_binned_baseline(areas: np.ndarray, counts: np.ndarray, scaling_params: ScalingLawParams, label: Optional[str] = None, out_path=None):
    logA, logD = np.log(areas), np.log(counts / areas)
    n_bins = freedman_diaconis_bins(logA)
    df_temp = pd.DataFrame({"logA": logA, "logD": logD})
    df_temp['bin'] = pd.qcut(df_temp['logA'], q=n_bins, duplicates='drop')
    binned = df_temp.groupby('bin', observed=True).agg({'logA': 'mean', 'logD': 'mean'})
    gamma, intercept = scaling_params.slope, scaling_params.intercept
    x_line = np.linspace(logA.min(), logA.max(), 100)
    y_line = intercept + gamma * x_line
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(logA, logD, s=15, alpha=0.1, color="gray", label="Observations")
    ax.scatter(binned['logA'], binned['logD'], s=70, color="crimson", edgecolors="black", zorder=3, label="Binned Means")
    ax.plot(x_line, y_line, color="black", linewidth=2, linestyle="--", label=f"γ={gamma:.2f}")
    ax.set_xlabel("log(Area)"), ax.set_ylabel("log(Density)"), ax.legend(frameon=False, loc='lower left')
    plt.tight_layout()
    suffix = f"_{label}" if label else ""
    if out_path:
        os.makedirs(out_path, exist_ok=True)
        plt.savefig(os.path.join(out_path, f"binned_baseline{suffix}.png"), dpi=200)
    plt.close(fig)
def run_experiment(out_dir: str, min_fix: int, period: str):
    os.makedirs(out_dir, exist_ok=True)
    loader = ReflacxLoader()
    loader.load_jsons()
    plot_data = defaultdict(list)
    for pid, studies in loader.transcripts_dict.items():
        anat = load_yolo_anat_regions(pid)
        for sid in studies:
            try:
                fixs = loader.get_study_fixations(pid, sid, period=period)
                chest = loader.get_chest(pid, sid)
                shuffled_fixs = get_shuffled_fixations(fixs, chest)
                gt_ellipses = loader.get_study_ellipses(pid, sid)
                targets = {**anat, 'chest': chest}
                for key, region in targets.items():
                    n_real = sum(1 for f in fixs if region.contains_point(f.x, f.y))
                    n_shuff = sum(1 for f in shuffled_fixs if region.contains_point(f.x, f.y))
                    if n_real >= min_fix:
                        plot_data[f"{key}_real"].append([region.area, n_real])
                        plot_data[f"{key}_shuffled"].append([region.area, n_shuff])
                for ellipse in gt_ellipses:
                    n_real = sum(1 for f in fixs if ellipse.contains_point(f.x, f.y))
                    n_shuff = sum(1 for f in shuffled_fixs if ellipse.contains_point(f.x, f.y))
                    if n_real >= min_fix:
                        plot_data["abnormality_real"].append([ellipse.area, n_real])
                        plot_data["abnormality_shuffled"].append([ellipse.area, n_shuff])
            except Exception:continue       
    summary_rows = []
    for name, data_list in plot_data.items():
        if len(data_list) < 5: continue 
        data_arr = np.array(data_list)
        areas, counts = data_arr[:, 0], data_arr[:, 1]
        counts_for_fit = np.where(counts == 0, 0.5, counts) 
        fit_res = fit_negative_binomial_to_anat_baseline(areas, counts_for_fit)
        plot_binned_baseline(areas, counts_for_fit, fit_res, label=name, out_path=out_dir)
        summary_rows.append({"Period": period, "Min_fixations": min_fix, "Label": name, "Slope": fit_res.slope, "Intercept": fit_res.intercept,"R2": fit_res.r2})
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "permutation_baseline_table.csv"), index=False)
if __name__ == "__main__":
    min_fixations = [5]
    periods = ["pre-reporting", "reporting"]
    for period in periods:
        for min_fix in min_fixations:
            output_directory = f"./results/exp1b/{period}/{min_fix}"
            run_experiment(output_directory, min_fix, period)