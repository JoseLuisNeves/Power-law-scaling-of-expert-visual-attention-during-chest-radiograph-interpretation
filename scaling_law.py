import os
import json
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
@dataclass
class ScalingLawParams:
    slope: float
    intercept: float
    r2: float
    gamma_ci: Tuple[float, float]
    def to_dict(self) -> Dict[str, float]:
        return {"slope": self.slope, "intercept": self.intercept, "r2": self.r2, "gamma_ci": self.gamma_ci}

def load_scaling_params(period: str, min_fix: int = 3, relate_to_mention: Optional[str] = None) -> Tuple[ScalingLawParams, Dict[str, ScalingLawParams]]:
    base_path = os.path.dirname(os.path.abspath(__file__))
    sub_folder = relate_to_mention if relate_to_mention else "all"
    file_suffix = f"_{relate_to_mention}" if relate_to_mention else ""
    file_name = f"scaling_law_params_{period}{file_suffix}.json"
    full_path = os.path.join(base_path, "results", "exp1a", period, str(min_fix), sub_folder, file_name)
    full_path = os.path.join(base_path, "results", "exp1a", "pre-reporting", str(min_fix), "all", f"scaling_law_params_pre-reporting.json")
    with open(full_path, "r") as f:
        data = json.load(f)
    overall_params = ScalingLawParams(**data["overall"]) 
    per_region_params = {region: ScalingLawParams(**params) for region, params in data["per_region"].items()}
    return overall_params, per_region_params
