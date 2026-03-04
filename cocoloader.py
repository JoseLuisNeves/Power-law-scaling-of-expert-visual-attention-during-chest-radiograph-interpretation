import json 
from pathlib import Path
from typing import List, Dict
from fixationbuilder import Fixation
from local_annotations import EllipseAnnotation, EllipseAttention
class CocoLoader:
    data_dir: Path = Path("COCOSearch18-fixations-TP")
    splits: List[str] = ["train", "validation"]
    def get_ellipses_attention(self, min_fix: int) -> List[EllipseAttention]:
        all_ellipse_attentions: List[EllipseAttention] = []
        correct_index_list: List[int] = []
        for split in self.splits:
            pattern = f"*_TP_{split}_*.json"
            json_files = self.data_dir.glob(pattern)
            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)
                for entry in data:
                    x, y, w, h = entry["bbox"]
                    ellipse = EllipseAnnotation(coords=(x, y, x + w, y + h), labels=entry["task"])
                    correct_index_list.append(entry["correct"]) # 1 if correct, 0 otherwise
                    fixations = [Fixation(x=entry["X"][i], y=entry["Y"][i], start_time=entry["T"][i], end_time=0) for i in range(len(entry["X"])) if ellipse.contains_point(entry["X"][i], entry["Y"][i])]
                    if len(fixations) < min_fix:
                        continue
                    all_ellipse_attentions.append(EllipseAttention(patient_id=str(entry["name"]),study_id=str(entry["subject"]), ellipse=ellipse, fixations=fixations, mention_time=None, start_reporting_time=0.0))
        return all_ellipse_attentions, correct_index_list
    