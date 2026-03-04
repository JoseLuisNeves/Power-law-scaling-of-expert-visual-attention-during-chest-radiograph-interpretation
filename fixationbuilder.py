from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass
class Fixation:
    x: float
    y: float
    start_time: float
    end_time: float
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

def build_fixations(raw_fixations: List[Dict[str,str]], period, start_reporting_timestamp) -> Fixation:
    fixations: List[Fixation] = []
    for fix in raw_fixations:
        x, y, start_time, end_time = float(fix["x_position"]), float(fix["y_position"]), float(fix["timestamp_start_fixation"]), float(fix["timestamp_end_fixation"])
        if period == "reporting":
            if start_time < start_reporting_timestamp:
                continue
        elif period == "pre-reporting":
            if start_time >= start_reporting_timestamp:
                continue
        fixations.append(Fixation(x=x, y=y, start_time=start_time, end_time=end_time))
    return fixations