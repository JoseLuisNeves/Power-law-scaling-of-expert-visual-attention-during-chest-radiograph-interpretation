import numpy as np
from dataclasses import dataclass, field
from fixationbuilder import Fixation
from typing import Optional, Tuple, List

@dataclass
class EllipseAnnotation:
    coords: Tuple[float, float, float, float] #(xmin, ymin, xmax, ymax)
    labels: List[str]
    @property
    def center(self) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = self.coords
        return (xmin + xmax) / 2, (ymin + ymax) / 2
    @property
    def radial_coords(self) -> Tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = self.coords
        cx, cy = self.center
        rx = (xmax - xmin) / 2
        ry = (ymax - ymin) / 2
        return cx, cy, rx, ry
    @property
    def area(self) -> float:
        _, _, rx, ry = self.radial_coords
        return np.pi * rx * ry
    def contains_point(self, x: float, y: float) -> bool:
        cx, cy, rx, ry = self.radial_coords
        norm_x = (x - cx) / rx
        norm_y = (y - cy) / ry
        return norm_x**2 + norm_y**2 <= 1

@dataclass
class AnatomicalRegion:
    coords: Tuple[float, float, float, float]  #(xmin, ymin, xmax, ymax)
    label: str
    def contains_point(self, x: float, y: float) -> bool:
        xmin, ymin, xmax, ymax = self.coords
        return xmin <= x <= xmax and ymin <= y <= ymax
    @property
    def area(self) -> float:
        xmin, ymin, xmax, ymax = self.coords
        return (xmax - xmin) * (ymax - ymin)

@dataclass
class EllipseAttention:
    patient_id: str
    study_id: str
    ellipse: EllipseAnnotation
    fixations: List[Fixation]
    mention_time: Optional[float]
    start_reporting_time: Optional[float]
    fixation_times_relative_to_reporting: List[float] = field(init=False)
    def __post_init__(self):
        reference_time = self.mention_time if self.mention_time is not None else self.start_reporting_time
        self.fixation_times_relative_to_reporting = [fix.start_time - reference_time for fix in self.fixations]
    @property
    def density(self) -> float:
        return len(self.fixations) / self.ellipse.area