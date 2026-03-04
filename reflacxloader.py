import os
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from fixationbuilder import Fixation, build_fixations
from local_annotations import EllipseAnnotation, AnatomicalRegion, EllipseAttention
from entity_extractor import EntityExtractor, align_gaze_with_mention

class ReflacxLoader:
    entity_extractor: EntityExtractor = EntityExtractor()
    def load_jsons(self):
        with open("./dataset/jsons/transcripts.json") as f:
            self.transcripts_dict = json.load(f)
        with open("./dataset/jsons/timestamps.json") as f:
            self.word_timestamps_dict = json.load(f)
        with open("./dataset/jsons/abnormality_ellipses.json") as f:
            self.ellipses_dict = json.load(f)
        with open("./dataset/jsons/fixations.json") as f:
            self.fixations_dict = json.load(f)  
        with open("./dataset/jsons/chest_bbs.json") as f:
            self.chest_dict = json.load(f)
    def get_chest(self, patient_id: str, study_id: str) -> AnatomicalRegion:
        study_chest_raw = self.chest_dict[patient_id][study_id][0]
        chest_coords = (int(study_chest_raw["xmin"]),int(study_chest_raw["ymin"]),int(study_chest_raw["xmax"]),int(study_chest_raw["ymax"]))
        return AnatomicalRegion(coords=chest_coords, label="chest")
    def get_study_ellipses(self, patient_id: str, study_id: str) -> List[EllipseAnnotation]:
        ellipses: List[EllipseAnnotation] = []
        study_ellipses = self.ellipses_dict[patient_id][study_id]
        for ellipse in study_ellipses:
            ellipse_coords = (int(ellipse["xmin"]),int(ellipse["ymin"]),int(ellipse["xmax"]),int(ellipse["ymax"]))
            conditions = {k: v.lower() == "true" for k, v in ellipse.items() if k not in ["xmin", "ymin", "xmax", "ymax"]}
            labels = [cond.lower() for cond, present in conditions.items() if present]
            ellipses.append(EllipseAnnotation(coords=ellipse_coords,labels=labels))
        return ellipses    
    def get_study_fixations(self,patient_id: str,study_id: str,period: str = "all"):
        raw_fixations = self.fixations_dict[patient_id][study_id]
        word_timestamps = self.word_timestamps_dict[patient_id][study_id]
        start_reporting_timestamp = float(word_timestamps[0]["timestamp_start_word"])
        return build_fixations(raw_fixations, period, start_reporting_timestamp) 
    def get_ellipses_attention(self, patient_id: str, study_id: str, period: str, relate_to_mention: Optional[str] = None) -> List[EllipseAttention]:
        ellipses = self.get_study_ellipses(patient_id, study_id)
        fixations = self.get_study_fixations(patient_id, study_id, period)
        ellipses_attention = []
        word_ts = self.word_timestamps_dict[patient_id][study_id]
        start_reporting_time = word_ts[0]["timestamp_start_word"]
        if not relate_to_mention:
            for ellipse in ellipses:
                relevant_fixations = []
                for fix in fixations:
                    if ellipse.contains_point(fix.x, fix.y):
                        relevant_fixations.append(fix)
                ellipses_attention.append(EllipseAttention(patient_id=patient_id, study_id=study_id, ellipse=ellipse, fixations=relevant_fixations, mention_time=None, start_reporting_time=float(start_reporting_time)))
        else:
            sentence_to_mentions = self.entity_extractor.extract(word_ts, ellipses)
            all_mentions = [m for ms in sentence_to_mentions.values() for m in ms]
            for e in ellipses:
                relevant_fixations: List[Fixation] = []
                for mention in all_mentions:
                    if e in mention.ellipses:
                        mention_fixations = align_gaze_with_mention(fixations, mention, relate_to_mention)
                        for fix in mention_fixations:
                            if e.contains_point(fix.x, fix.y):
                                relevant_fixations.append(fix)
                ellipses_attention.append(EllipseAttention(patient_id=patient_id, study_id=study_id, ellipse=e, fixations=relevant_fixations, mention_time=mention.end_time, start_reporting_time=float(start_reporting_time)))           
        return ellipses_attention      