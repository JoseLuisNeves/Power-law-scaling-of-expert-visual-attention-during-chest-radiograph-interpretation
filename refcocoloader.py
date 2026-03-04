import json
import numpy as np
from pathlib import Path
from typing import List
from fixationbuilder import Fixation
from local_annotations import EllipseAnnotation, EllipseAttention
class RefCocoLoader:
    def __init__(self):
        with open("refcoco/refcocogaze_train_correct.json") as f: 
            self.data = json.load(f)
        with open("refcoco/refcocogaze_val_correct.json") as f: 
            val_data = json.load(f)
        self.data.extend(val_data)
        with open("refcoco/word-timing.json") as f:
            timings = json.load(f)
        self.timing_map = {item['ref_id']: item['target_period'] for item in timings} # ref_id -> target_period
    def _find_target_idx(self, ref_words, target):
        tgt = target.lower().strip()
        for i, w in enumerate(ref_words):
            if w.lower() == tgt: return i
        toks = tgt.split()
        for i in range(len(ref_words) - len(toks) + 1):
            if [w.lower() for w in ref_words[i:i+len(toks)]] == toks: return i
        return -1
    def get_ellipses_attention(self, period: str, min_fix: int = 1, first_word_only: bool = True) -> List[EllipseAttention]:
        records = []
        for d in self.data:
            target_idx = self._find_target_idx(d['REF_WORDS'], d['TARGET'])
            if target_idx is None or target_idx == -1: continue
            if first_word_only and target_idx > 0: continue
            raw_period = self.timing_map.get(d['REF_ID'], d.get('TARGET_SPOKEN_PERIOD'))
            if not raw_period: continue
            word_begin = raw_period[0] + d['SOUND_ON']       
            relevant_fixs = [
                Fixation(x=x, y=y, start_time=start, end_time=start + dur)
                for x, y, start, dur, in_bbox in zip(d['FIX_X'], d['FIX_Y'], d['FIX_START'], d['FIX_DURATION'], d['FIX_IN_BBOX'])
                if in_bbox and ((period == "pretarget" and 0 <= start < word_begin) or (period == "posttarget" and start >= word_begin))]
            if len(relevant_fixs) >= min_fix:
                x, y, w, h = d['BBOX']
                records.append(EllipseAttention(patient_id=str(d['SUBJECT_ID']), study_id=str(d['REF_GAZE_ID']), ellipse=EllipseAnnotation(coords=(x, y, x+w, y+h), labels=[d['TARGET']]), fixations=relevant_fixs, mention_time=word_begin,start_reporting_time=d['SOUND_ON']))
        return records