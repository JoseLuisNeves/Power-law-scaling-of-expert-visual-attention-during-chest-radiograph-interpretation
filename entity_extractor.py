import spacy
import numpy as np
import re
from fixationbuilder import Fixation
from dataclasses import dataclass
from typing import List, Dict, Tuple
from local_annotations import EllipseAnnotation
from span_abnormality_mapping import term_abnormality_mapping
nlp = spacy.load("en_core_web_sm")
LOOKBACK_PERIOD = 20
@dataclass
class AbnormalityMention:
    label: str
    start_time: float
    end_time: float
    ellipses: List[EllipseAnnotation]

class EntityExtractor:
    def __init__(self):
        self.term_abnormality_mapping = term_abnormality_mapping

    def link_entity(self, span: str) -> str:
        """Links a text span to a canonical abnormality label."""
        return self.term_abnormality_mapping.get(span.lower())

    def segment_timestamps_by_sentence(self, word_timestamps: List[Dict]) -> Tuple[List[List[Dict]], List]:
        full_text = " ".join([item['word'] for item in word_timestamps])
        doc = nlp(full_text)

        sentences_timestamps = []
        word_index = 0
        doc_sents = list(doc.sents)

        for sent in doc_sents:
            num_words_in_sentence = len(sent)
            sentence_slice = word_timestamps[word_index:word_index+num_words_in_sentence]
            sentences_timestamps.append(sentence_slice)
            word_index += num_words_in_sentence
        return sentences_timestamps, doc_sents

    def sliding_window(self, sentence: str, sentence_word_timestamps: List[Dict], all_ellipses: List[EllipseAnnotation]) -> List[AbnormalityMention]:
        tokens = [d["word"] for d in sentence_word_timestamps]
        mentions = []
        all_ellipse_labels = {label.lower() for ellipse in all_ellipses for label in ellipse.labels}
        #print(f"ellipse labels: {all_ellipse_labels}")
        max_ngram_len = min(len(tokens),3)
        for window_size in range(max_ngram_len, 0, -1):
            for i in range(len(tokens)-window_size+1):
                span_text = ' '.join(tokens[i:i+window_size]).lower()
                #print(f"text: {span_text}")
                record = self.link_entity(span_text)
                #print(f"record: {record}")
                if record and any(label in all_ellipse_labels for label in record):
                    start_time = float(sentence_word_timestamps[i]["timestamp_start_word"])
                    end_time = float(sentence_word_timestamps[i+window_size-1]["timestamp_end_word"])
                    actual_mention_label = [label for label in all_ellipse_labels if label in record]
                    #print(f"actual_mention_label: {actual_mention_label}")
                    if len(actual_mention_label) != 1:
                        print(f"More than one label associated with mention: record: {record}, all_ellipse_labels: {all_ellipse_labels}, actual_mention_label: {actual_mention_label}")
                    mention_ellipses = []
                    for ellipse in all_ellipses:
                        for label in ellipse.labels:
                            if label.lower() in actual_mention_label:
                                mention_ellipses.append(ellipse)
                    if mention_ellipses:
                        already_counted = False
                        for mention in mentions:
                            if end_time == mention.end_time:
                                already_counted = True
                        if not already_counted:
                            mentions.append(AbnormalityMention(actual_mention_label[0], start_time, end_time, mention_ellipses))
        return mentions

    def extract(self, word_timestamps: List[Dict], ellipses: List[EllipseAnnotation]) -> Dict[str, List[AbnormalityMention]]:        
        sentences_timestamps, sentences = self.segment_timestamps_by_sentence(word_timestamps)
        sentence_mention_mapping = {}
        for i, sent_text in enumerate(sentences):
            sentence = str(sent_text)
            timestamps = sentences_timestamps[i]
            mentions = self.sliding_window(sentence, timestamps, ellipses)
            if mentions:
                sentence_mention_mapping[sentence] = mentions
        return sentence_mention_mapping

def align_gaze_with_mention(gaze_sequence: List[Fixation], mention: AbnormalityMention, relate_to_mention: str)->List[Fixation]:
    aligned_events = []
    for event in gaze_sequence:
        if relate_to_mention == "pre-mention":
            if event.start_time >= mention.start_time - LOOKBACK_PERIOD and event.start_time <=  mention.start_time:
                aligned_events.append(event)
        elif relate_to_mention == "post-mention":
            if event.start_time >= mention.end_time:
                aligned_events.append(event)
    return aligned_events