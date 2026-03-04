# Fixation Scaling Laws in Radiology
<img width="1165" height="656" alt="image" src="https://github.com/user-attachments/assets/fe9a32ea-1f42-4c0d-afeb-2b7982e693df" />


This repository contains the analysis code for the paper *"Power-law scaling of expert visual attention during chest radiograph interpretation"*.

The paper establishes three findings. First, radiologists' fixation counts on abnormalities follow a power law in abnormality area, and this relationship replicates in non-clinical referential attention datasets. Second, the exponent of this law compresses monotonically as radiologists approach diagnostic commitment. Third, geometry-corrected residuals from this law identify diagnostically difficult findings within individual cases beyond what fixation magnitude or latency metrics capture alone.

## Repository Structure

**Experiment scripts**

`exp1a_fit_scaling_laws.py` fits the negative binomial scaling law between abnormality area and fixation count across the REFLACX dataset, with sensitivity analysis across fixation count thresholds.

`exp1b_anatomical.py` fits the same model to anatomical regions and a shuffled fixation baseline.

`exp1c_coco.py` replicates the scaling law in COCO-Search18.

`exp1d_refcoco.py` replicates the scaling law in RefCOCO-Gaze, comparing pre-target and post-target periods.

`exp2_scaling_law_evolution.py` tracks the scaling exponent across sliding temporal windows relative to the moment of diagnostic mention.

`exp3_relative_attention_allocation.py` tests whether the power law outperforms uniform and area-proportional models in predicting within-case fixation allocation, using five-fold cross-validation.

`exp4_residuals_and_diagnostic_difficulty.py` computes geometry-corrected allocation residuals on a held-out split and tests whether they discriminate diagnostically difficult findings from easier ones within individual cases.

**Helper modules**

`reflacxloader.py` loads REFLACX transcripts, fixations, and ellipse annotations.

`fixationbuilder.py` parses raw fixation records filtered by temporal period.

`local_annotations.py` defines the core `EllipseAnnotation`, `AnatomicalRegion`, and `EllipseAttention` dataclasses.

`entity_extractor.py` extracts abnormality mentions from radiologist transcripts and aligns fixations to mention windows.

`span_abnormality_mapping.py` maps transcript terms to canonical REFLACX abnormality labels.

`scaling_law.py` defines the `ScalingLawParams` dataclass and utilities for loading pre-fitted parameters.

`cocoloader.py` and `refcocoloader.py` load the COCO-Search18 and RefCOCO-Gaze datasets respectively.

## Data Access

REFLACX is available from PhysioNet at https://physionet.org/content/reflacx-xray-localization/ and requires credentialing under the PhysioNet data use agreement. COCO-Search18 is available at https://sites.google.com/view/cocosearch/. This repository does not redistribute any data.


