# Business Entity Resolution Pipeline

## Overview
Memory-safe entity resolution pipeline for the Amazon ML Challenge.

## Setup & Environment
pip install -r requirements.txt

## Reproduction Steps
$env:PYTHONPATH = 'code/business_entity_resolution'
python code/business_entity_resolution/src/models/train_lgbm.py
python code/business_entity_resolution/src/pipeline/predict.py
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
