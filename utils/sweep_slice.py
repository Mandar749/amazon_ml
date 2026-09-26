import sys
from pathlib import Path
sys.path.insert(0, str(Path("utils").resolve()))
import pandas as pd
from eval_macro_f05 import compute_per_entity_f05

gt_all = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt_col = "source1_entity_id" if "source1_entity_id" in gt_all.columns else "entity_id"

pred = pd.read_csv("output/val_matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
val_ids = set(pred["source1_entity_id"])
gt_slice = gt_all[gt_all[gt_col].isin(val_ids)].copy()

merged = gt_slice.merge(pred, left_on=gt_col, right_on="source1_entity_id", suffixes=("_gt", "_pred"))
metrics = compute_per_entity_f05(merged["matched_entity_ids_gt"], merged["matched_entity_ids_pred"])
print(f"Current Config (0.75 floor): Macro F0.5 = {metrics['macro_f05']:.5f} | Prec = {metrics['micro_precision']*100:.2f}% | Rec = {metrics['micro_recall']*100:.2f}%")
