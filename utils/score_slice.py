import sys
from pathlib import Path
sys.path.insert(0, str(Path("utils").resolve()))
import pandas as pd
from eval_macro_f05 import compute_per_entity_f05

# 1. Load predictions slice
pred = pd.read_csv("output/val_matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
val_ids = set(pred["source1_entity_id"])

# 2. Load ground truth and filter strictly to the 5,000 evaluated entities
gt_all = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt_col = "source1_entity_id" if "source1_entity_id" in gt_all.columns else "entity_id"
gt_slice = gt_all[gt_all[gt_col].isin(val_ids)].copy()

merged = gt_slice.merge(pred, left_on=gt_col, right_on="source1_entity_id", suffixes=("_gt", "_pred"))

metrics = compute_per_entity_f05(merged["matched_entity_ids_gt"], merged["matched_entity_ids_pred"])

f05 = metrics["macro_f05"]
prec = metrics["micro_precision"] * 100
rec = metrics["micro_recall"] * 100
s_acc = metrics["singleton_accuracy"] * 100
s_cnt = metrics["singletons_count"]
tot = metrics["total_evaluated"]

print("=" * 45)
print(f"  LOCAL SLICE MACRO F_0.5 : {f05:.5f}")
print("=" * 45)
print(f"  Entities Evaluated      : {tot:,}")
print(f"  Singleton Accuracy      : {s_acc:.2f}% ({s_cnt:,} singletons)")
print(f"  Pairwise Precision      : {prec:.2f}%")
print(f"  Pairwise Recall         : {rec:.2f}%")
print("=" * 45)
