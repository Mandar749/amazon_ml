import sys
from pathlib import Path
sys.path.insert(0, str(Path("utils").resolve()))
import pandas as pd
from eval_macro_f05 import compute_per_entity_f05

# Test 1: Competition Problem Statement Example
# Pred: [S2-00047, S2-00193, S3-00812], True: [S2-00047, S3-00812]
# Expected: P=0.6667, R=1.0000, F0.5=0.7143
s1_gt = pd.Series(["S2-00047,S3-00812"])
s1_pred = pd.Series(["S2-00047,S2-00193,S3-00812"])
res1 = compute_per_entity_f05(s1_gt, s1_pred)
print(f"Worked Example: F0.5 = {res1['macro_f05']:.4f} (Expected: ~0.7143)")

# Test 2: True Singleton correctly predicted empty -> must be 1.0
res2 = compute_per_entity_f05(pd.Series([""]), pd.Series([""]))
print(f"Singleton True-Negative: F0.5 = {res2['macro_f05']:.4f} (Expected: 1.0000)")

# Test 3: True Singleton wrongly matched -> must be 0.0
res3 = compute_per_entity_f05(pd.Series([""]), pd.Series(["S2-00001"]))
print(f"Singleton False-Positive: F0.5 = {res3['macro_f05']:.4f} (Expected: 0.0000)")

# Test 4: True Match predicted empty (complete miss) -> must be 0.0
res4 = compute_per_entity_f05(pd.Series(["S2-00001"]), pd.Series([""]))
print(f"Non-Singleton Complete Miss: F0.5 = {res4['macro_f05']:.4f} (Expected: 0.0000)")
