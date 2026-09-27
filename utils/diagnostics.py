import sys
from pathlib import Path
# Insert the root of the project sub-package so "src" is importable directly
sys.path.insert(0, str(Path("code/business_entity_resolution").resolve()))
sys.path.insert(0, str(Path("utils").resolve()))

import pandas as pd
import numpy as np
import joblib
from eval_macro_f05 import compute_per_entity_f05
from src.blocking.blocker import FastInvertedIndexBlocker
from src.features.build_features import compute_pair_features

N = 5000
print(f"Loading data slice ({N} entities)...")
s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=N)
gt_all = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt_col = "source1_entity_id" if "source1_entity_id" in gt_all.columns else "entity_id"

val_ids = set(s1_df["entity_id"])
gt_slice = gt_all[gt_all[gt_col].isin(val_ids)].copy()
gt_map = dict(zip(gt_slice[gt_col], gt_slice["matched_entity_ids"]))

countries = s1_df["country"].unique()
print(f"Loading candidate chunks for countries: {list(countries)}...")
s2_chunks = [c for c in pd.read_csv("dataset/train/train_source2.tsv", sep="\t", chunksize=200000) if not c[c["country"].isin(countries)].empty]
s2_df = pd.concat([c[c["country"].isin(countries)] for c in s2_chunks], ignore_index=True)
s3_chunks = [c for c in pd.read_csv("dataset/train/train_source3.tsv", sep="\t", chunksize=200000) if not c[c["country"].isin(countries)].empty]
s3_df = pd.concat([c[c["country"].isin(countries)] for c in s3_chunks], ignore_index=True)

cand_df = pd.concat([s2_df, s3_df]).drop_duplicates(subset=["entity_id"])
cand_dict = {row.entity_id: (row.business_name, row.business_address) for row in cand_df.itertuples(index=False)}

print(f"Indexing {len(cand_df):,} candidates...")
blocker = FastInvertedIndexBlocker(max_bucket_size=1000)
blocker.index_candidates(cand_df)
clf = joblib.load("output/lgbm_entity_resolver.joblib")

total_true_pairs = 0
recalled_in_blocking = 0
passed_model_60 = 0
passed_model_85 = 0
survived_selection = 0

entity_diagnostics = []

floor = 0.85
margin = 0.03

print("Running diagnostics over slice...")
for row in s1_df.itertuples(index=False):
    s1_id = row.entity_id
    s1_name, s1_addr = row.business_name, row.business_address
    country = row.country
    
    true_matches_str = gt_map.get(s1_id, "").strip()
    true_matches = set(true_matches_str.split(",")) if true_matches_str else set()
    total_true_pairs += len(true_matches)
    
    cands = blocker.retrieve_candidates(s1_name, s1_addr, max_candidates_per_entity=35)
    cands_set = set(cands)
    recalled_in_blocking += len(true_matches & cands_set)
    
    eval_pairs = []
    cids = []
    for cid in cands:
        if cid in cand_dict:
            c_name, c_addr = cand_dict[cid]
            eval_pairs.append(compute_pair_features(s1_name, s1_addr, c_name, c_addr, cid))
            cids.append(cid)
            
    if not eval_pairs:
        entity_diagnostics.append({"id": s1_id, "country": country, "gt": true_matches_str, "pred": ""})
        continue
        
    probs = clf.predict_proba(pd.DataFrame(eval_pairs))[:, 1]
    
    for cid, p in zip(cids, probs):
        if cid in true_matches:
            if p >= 0.60: passed_model_60 += 1
            if p >= 0.85: passed_model_85 += 1

    scores_by_src = {"S2": [], "S3": []}
    for cid, p in zip(cids, probs):
        src = "S2" if cid.startswith("S2-") else "S3"
        scores_by_src[src].append((cid, float(p)))
        
    selected = []
    for src in ("S2", "S3"):
        c_list = scores_by_src[src]
        if not c_list:
            continue
        c_list.sort(key=lambda x: x[1], reverse=True)
        top_p = c_list[0][1]
        if top_p < floor:
            continue
        for cid, p in c_list[:2]:
            if p >= floor and p >= (top_p - margin):
                selected.append(cid)
                
    survived_selection += len(true_matches & set(selected))
    entity_diagnostics.append({"id": s1_id, "country": country, "gt": true_matches_str, "pred": ",".join(selected)})

diag_df = pd.DataFrame(entity_diagnostics)

print("\n" + "=" * 55)
print("1. SINGLETON ANALYSIS")
print("=" * 55)
singletons = diag_df[diag_df["gt"] == ""]
n_singletons = len(singletons)
fp_singletons = singletons[singletons["pred"] != ""]
print(f"Total True Singletons : {n_singletons} ({n_singletons / len(diag_df) * 100:.2f}% of entities)")
print(f"False-Positive Leaks  : {len(fp_singletons)} ({len(fp_singletons) / max(1, n_singletons) * 100:.2f}% leaked)")
print(f"Clean Singletons      : {n_singletons - len(fp_singletons)} ({(n_singletons - len(fp_singletons)) / max(1, n_singletons) * 100:.2f}% accuracy)")

print("\n" + "=" * 55)
print("2. RECALL FUNNEL (Where are the true pairs lost?)")
print("=" * 55)
print(f"Total True Ground-Truth Pairs : {total_true_pairs}")
print(f"  Stage 1: Passed Blocker     : {recalled_in_blocking} ({recalled_in_blocking / max(1, total_true_pairs) * 100:.2f}%)")
print(f"  Stage 2: Model Score >= 0.60: {passed_model_60} ({passed_model_60 / max(1, total_true_pairs) * 100:.2f}%)")
print(f"  Stage 3: Model Score >= 0.85: {passed_model_85} ({passed_model_85 / max(1, total_true_pairs) * 100:.2f}%)")
print(f"  Stage 4: Survived Selection : {survived_selection} ({survived_selection / max(1, total_true_pairs) * 100:.2f}%)")

print("\n" + "=" * 55)
print("3. COUNTRY-BY-COUNTRY MACRO F0.5")
print("=" * 55)
for c, group in diag_df.groupby("country"):
    c_metrics = compute_per_entity_f05(group["gt"], group["pred"])
    print(f"Country {c:<6} | Macro F0.5: {c_metrics['macro_f05']:.5f} | Precision: {c_metrics['micro_precision']*100:.2f}% | Recall: {c_metrics['micro_recall']*100:.2f}% (N={len(group)})")
print("=" * 55)
