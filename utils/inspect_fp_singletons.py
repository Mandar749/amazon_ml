import sys
from pathlib import Path
sys.path.insert(0, str(Path("code/business_entity_resolution").resolve()))
sys.path.insert(0, str(Path("utils").resolve()))

import pandas as pd
import joblib
from src.blocking.blocker import FastInvertedIndexBlocker
from src.features.build_features import compute_pair_features

N = 5000
print(f"Loading {N} validation entities...")
s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=N)
gt_all = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt_col = "source1_entity_id" if "source1_entity_id" in gt_all.columns else "entity_id"

val_ids = set(s1_df["entity_id"])
gt_slice = gt_all[gt_all[gt_col].isin(val_ids)].copy()
gt_map = dict(zip(gt_slice[gt_col], gt_slice["matched_entity_ids"]))

countries = list(s1_df["country"].unique())
s2_chunks = [c for c in pd.read_csv("dataset/train/train_source2.tsv", sep="\t", chunksize=200000) if not c[c["country"].isin(countries)].empty]
s2_df = pd.concat([c[c["country"].isin(countries)] for c in s2_chunks], ignore_index=True)
s3_chunks = [c for c in pd.read_csv("dataset/train/train_source3.tsv", sep="\t", chunksize=200000) if not c[c["country"].isin(countries)].empty]
s3_df = pd.concat([c[c["country"].isin(countries)] for c in s3_chunks], ignore_index=True)

cand_df = pd.concat([s2_df, s3_df]).drop_duplicates(subset=["entity_id"])
cand_dict = {row.entity_id: (row.business_name, row.business_address) for row in cand_df.itertuples(index=False)}

blocker = FastInvertedIndexBlocker(max_bucket_size=1000)
blocker.index_candidates(cand_df)
clf = joblib.load("output/lgbm_entity_resolver.joblib")

floor = 0.85
margin = 0.03
fp_singleton_cases = []

print("Searching for false-positive singleton predictions...")
for row in s1_df.itertuples(index=False):
    s1_id = row.entity_id
    gt_str = gt_map.get(s1_id, "").strip()
    # Is it a true singleton?
    if gt_str != "":
        continue
        
    s1_name, s1_addr = row.business_name, row.business_address
    cands = blocker.retrieve_candidates(s1_name, s1_addr, max_candidates_per_entity=35)
    eval_pairs = []
    cids = []
    for cid in cands:
        if cid in cand_dict:
            c_name, c_addr = cand_dict[cid]
            eval_pairs.append(compute_pair_features(s1_name, s1_addr, c_name, c_addr, cid))
            cids.append(cid)
            
    if not eval_pairs:
        continue
        
    probs = clf.predict_proba(pd.DataFrame(eval_pairs))[:, 1]
    
    # Selection rule
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
                selected.append((cid, p))
                
    if selected:
        fp_singleton_cases.append({
            "country": row.country,
            "s1_id": s1_id,
            "s1_name": s1_name,
            "s1_addr": s1_addr,
            "matched_cands": [(cid, p, cand_dict[cid][0], cand_dict[cid][1]) for cid, p in selected]
        })

print("\n" + "=" * 65)
print(f"SINGLETON FALSE POSITIVE INSPECTION (Found {len(fp_singleton_cases)} cases)")
print("=" * 65)
for i, case in enumerate(fp_singleton_cases[:8], 1):
    print(f"[{i}] Country: {case['country']} | S1 Entity: {case['s1_id']}")
    print(f"    S1   : {case['s1_name']} | {case['s1_addr']}")
    for cid, p, c_name, c_addr in case["matched_cands"]:
        print(f"    PRED : [{cid}] (p={p:.4f})")
        print(f"           {c_name} | {c_addr}")
    print("-" * 65)
