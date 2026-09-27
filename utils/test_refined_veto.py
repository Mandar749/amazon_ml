import sys
from pathlib import Path
sys.path.insert(0, str(Path("code/business_entity_resolution").resolve()))
sys.path.insert(0, str(Path("utils").resolve()))

import pandas as pd
import re
import joblib
from eval_macro_f05 import compute_per_entity_f05
from src.blocking.blocker import FastInvertedIndexBlocker
from src.features.build_features import compute_pair_features

def get_all_numbers(addr):
    if not isinstance(addr, str) or addr.lower() == "nan":
        return set()
    return set(re.findall(r'\b\d+\b', addr))

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

gt_list = []
baseline_preds = []
refined_veto_preds = []

print("Running refined set-intersection number simulation...")
for row in s1_df.itertuples(index=False):
    s1_id = row.entity_id
    s1_name, s1_addr = row.business_name, row.business_address
    gt_str = gt_map.get(s1_id, "").strip()
    gt_list.append(gt_str)
    
    s1_nums = get_all_numbers(s1_addr)
    
    cands = blocker.retrieve_candidates(s1_name, s1_addr, max_candidates_per_entity=35)
    eval_pairs = []
    cids = []
    for cid in cands:
        if cid in cand_dict:
            c_name, c_addr = cand_dict[cid]
            eval_pairs.append(compute_pair_features(s1_name, s1_addr, c_name, c_addr, cid))
            cids.append(cid)
            
    if not eval_pairs:
        baseline_preds.append("")
        refined_veto_preds.append("")
        continue
        
    probs = clf.predict_proba(pd.DataFrame(eval_pairs))[:, 1]
    
    def select_matches(mode="base"):
        scores_by_src = {"S2": [], "S3": []}
        for cid, p in zip(cids, probs):
            if mode == "refined":
                c_addr = cand_dict[cid][1]
                c_nums = get_all_numbers(c_addr)
                # If both addresses contain numbers, but have ZERO numbers in common -> Veto
                if len(s1_nums) > 0 and len(c_nums) > 0 and len(s1_nums & c_nums) == 0:
                    continue
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
        return ",".join(selected)

    baseline_preds.append(select_matches("base"))
    refined_veto_preds.append(select_matches("refined"))

m_base = compute_per_entity_f05(pd.Series(gt_list), pd.Series(baseline_preds))
m_ref = compute_per_entity_f05(pd.Series(gt_list), pd.Series(refined_veto_preds))

print("\n" + "=" * 65)
print("REFINED NUMBER OVERLAP TEST (Set Intersection)")
print("=" * 65)
print(f"Baseline (Competitive 0.85): Macro F0.5 = {m_base['macro_f05']:.5f} | Prec: {m_base['micro_precision']*100:.2f}% | Rec: {m_base['micro_recall']*100:.2f}%")
print(f"Set-Intersection Veto       : Macro F0.5 = {m_ref['macro_f05']:.5f} | Prec: {m_ref['micro_precision']*100:.2f}% | Rec: {m_ref['micro_recall']*100:.2f}%")
