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

entity_scored = {}
for row in s1_df.itertuples(index=False):
    s1_id = row.entity_id
    s1_name, s1_addr = row.business_name, row.business_address
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
        entity_scored[s1_id] = []
        continue
    probs = clf.predict_proba(pd.DataFrame(eval_pairs))[:, 1]
    
    # Store candidate id, prob, and number conflict status
    scored = []
    for cid, p in zip(cids, probs):
        c_addr = cand_dict[cid][1]
        c_nums = get_all_numbers(c_addr)
        has_num_conflict = (len(s1_nums) > 0 and len(c_nums) > 0 and len(s1_nums & c_nums) == 0)
        scored.append((cid, float(p), has_num_conflict))
    entity_scored[s1_id] = scored

gt_series = [gt_map.get(s1_id, "") for s1_id in s1_df["entity_id"]]

print("--- Testing Numeric Veto with Different Caps & Margins ---")
for cap in [2, 3, 4]:
    for delta in [0.03, 0.05]:
        preds = []
        for s1_id in s1_df["entity_id"]:
            scores_by_src = {"S2": [], "S3": []}
            for cid, p, conflict in entity_scored[s1_id]:
                if conflict:
                    continue
                src = "S2" if cid.startswith("S2-") else "S3"
                scores_by_src[src].append((cid, p))
            selected = []
            for src in ("S2", "S3"):
                c_list = scores_by_src[src]
                if not c_list:
                    continue
                c_list.sort(key=lambda x: x[1], reverse=True)
                top_p = c_list[0][1]
                if top_p < 0.85:
                    continue
                for cid, p in c_list[:cap]:
                    if p >= 0.85 and p >= (top_p - delta):
                        selected.append(cid)
            preds.append(",".join(selected))
        m = compute_per_entity_f05(pd.Series(gt_series), pd.Series(preds))
        print(f"Cap: {cap} | Delta: {delta:.2f} -> Macro F0.5: {m['macro_f05']:.5f} | Prec: {m['micro_precision']*100:.2f}% | Rec: {m['micro_recall']*100:.2f}%")
