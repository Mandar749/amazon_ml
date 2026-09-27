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

def extract_street_and_postal_numbers(addr: str):
    if not isinstance(addr, str) or addr.lower() == "nan":
        return set(), set()
    tokens = re.findall(r'\b\d+\b', addr)
    if not tokens:
        return set(), set()
    postal = set()
    premise = set()
    for t in tokens:
        if len(t) in (5, 6):
            postal.add(t)
        else:
            premise.add(t)
    if not premise and len(tokens) >= 2:
        premise.add(tokens[0])
        postal.update(tokens[1:])
    return premise, postal

def numeric_veto(addr1: str, addr2: str) -> bool:
    p1, post1 = extract_street_and_postal_numbers(addr1)
    p2, post2 = extract_street_and_postal_numbers(addr2)
    if len(post1) > 0 and len(post2) > 0 and len(post1 & post2) == 0:
        return True
    if len(p1) > 0 and len(p2) > 0 and len(p1 & p2) == 0:
        return True
    return False

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

print("Checking collateral damage on True Positive ground-truth pairs...")
true_pairs_total = 0
true_pairs_vetoed = 0
true_pairs_vetoed_by_country = {c: 0 for c in countries}
true_pairs_total_by_country = {c: 0 for c in countries}

for row in s1_df.itertuples(index=False):
    s1_id = row.entity_id
    c = row.country
    gt_str = gt_map.get(s1_id, "").strip()
    if not gt_str:
        continue
    for tid in gt_str.split(","):
        if tid in cand_dict:
            true_pairs_total += 1
            true_pairs_total_by_country[c] += 1
            if numeric_veto(row.business_address, cand_dict[tid][1]):
                true_pairs_vetoed += 1
                true_pairs_vetoed_by_country[c] += 1

print("\n" + "=" * 65)
print("COLLATERAL DAMAGE REPORT")
print("=" * 65)
print(f"Total True Pairs in Candidate Dict : {true_pairs_total}")
print(f"True Pairs Killed by Veto           : {true_pairs_vetoed} ({true_pairs_vetoed / max(1, true_pairs_total) * 100:.2f}%)")
for c in countries:
    tot_c = true_pairs_total_by_country[c]
    vet_c = true_pairs_vetoed_by_country[c]
    print(f"  Country {c:<6} : {vet_c}/{tot_c} killed ({vet_c / max(1, tot_c) * 100:.2f}%)")

print("\nScoring pairs with postal-aware veto active...")
entity_scored = {}
for row in s1_df.itertuples(index=False):
    s1_id = row.entity_id
    s1_name, s1_addr = row.business_name, row.business_address
    cands = blocker.retrieve_candidates(s1_name, s1_addr, max_candidates_per_entity=35)
    eval_pairs = []
    cids = []
    for cid in cands:
        if cid in cand_dict:
            c_name, c_addr = cand_dict[cid]
            if numeric_veto(s1_addr, c_addr):
                continue
            eval_pairs.append(compute_pair_features(s1_name, s1_addr, c_name, c_addr, cid))
            cids.append(cid)
    if not eval_pairs:
        entity_scored[s1_id] = []
        continue
    probs = clf.predict_proba(pd.DataFrame(eval_pairs))[:, 1]
    entity_scored[s1_id] = list(zip(cids, probs))

gt_series = pd.Series([gt_map.get(s1_id, "") for s1_id in s1_df["entity_id"]])

print("\n" + "=" * 65)
print("SWEEPING CAPS & MARGINS WITH POSTAL-AWARE VETO")
print("=" * 65)
for cap in [2, 3, 4]:
    for delta in [0.03, 0.05]:
        preds = []
        for s1_id in s1_df["entity_id"]:
            scores_by_src = {"S2": [], "S3": []}
            for cid, p in entity_scored[s1_id]:
                src = "S2" if cid.startswith("S2-") else "S3"
                scores_by_src[src].append((cid, float(p)))
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
        m = compute_per_entity_f05(gt_series, pd.Series(preds))
        print(f"Cap: {cap} | Delta: {delta:.2f} -> Macro F0.5: {m['macro_f05']:.5f} | Prec: {m['micro_precision']*100:.2f}% | Rec: {m['micro_recall']*100:.2f}%")
