import sys
from pathlib import Path
sys.path.insert(0, str(Path("code/business_entity_resolution").resolve()))
sys.path.insert(0, str(Path("utils").resolve()))

import pandas as pd
from src.blocking.blocker import FastInvertedIndexBlocker
from src.blocking.normalize import extract_name_tokens, extract_numbers_and_anchors

def to_flat_str_set(items):
    out = set()
    for item in items:
        if isinstance(item, (list, tuple, set)):
            out.update(str(x) for x in item)
        else:
            out.add(str(item))
    return out

N = 5000
print(f"Loading {N} validation entities and candidate pool...")
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

print(f"Indexing {len(cand_df):,} candidates...")
blocker = FastInvertedIndexBlocker(max_bucket_size=1000)
blocker.index_candidates(cand_df)

stats_by_country = {c: {"entities_with_gt": 0, "entities_hit": 0, "total_pairs": 0, "pairs_hit": 0} for c in countries}
missed_samples = []

print("Evaluating retrieval across entities...")
for row in s1_df.itertuples(index=False):
    s1_id = row.entity_id
    c = row.country
    gt_str = gt_map.get(s1_id, "").strip()
    if not gt_str:
        continue
    
    true_pairs = set(gt_str.split(","))
    n_true = len(true_pairs)
    stats_by_country[c]["entities_with_gt"] += 1
    stats_by_country[c]["total_pairs"] += n_true
    
    cands = set(blocker.retrieve_candidates(row.business_name, row.business_address, max_candidates_per_entity=35))
    hits = true_pairs & cands
    n_hits = len(hits)
    
    stats_by_country[c]["pairs_hit"] += n_hits
    if n_hits > 0:
        stats_by_country[c]["entities_hit"] += 1
        
    missed = true_pairs - cands
    for m_id in missed:
        if m_id in cand_dict and len(missed_samples) < 15:
            c_name, c_addr = cand_dict[m_id]
            s1_tokens = to_flat_str_set(extract_name_tokens(row.business_name))
            c_tokens = to_flat_str_set(extract_name_tokens(c_name))
            s1_nums = to_flat_str_set(extract_numbers_and_anchors(row.business_address))
            c_nums = to_flat_str_set(extract_numbers_and_anchors(c_addr))
            
            missed_samples.append({
                "country": c,
                "s1_id": s1_id,
                "s1_name": row.business_name,
                "s1_addr": row.business_address,
                "m_id": m_id,
                "m_name": c_name,
                "m_addr": c_addr,
                "shared_name_tokens": list(s1_tokens & c_tokens),
                "shared_nums": list(s1_nums & c_nums)
            })

print("\n" + "=" * 65)
print("1. BLOCKING RECALL RECONCILIATION")
print("=" * 65)
tot_ent = sum(s["entities_with_gt"] for s in stats_by_country.values())
tot_ent_hit = sum(s["entities_hit"] for s in stats_by_country.values())
tot_pairs = sum(s["total_pairs"] for s in stats_by_country.values())
tot_pairs_hit = sum(s["pairs_hit"] for s in stats_by_country.values())

print(f"Overall Entity-Level Recall (>=1 hit) : {tot_ent_hit / tot_ent * 100:.2f}% ({tot_ent_hit}/{tot_ent})")
print(f"Overall Pair-Level Recall   (all hits): {tot_pairs_hit / tot_pairs * 100:.2f}% ({tot_pairs_hit}/{tot_pairs})")

print("\n" + "=" * 65)
print("2. PER-COUNTRY BLOCKING BREAKDOWN")
print("=" * 65)
for c in countries:
    st = stats_by_country[c]
    ent_rec = st["entities_hit"] / max(1, st["entities_with_gt"]) * 100
    pair_rec = st["pairs_hit"] / max(1, st["total_pairs"]) * 100
    print(f"Country {c:<6} | Entity Recall: {ent_rec:.2f}% | Pair Recall: {pair_rec:.2f}% | True Pairs: {st['total_pairs']:,}")

print("\n" + "=" * 65)
print("3. SAMPLE MISSED PAIRS (Why did they get dropped?)")
print("=" * 65)
for i, m in enumerate(missed_samples[:8], 1):
    print(f"[{i}] Country: {m['country']}")
    print(f"    S1  : {m['s1_name']} | {m['s1_addr']}")
    print(f"    TRUE: {m['m_name']} | {m['m_addr']}")
    print(f"    Shared Tokens: {m['shared_name_tokens']} | Shared Numbers: {m['shared_nums']}")
    print("-" * 65)
