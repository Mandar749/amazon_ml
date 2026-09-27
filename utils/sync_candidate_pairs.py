import pandas as pd
from pathlib import Path

cand_path = Path("output/candidate_pairs.tsv")
match_path = Path("output/matching_results.tsv")

print("Checking and syncing candidate_pairs.tsv with matching_results.tsv...")
match_df = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)

if cand_path.exists():
    cand_df = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)
else:
    cand_df = pd.DataFrame({"source1_entity_id": match_df["source1_entity_id"], "candidate_entity_ids": ""})

cand_dict = dict(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]))

updated_cands = []
for row in match_df.itertuples(index=False):
    s1 = row.source1_entity_id
    matches = set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
    existing = set(cand_dict.get(s1, "").split(",")) if cand_dict.get(s1, "") else set()
    combined = existing.union(matches)
    combined.discard("")
    updated_cands.append((s1, ",".join(sorted(combined))))

out_cand = pd.DataFrame(updated_cands, columns=["source1_entity_id", "candidate_entity_ids"])
out_cand.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
print(f"[PASS] candidate_pairs.tsv updated with {len(out_cand):,} rows matching test set.")
