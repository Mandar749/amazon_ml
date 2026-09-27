import pandas as pd

sub = pd.read_csv("output/submission_smoke100.tsv", sep="\t", dtype=str, keep_default_na=False)
s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False).head(100)
merged = pd.merge(s1, sub, left_on="entity_id", right_on="source1_entity_id")

# Load candidate lookup to display the actual matched names/addresses
test_s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t", nrows=200000, dtype=str, keep_default_na=False)
test_s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t", nrows=200000, dtype=str, keep_default_na=False)
cand_df = pd.concat([test_s2, test_s3]).drop_duplicates(subset=["entity_id"])
cand_map = dict(zip(cand_df["entity_id"], zip(cand_df["business_name"], cand_df["business_address"])))

print("=" * 70)
print("SPOT-CHECK MATCH SAMPLES BY COUNTRY")
print("=" * 70)

for country in ["France", "US", "India"]:
    c_all = merged[merged["country"] == country]
    subset = c_all[c_all["matched_entity_ids"] != ""]
    print(f"\n--- Country: {country} ({len(subset)} matched / {len(c_all)} total entities) ---")
    
    for row in subset.head(2).itertuples():
        print(f"S1:   {row.business_name} | {row.business_address}")
        for match_id in row.matched_entity_ids.split(","):
            if match_id in cand_map:
                m_name, m_addr = cand_map[match_id]
                print(f"PRED: [{match_id}] {m_name} | {m_addr}")
            else:
                print(f"PRED: [{match_id}]")
        print("-" * 50)
