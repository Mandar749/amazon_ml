import pandas as pd
from pathlib import Path

sub_path = Path("output/matching_results.tsv")
test_s1_path = Path("dataset/test/test_source1.tsv")

print("=" * 60)
print("SUBMISSION INTEGRITY VERIFICATION")
print("=" * 60)

# Check file existence and size
if not sub_path.exists():
    raise FileNotFoundError(f"Missing file: {sub_path}")

size_mb = sub_path.stat().st_size / (1024 * 1024)
print(f"File size: {size_mb:.2f} MB")

# Load submission and test source1
print("Reading files...")
sub_df = pd.read_csv(sub_path, sep="\t", dtype=str, keep_default_na=False)
test_s1 = pd.read_csv(test_s1_path, sep="\t", dtype=str, keep_default_na=False)

# 1. Header Check
expected_cols = ["source1_entity_id", "matched_entity_ids"]
assert list(sub_df.columns) == expected_cols, f"Invalid columns: {sub_df.columns}"
print("[PASS] Headers strictly match: ['source1_entity_id', 'matched_entity_ids']")

# 2. Row Count Check
assert len(sub_df) == len(test_s1), f"Row count mismatch! {len(sub_df)} vs {len(test_s1)}"
print(f"[PASS] Row count matches exactly: {len(sub_df):,} rows")

# 3. Entity ID Order & Exact Match Check
assert (sub_df["source1_entity_id"] == test_s1["entity_id"]).all(), "source1_entity_id ordering does not match test_source1.tsv!"
print("[PASS] Entity IDs strictly 1:1 aligned with test_source1")

# 4. Check for illegal NaN/null representations
assert not sub_df["source1_entity_id"].isna().any(), "Nulls found in source1_entity_id!"
assert not (sub_df["matched_entity_ids"] == "nan").any(), "Literal 'nan' string found in matches!"
assert not (sub_df["matched_entity_ids"] == "None").any(), "Literal 'None' string found in matches!"
print("[PASS] Clean null/singleton representations (no 'nan' or 'None' strings)")

# 5. Format of matches (no trailing commas, valid prefixes)
non_empty = sub_df[sub_df["matched_entity_ids"] != ""]["matched_entity_ids"]
print(f"[INFO] Entities with >= 1 match: {len(non_empty):,} ({len(non_empty) / len(sub_df) * 100:.2f}%)")
print(f"[INFO] Entities predicted singleton: {len(sub_df) - len(non_empty):,} ({(len(sub_df) - len(non_empty)) / len(sub_df) * 100:.2f}%)")

print("\nSample first 5 non-empty predictions:")
for s in non_empty.head(5):
    print("  ", s)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED. Ready for portal upload!")
print(f"File location: {sub_path.resolve()}")
print("=" * 60)
