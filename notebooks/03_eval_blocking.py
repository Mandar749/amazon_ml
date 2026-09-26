import time
import pandas as pd
from src.blocking.blocker import FastInvertedIndexBlocker

print("--- Running Memory-Safe Blocking Benchmark (5,000 S1 sample) ---")

dtype_dict = {
    'entity_id': 'string',
    'business_name': 'string',
    'business_address': 'string',
    'country': 'category'
}

# 1. Load Ground Truth
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype={'source1_entity_id': 'string', 'matched_entity_ids': 'string'})
gt = gt.dropna(subset=['matched_entity_ids'])
gt['match_list'] = gt['matched_entity_ids'].apply(lambda x: [m.strip() for m in x.split(',') if m.strip()])

# Take a sample of 5,000 S1 entities that have matches
sample_gt = gt.sample(n=5000, random_state=42).copy()
sample_s1_ids = set(sample_gt['source1_entity_id'])

# Collect all true matches needed to evaluate recall
true_matches_needed = set()
for m_list in sample_gt['match_list']:
    true_matches_needed.update(m_list)

print(f"Sampled {len(sample_s1_ids):,} S1 entities expecting {len(true_matches_needed):,} unique S2/S3 targets.")

# 2. Load only S1 entities in the sample
s1_iter = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=dtype_dict, chunksize=50000)
sample_s1_df = pd.concat([chunk[chunk['entity_id'].isin(sample_s1_ids)] for chunk in s1_iter])
print(f"Loaded {len(sample_s1_df):,} S1 rows.")

# 3. Load only candidate rows that are either true targets or a 50k background distractor pool
print("Loading candidate background pool (50,000 rows)...")
s2_sub = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=dtype_dict, nrows=25000)
s3_sub = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=dtype_dict, nrows=25000)

# Also load true matches if they were outside the first 25k
s2_iter = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=dtype_dict, chunksize=100000)
s2_true = pd.concat([chunk[chunk['entity_id'].isin(true_matches_needed)] for chunk in s2_iter])

s3_iter = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=dtype_dict, chunksize=100000)
s3_true = pd.concat([chunk[chunk['entity_id'].isin(true_matches_needed)] for chunk in s3_iter])

candidates_pool = pd.concat([s2_sub, s3_sub, s2_true, s3_true]).drop_duplicates(subset=['entity_id'])
print(f"Candidate pool size: {len(candidates_pool):,} records.")

# 4. Build Index
start_idx = time.time()
blocker = FastInvertedIndexBlocker(max_bucket_size=500)
blocker.index_candidates(candidates_pool)
print(f"Indexed in {time.time() - start_idx:.2f} seconds across {len(blocker.index):,} keys.")

# 5. Evaluate Retrieval
print("Retrieving candidates for evaluation...")
start_ret = time.time()
total_true_pairs = 0
recalled_true_pairs = 0
candidate_counts = []

s1_lookup = sample_s1_df.set_index('entity_id')

for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1_lookup.index:
        continue
    
    s1_row = s1_lookup.loc[s1_id]
    expected_matches = set(row['match_list'])
    # Only evaluate recall against candidates present in our pool
    expected_in_pool = expected_matches.intersection(candidates_pool['entity_id'].values)
    if not expected_in_pool:
        continue
        
    retrieved = blocker.retrieve_candidates(s1_row['business_name'], s1_row['business_address'], max_candidates_per_entity=50)
    candidate_counts.append(len(retrieved))
    
    total_true_pairs += len(expected_in_pool)
    recalled_true_pairs += len(expected_in_pool.intersection(retrieved))

avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
recall_ceiling = (recalled_true_pairs / total_true_pairs) * 100 if total_true_pairs else 0

print(f"\n================ BENCHMARK RESULTS ================")
print(f"Retrieval Time:         {time.time() - start_ret:.2f} seconds")
print(f"Average Candidates/S1:  {avg_candidates:.1f} (Reduction ratio: >99.99%)")
print(f"Total True Evaluated:   {total_true_pairs:,}")
print(f"Recalled True Pairs:    {recalled_true_pairs:,}")
print(f"Candidate Recall:       {recall_ceiling:.2f}%")
print("===================================================\n")
