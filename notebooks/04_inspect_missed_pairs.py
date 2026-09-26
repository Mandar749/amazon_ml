import pandas as pd
from src.blocking.blocker import FastInvertedIndexBlocker, generate_blocking_keys

dtype_dict = {
    'entity_id': 'string',
    'business_name': 'string',
    'business_address': 'string',
    'country': 'category'
}

print("Loading data to analyze missed candidate pairs...")
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype={'source1_entity_id': 'string', 'matched_entity_ids': 'string'}).dropna()
gt['match_list'] = gt['matched_entity_ids'].apply(lambda x: [m.strip() for m in x.split(',') if m.strip()])
sample_gt = gt.sample(n=3000, random_state=42)

sample_s1_ids = set(sample_gt['source1_entity_id'])
true_targets = {m for m_list in sample_gt['match_list'] for m in m_list}

s1_iter = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=dtype_dict, chunksize=50000)
s1_df = pd.concat([c[c['entity_id'].isin(sample_s1_ids)] for c in s1_iter]).set_index('entity_id')

s2_iter = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=dtype_dict, chunksize=100000)
s2_true = pd.concat([c[c['entity_id'].isin(true_targets)] for c in s2_iter])

s3_iter = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=dtype_dict, chunksize=100000)
s3_true = pd.concat([c[c['entity_id'].isin(true_targets)] for c in s3_iter])

cand_df = pd.concat([s2_true, s3_true]).drop_duplicates('entity_id')
blocker = FastInvertedIndexBlocker(max_bucket_size=1000)
blocker.index_candidates(cand_df)
cand_lookup = cand_df.set_index('entity_id')

print("\n--- FINDING MISSED MATCHES ---")
missed_count = 0
for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1_df.index:
        continue
    s1_row = s1_df.loc[s1_id]
    expected = set(row['match_list']).intersection(cand_lookup.index)
    retrieved = blocker.retrieve_candidates(s1_row['business_name'], s1_row['business_address'], max_candidates_per_entity=50)
    
    missed = expected - retrieved
    if missed:
        for m_id in list(missed)[:1]:
            m_row = cand_lookup.loc[m_id]
            print(f"S1 ({s1_id}):")
            print(f"   Name: {s1_row['business_name']}")
            print(f"   Addr: {s1_row['business_address']}")
            print(f"S1 Keys: {generate_blocking_keys(s1_row['business_name'], s1_row['business_address'])}")
            print(f"Target ({m_id}):")
            print(f"   Name: {m_row['business_name']}")
            print(f"   Addr: {m_row['business_address']}")
            print(f"Target Keys: {generate_blocking_keys(m_row['business_name'], m_row['business_address'])}")
            print("-" * 70)
            missed_count += 1
            if missed_count >= 5:
                break
    if missed_count >= 5:
        break
