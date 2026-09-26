import pandas as pd

dtype_dict = {
    'entity_id': 'string',
    'business_name': 'string',
    'business_address': 'string',
    'country': 'category'
}

print('Loading datasets for match inspection...')
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=dtype_dict).set_index('entity_id')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=dtype_dict).set_index('entity_id')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=dtype_dict).set_index('entity_id')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype={'source1_entity_id': 'string', 'matched_entity_ids': 'string'})

# Sample 10 matches (5 India, 5 US to see both behaviors)
gt_merged = gt.merge(s1[['country']], left_on='source1_entity_id', right_index=True)

sample_in = gt_merged[(gt_merged['country'] == 'India') & (gt_merged['matched_entity_ids'].str.contains(',', na=False))].sample(5, random_state=42)
sample_us = gt_merged[(gt_merged['country'] == 'US') & (gt_merged['matched_entity_ids'].str.contains(',', na=False))].sample(5, random_state=42)

samples = pd.concat([sample_in, sample_us])

print('\n================ SAMPLE MATCH GROUPS ================\n')
for idx, row in samples.iterrows():
    s1_id = row['source1_entity_id']
    m_ids = row['matched_entity_ids'].split(',')
    
    if s1_id in s1.index:
        s1_row = s1.loc[s1_id]
        c = s1_row['country']
        name = s1_row['business_name']
        addr = s1_row['business_address']
        print(f"[{c}] S1 ({s1_id}):")
        print(f"   Name:    {name}")
        print(f"   Address: {addr}")
    
    for mid in m_ids:
        src = s2 if mid.startswith('S2') else s3
        if mid in src.index:
            m_row = src.loc[mid]
            m_name = m_row['business_name']
            m_addr = m_row['business_address']
            print(f" -> MATCH {mid}:")
            print(f"      Name:    {m_name}")
            print(f"      Address: {m_addr}")
    print('-' * 70)
