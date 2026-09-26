import pandas as pd
import numpy as np

print('--- Loading datasets with optimized memory types ---')
dtype_dict = {
    'entity_id': 'string',
    'business_name': 'string',
    'business_address': 'string',
    'country': 'category'
}

s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=dtype_dict)
print(f'Source 1 loaded: {len(s1):,} rows')

s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=dtype_dict)
print(f'Source 2 loaded: {len(s2):,} rows')

s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=dtype_dict)
print(f'Source 3 loaded: {len(s3):,} rows')

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype={'source1_entity_id': 'string', 'matched_entity_ids': 'string'})
print(f'Ground Truth loaded: {len(gt):,} rows\n')

print('=== 1. NULL VALUES SUMMARY ===')
for name, df in [('Source 1', s1), ('Source 2', s2), ('Source 3', s3)]:
    print(f'\n{name}:')
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        print(f'  {col}: {null_pct:.2f}% null')

print('\n=== 2. COUNTRY DISTRIBUTION ===')
for name, df in [('Source 1', s1), ('Source 2', s2), ('Source 3', s3)]:
    print(f'\n{name} top countries:')
    print(df['country'].value_counts(dropna=False).head(5))

print('\n=== 3. GROUND TRUTH MATCH PROFILE ===')
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
gt['match_count'] = gt['matched_entity_ids'].apply(lambda x: 0 if x == '' else len(x.split(',')))

match_dist = gt['match_count'].value_counts().sort_index()
print('Number of matches per S1 entity distribution:')
for count, freq in match_dist.items():
    pct = (freq / len(gt)) * 100
    print(f'  {count} matches: {freq:,} ({pct:.2f}%)')

singletons = (gt['match_count'] == 0).sum()
print(f'\nTotal Singletons (0 matches): {singletons:,} ({(singletons/len(gt))*100:.2f}%)')
