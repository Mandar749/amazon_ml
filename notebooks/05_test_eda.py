import pandas as pd

dtype_dict = {
    'entity_id': 'string',
    'business_name': 'string',
    'business_address': 'string',
    'country': 'category'
}

for name in ['test_source1', 'test_source2', 'test_source3']:
    path = f'dataset/test/{name}.tsv'
    df = pd.read_csv(path, sep='\t', dtype=dtype_dict)
    print(f"=== {name} ({len(df):,} rows) ===")
    print(df['country'].value_counts())
    print()
