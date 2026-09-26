import time
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from src.blocking.blocker import FastInvertedIndexBlocker
from src.features.build_features import compute_pair_features

def macro_f_beta(y_true, y_pred, beta=0.5):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    beta_sq = beta ** 2
    if precision + recall == 0:
        return 0.0, precision, recall
    f_score = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
    return f_score, precision, recall

print("=== Building Fast & Memory-Safe LightGBM Training Set ===")
dtype_dict = {
    'entity_id': 'string',
    'business_name': 'string',
    'business_address': 'string',
    'country': 'category'
}

# 1. Sample Ground Truth (10,000 entities for fast, stable training)
print("Loading Ground Truth...")
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype={'source1_entity_id': 'string', 'matched_entity_ids': 'string'}).fillna('')
gt['match_list'] = gt['matched_entity_ids'].apply(lambda x: [m.strip() for m in x.split(',') if m.strip()])

matched_sample = gt[gt['match_list'].apply(len) > 0].sample(n=9500, random_state=42)
singleton_sample = gt[gt['match_list'].apply(len) == 0].sample(n=500, random_state=42)
train_gt = pd.concat([matched_sample, singleton_sample]).sample(frac=1.0, random_state=42)

train_s1_ids = set(train_gt['source1_entity_id'])
all_true_targets = {m for m_list in train_gt['match_list'] for m in m_list}
print(f"Sampled {len(train_s1_ids):,} S1 entities targeting {len(all_true_targets):,} true matches.")

# 2. Load S1 data into a native Python dictionary
s1_iter = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=dtype_dict, chunksize=50000)
s1_df = pd.concat([chunk[chunk['entity_id'].isin(train_s1_ids)] for chunk in s1_iter])
s1_dict = {row.entity_id: (row.business_name, row.business_address) for row in s1_df.itertuples(index=False)}
print(f"Loaded {len(s1_dict):,} S1 records into memory lookup.")

# 3. Load Candidate Data into a native Python dictionary and set
print("Loading candidate records...")
s2_sub = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=dtype_dict, nrows=40000)
s3_sub = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=dtype_dict, nrows=40000)

s2_iter = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=dtype_dict, chunksize=100000)
s2_true = pd.concat([chunk[chunk['entity_id'].isin(all_true_targets)] for chunk in s2_iter])

s3_iter = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=dtype_dict, chunksize=100000)
s3_true = pd.concat([chunk[chunk['entity_id'].isin(all_true_targets)] for chunk in s3_iter])

cand_df = pd.concat([s2_sub, s3_sub, s2_true, s3_true]).drop_duplicates(subset=['entity_id'])
cand_dict = {row.entity_id: (row.business_name, row.business_address) for row in cand_df.itertuples(index=False)}
cand_id_set = set(cand_dict.keys())
print(f"Candidate pool active: {len(cand_dict):,} records.")

# 4. Build Blocker Index
print("Indexing candidate pool...")
blocker = FastInvertedIndexBlocker(max_bucket_size=1000)
blocker.index_candidates(cand_df)

# 5. Extract Feature Matrix with pure Python set operations
print("Extracting feature vectors...")
t_start = time.time()
feature_rows = []
labels = []
s1_groups = []

total_entities = len(train_gt)
for idx, row in enumerate(train_gt.itertuples(index=False), 1):
    s1_ent_id = row.source1_entity_id
    if s1_ent_id not in s1_dict:
        continue
    
    s1_name, s1_addr = s1_dict[s1_ent_id]
    true_set = set(row.match_list)
    
    # Retrieve candidates and add true targets present in cand_id_set
    cands = blocker.retrieve_candidates(s1_name, s1_addr, max_candidates_per_entity=30)
    valid_true = true_set.intersection(cand_id_set)
    cands.update(valid_true)

    for cid in cands:
        if cid not in cand_dict:
            continue
        c_name, c_addr = cand_dict[cid]
        feats = compute_pair_features(s1_name, s1_addr, c_name, c_addr, cid)
        feature_rows.append(feats)
        labels.append(1 if cid in true_set else 0)
        s1_groups.append(s1_ent_id)

    if idx % 2000 == 0 or idx == total_entities:
        elapsed = time.time() - t_start
        print(f"  Processed {idx:,}/{total_entities:,} entities ({len(feature_rows):,} pairs, {elapsed:.1f}s elapsed)...")

X = pd.DataFrame(feature_rows)
y = np.array(labels)
print(f"\nFeature extraction complete in {time.time() - t_start:.2f}s!")
print(f"Total Rows: {X.shape[0]:,}, Positive Pairs: {np.sum(y):,} ({np.mean(y)*100:.2f}%)")

# 6. Train-Val Split by S1 Group
unique_s1 = np.unique(s1_groups)
train_entities, val_entities = train_test_split(unique_s1, test_size=0.2, random_state=42)
train_mask = np.isin(s1_groups, train_entities)
val_mask = ~train_mask

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

# 7. Train LightGBM Classifier
print("\nTraining LightGBM Classifier...")
clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])

# 8. Optimize Threshold for Macro F_0.5
val_preds_prob = clf.predict_proba(X_val)[:, 1]

best_thresh = 0.5
best_f05 = 0.0
best_prec = 0.0
best_rec = 0.0

print("\n--- Tuning Decision Threshold for Macro F_0.5 ---")
for thresh in np.arange(0.20, 0.85, 0.05):
    preds = (val_preds_prob >= thresh).astype(int)
    f05, prec, rec = macro_f_beta(y_val, preds, beta=0.5)
    print(f"Threshold: {thresh:.2f} | F_0.5: {f05:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    if f05 > best_f05:
        best_f05 = f05
        best_thresh = thresh
        best_prec = prec
        best_rec = rec

print(f"\n>>> Optimal Threshold: {best_thresh:.2f} with Validation F_0.5 = {best_f05:.4f} (Precision: {best_prec:.4f}, Recall: {best_rec:.4f})")

# 9. Save Artifacts
os.makedirs('output', exist_ok=True)
joblib.dump(clf, 'output/lgbm_entity_resolver.joblib')
with open('output/best_threshold.txt', 'w') as f:
    f.write(str(best_thresh))

print("\nModel artifact saved cleanly to output/lgbm_entity_resolver.joblib")
