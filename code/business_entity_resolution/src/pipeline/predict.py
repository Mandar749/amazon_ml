import time
import os
import joblib
import pandas as pd
from src.blocking.blocker import FastInvertedIndexBlocker
from src.features.build_features import compute_pair_features

DTYPE_DICT = {
    'entity_id': 'string',
    'business_name': 'string',
    'business_address': 'string',
    'country': 'category'
}

def run_inference():
    print("=== Amazon ML Challenge: Entity Resolution Inference Pipeline ===")
    start_total = time.time()
    
    # 1. Load Model and Decision Threshold
    model_path = 'output/lgbm_entity_resolver.joblib'
    thresh_path = 'output/best_threshold.txt'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing trained model artifact: {model_path}")
        
    clf = joblib.load(model_path)
    threshold = 0.70
    if os.path.exists(thresh_path):
        with open(thresh_path, 'r') as f:
            threshold = float(f.read().strip())
    print(f"Loaded LightGBM model. Operating Decision Threshold: {threshold:.2f}")

    # Output file destinations
    os.makedirs('output', exist_ok=True)
    matching_file = 'output/matching_results.tsv'
    candidate_file = 'output/candidate_pairs.tsv'

    f_match = open(matching_file, 'w', encoding='utf-8', buffering=1048576)
    f_cand = open(candidate_file, 'w', encoding='utf-8', buffering=1048576)

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # 2. Inspect Countries present in test set
    print("Inspecting test country partitions...")
    s1_all = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', dtype=DTYPE_DICT)
    countries = s1_all['country'].dropna().unique().tolist()
    print(f"Discovered test countries: {countries}")

    total_s1_processed = 0

    for country in countries:
        print(f"\n================ Processing Country: {country} ================")
        t_country = time.time()
        
        # Filter S1 records for active country
        s1_country = s1_all[s1_all['country'] == country]
        s1_records = [(row.entity_id, row.business_name, row.business_address) for row in s1_country.itertuples(index=False)]
        print(f"  Source 1 [{country}]: {len(s1_records):,} entities")

        # Load S2 and S3 candidates strictly for the active country
        print(f"  Loading S2 and S3 records for [{country}]...")
        s2_chunks = [c for c in pd.read_csv('dataset/test/test_source2.tsv', sep='\t', dtype=DTYPE_DICT, chunksize=250000) if not c[c['country'] == country].empty]
        s2_df = pd.concat([c[c['country'] == country] for c in s2_chunks]) if s2_chunks else pd.DataFrame()
        del s2_chunks

        s3_chunks = [c for c in pd.read_csv('dataset/test/test_source3.tsv', sep='\t', dtype=DTYPE_DICT, chunksize=250000) if not c[c['country'] == country].empty]
        s3_df = pd.concat([c[c['country'] == country] for c in s3_chunks]) if s3_chunks else pd.DataFrame()
        del s3_chunks

        cand_df = pd.concat([s2_df, s3_df]).drop_duplicates(subset=['entity_id'])
        del s2_df, s3_df

        print(f"  Candidate Pool [{country}]: {len(cand_df):,} records")

        # Build Inverted Index Blocker
        print(f"  Building Inverted Index for [{country}]...")
        blocker = FastInvertedIndexBlocker(max_bucket_size=1000)
        blocker.index_candidates(cand_df)
        
        # Convert candidate records to dictionary for O(1) lookups
        cand_dict = {row.entity_id: (row.business_name, row.business_address) for row in cand_df.itertuples(index=False)}
        del cand_df

        # Run candidate retrieval and batch scoring
        print(f"  Scoring [{country}] entities in batches...")
        batch_size = 50000
        n_records = len(s1_records)

        for b_start in range(0, n_records, batch_size):
            b_end = min(b_start + batch_size, n_records)
            chunk = s1_records[b_start:b_end]

            eval_pairs = []
            pair_meta = []
            entity_candidates_map = {}

            for s1_id, s1_name, s1_addr in chunk:
                cands = blocker.retrieve_candidates(s1_name, s1_addr, max_candidates_per_entity=35)
                cand_list = list(cands)
                entity_candidates_map[s1_id] = cand_list

                for cid in cand_list:
                    if cid in cand_dict:
                        c_name, c_addr = cand_dict[cid]
                        feats = compute_pair_features(s1_name, s1_addr, c_name, c_addr, cid)
                        eval_pairs.append(feats)
                        pair_meta.append((s1_id, cid))

            # Batch predict with LightGBM + Source-Aware Competitive Selection
            matched_map = {s1_id: [] for s1_id, _, _ in chunk}
            if eval_pairs:
                X_batch = pd.DataFrame(eval_pairs)
                probs = clf.predict_proba(X_batch)[:, 1]

                # Group candidate scores by s1_id and source (S2 / S3)
                scores_by_entity = {s1_id: {'S2': [], 'S3': []} for s1_id, _, _ in chunk}
                for (s1_id, cid), p in zip(pair_meta, probs):
                    src = 'S2' if cid.startswith('S2-') else 'S3'
                    scores_by_entity[s1_id][src].append((cid, float(p)))

                # Calibrated threshold, margin, and capacity
                conf_thresh = 0.85
                margin = 0.03
                max_per_source = 2

                for s1_id, src_dict in scores_by_entity.items():
                    selected = []
                    for src in ('S2', 'S3'):
                        cands = src_dict[src]
                        if not cands:
                            continue
                        # Sort descending by probability
                        cands.sort(key=lambda x: x[1], reverse=True)
                        top_p = cands[0][1]
                        if top_p < conf_thresh:
                            continue
                        
                        # Keep top candidates within margin of the winner
                        for cid, p in cands[:max_per_source]:
                            if p >= conf_thresh and p >= (top_p - margin):
                                selected.append(cid)
                    matched_map[s1_id] = selected

            # Stream chunk results directly to output files
            for s1_id, _, _ in chunk:
                c_str = ",".join(entity_candidates_map.get(s1_id, []))
                m_str = ",".join(matched_map.get(s1_id, []))
                f_cand.write(f"{s1_id}\t{c_str}\n")
                f_match.write(f"{s1_id}\t{m_str}\n")

            total_s1_processed += len(chunk)
            print(f"    Completed {b_end:,}/{n_records:,} [{country}] entities...")

        del blocker, cand_dict, s1_records
        print(f"  Finished [{country}] in {time.time() - t_country:.1f}s.")

    f_match.close()
    f_cand.close()
    print(f"\nPipeline successfully processed {total_s1_processed:,} entities in {time.time() - start_total:.1f}s.")
    print(f"Artifacts ready:\n  - {matching_file}\n  - {candidate_file}")

if __name__ == '__main__':
    run_inference()
