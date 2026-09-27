import sys
import os
import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.blocker import FastInvertedIndexBlocker
from src.features.build_features import compute_pair_features

def extract_street_and_postal_numbers(addr: str):
    if not isinstance(addr, str) or addr.lower() == "nan":
        return set(), set()
    tokens = re.findall(r'\b\d+\b', addr)
    if not tokens:
        return set(), set()
    postal = set()
    premise = set()
    for t in tokens:
        if len(t) in (5, 6):
            postal.add(t)
        else:
            premise.add(t)
    if not premise and len(tokens) >= 2:
        premise.add(tokens[0])
        postal.update(tokens[1:])
    return premise, postal

def numeric_veto(addr1: str, addr2: str) -> bool:
    p1, post1 = extract_street_and_postal_numbers(addr1)
    p2, post2 = extract_street_and_postal_numbers(addr2)
    # Postal code conflict
    if len(post1) > 0 and len(post2) > 0 and len(post1 & post2) == 0:
        return True
    # Street / premise number conflict
    if len(p1) > 0 and len(p2) > 0 and len(p1 & p2) == 0:
        return True
    return False

def parse_args():
    parser = argparse.ArgumentParser(description="Inference pipeline for Business Entity Resolution")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test datasets")
    parser.add_argument("--model-path", type=str, default="output/lgbm_entity_resolver.joblib", help="Path to trained LightGBM model")
    parser.add_argument("--output-path", type=str, default="output/matching_results.tsv", help="Path for output submission tsv")
    parser.add_argument("--floor", type=float, default=0.85, help="Confidence floor threshold")
    parser.add_argument("--delta", type=float, default=0.03, help="Relative score delta margin from top candidate")
    parser.add_argument("--cap-per-source", type=int, default=3, help="Maximum matches allowed per source table")
    parser.add_argument("--smoke-test", type=int, default=0, help="If > 0, run only on first N test entities")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 65)
    print("STARTING TEST INFERENCE PIPELINE")
    print(f"Config: Floor={args.floor}, Delta={args.delta}, Cap={args.cap_per_source}")
    if args.smoke_test > 0:
        print(f"SMOKE TEST MODE: Running on first {args.smoke_test} rows only.")
    print("=" * 65)

    test_dir = Path(args.test_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading test_source1...")
    s1_df = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    if args.smoke_test > 0:
        s1_df = s1_df.head(args.smoke_test).copy()
    print(f"Total S1 entities to resolve: {len(s1_df):,}")

    countries = list(s1_df["country"].unique())
    print(f"Countries present: {countries}")

    print("Loading candidate records from test_source2 and test_source3...")
    s2_chunks = [c for c in pd.read_csv(test_dir / "test_source2.tsv", sep="\t", chunksize=250000, dtype=str, keep_default_na=False) if not c[c["country"].isin(countries)].empty]
    s2_df = pd.concat([c[c["country"].isin(countries)] for c in s2_chunks], ignore_index=True)
    
    s3_chunks = [c for c in pd.read_csv(test_dir / "test_source3.tsv", sep="\t", chunksize=250000, dtype=str, keep_default_na=False) if not c[c["country"].isin(countries)].empty]
    s3_df = pd.concat([c[c["country"].isin(countries)] for c in s3_chunks], ignore_index=True)

    cand_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"Total candidate pool: {len(cand_df):,} records")
    cand_dict = {row.entity_id: (row.business_name, row.business_address) for row in cand_df.itertuples(index=False)}

    print("Building inverted index blocker...")
    blocker = FastInvertedIndexBlocker(max_bucket_size=1000)
    blocker.index_candidates(cand_df)

    print(f"Loading LightGBM classifier from {args.model_path}...")
    clf = joblib.load(args.model_path)

    results = []
    total = len(s1_df)

    print("Running candidate retrieval, feature extraction, and competitive scoring...")
    for idx, row in enumerate(s1_df.itertuples(index=False), start=1):
        if idx % 10000 == 0 or idx == total:
            print(f"  Processed {idx:,}/{total:,} entities ({idx / total * 100:.1f}%)...")

        s1_id = row.entity_id
        s1_name = row.business_name
        s1_addr = row.business_address

        cands = blocker.retrieve_candidates(s1_name, s1_addr, max_candidates_per_entity=35)
        if not cands:
            results.append((s1_id, ""))
            continue

        eval_pairs = []
        cids = []
        for cid in cands:
            if cid in cand_dict:
                c_name, c_addr = cand_dict[cid]
                # Postal-aware street & postal code veto
                if numeric_veto(s1_addr, c_addr):
                    continue
                eval_pairs.append(compute_pair_features(s1_name, s1_addr, c_name, c_addr, cid))
                cids.append(cid)

        if not eval_pairs:
            results.append((s1_id, ""))
            continue

        probs = clf.predict_proba(pd.DataFrame(eval_pairs))[:, 1]

        scores_by_src = {"S2": [], "S3": []}
        for cid, p in zip(cids, probs):
            src = "S2" if cid.startswith("S2-") else "S3"
            scores_by_src[src].append((cid, float(p)))

        selected = []
        for src in ("S2", "S3"):
            c_list = scores_by_src[src]
            if not c_list:
                continue
            c_list.sort(key=lambda x: x[1], reverse=True)
            top_p = c_list[0][1]
            if top_p < args.floor:
                continue
            for cid, p in c_list[:args.cap_per_source]:
                if p >= args.floor and p >= (top_p - args.delta):
                    selected.append(cid)

        results.append((s1_id, ",".join(selected)))

    sub_df = pd.DataFrame(results, columns=["source1_entity_id", "matched_entity_ids"])
    sub_df.to_csv(output_path, sep="\t", index=False)
    print(f"\nFinished! Submission written to: {output_path}")
    print(f"Summary: {len(sub_df):,} rows generated.")
    non_empty = (sub_df["matched_entity_ids"] != "").sum()
    print(f"Entities with >= 1 match: {non_empty:,} ({non_empty / len(sub_df) * 100:.2f}%)")
    print(f"Entities predicted singleton: {len(sub_df) - non_empty:,} ({(len(sub_df) - non_empty) / len(sub_df) * 100:.2f}%)")

if __name__ == "__main__":
    main()
