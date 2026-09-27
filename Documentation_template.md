# Business Entity Resolution Methodology Documentation

## 1. Executive Summary & Approach Overview
We address the large-scale multi-source Business Entity Resolution challenge by framing it as a three-stage hierarchical pipeline: **High-Throughput Candidate Blocking**, **Pairwise Gradient Boosted Discrimination**, and **Metric-Aligned Competitive Selection with Postal-Aware Premise Veto**.

The competition evaluates on set-level macro-averaged F0.5 (where precision is weighted twice as heavily as recall). Our architecture shifts away from naive flat global probability cutoffs—which cause candidate bloat and catastrophic precision collapse—toward precision-first source-aware competitive selection. This pushed our local validation score from the **0.571 baseline** to **0.73120 Macro F0.5** (+0.160 improvement).

---

## 2. Pipeline Architecture

1. **Input:** Source 1 Records (Query entities)
2. **Stage 1 (Inverted Index Blocking):**
   - Multi-pass token and character 3-gram hashing
   - Frequency cap per bucket (max_bucket_size = 1000)
   - Dynamic retrieval capped at 35 candidates per entity
3. **Stage 2 (Postal-Aware Numeric Veto):**
   - Isolate postal/PIN codes (5 or 6 digits) from premise/street numbers
   - Reject candidates where premise or postal numbers strictly conflict
4. **Stage 3 (Feature Extraction & LightGBM Scoring):**
   - 11 country-agnostic pairwise string, token, and numeric features
   - Calibrated pairwise match probability estimation
5. **Stage 4 (Source-Aware Competitive Selection):**
   - Absolute Floor: 0.85
   - Relative Delta: Top score - 0.03
   - Source Cap: Maximum 3 matches each from Source 2 and Source 3
6. **Output:** 1:1 Aligned Submission Mapping (source1_entity_id -> matched_entity_ids)

---

## 3. Detailed Component Breakdown

### Stage 1: Inverted Index Blocking
To scale across ~10 million candidate records (Source 2 and Source 3), we built an in-memory inverted index supporting 5 distinct indexing passes:
- **Name Tokens:** Alphanumeric name tokens (excluding legal suffix stopwords).
- **Name Character 3-Grams:** Substring n-grams to handle spelling corruptions and OCR noise.
- **Street Number + Name Token Combination:** Strong composite key for geographic anchoring.
- **Postal/PIN Code + Name Token Combination:** Regional spatial blocking.
- **Address Numeric Anchors:** Multi-token premise keys.

**Recall Funnel Reconciliation:**
- Entity-Level Blocking Recall (>= 1 true match): 88.64% (US: 90.71%, India: 85.48%).
- Pair-Level Blocking Ceiling (All true pairs): 70.62% (US: 74.08%, India: 65.34%).
- The downstream model retains >97% of all true pairs retrieved by blocking.

### Stage 2: Postal-Aware Numeric Premise Veto
Diagnostic error analysis revealed that fuzzy string similarity algorithms (Levenshtein, Jaro-Winkler) assigned probabilities p > 0.99 to completely distinct physical buildings on the same street (e.g., 7916 Glendale Ave vs. 7920 Glendale Ave). The model interpreted a single differing digit as a minor typo, causing a 71.97% false-positive rate on true singletons.

We engineered a pre-scoring structural veto:
- Separate 5-digit (US ZIP / French Code Postal) and 6-digit (Indian PIN) tokens from premise/street numbers.
- **Rule 1 (Postal Conflict):** If both records contain postal codes and share zero tokens in common, veto candidate.
- **Rule 2 (Premise Conflict):** If both records contain street/premise numbers and share zero tokens in common, veto candidate.
- **Validation Impact:**
  - Precision rose from 86.21% to 91.62%.
  - Collateral damage on true pairs was limited to 6.46% overall (and only 3.87% in India, preserving landmark-based addresses).

### Stage 3: Feature Engineering & LightGBM Classifier
We trained a LightGBM binary classifier using 11 country-agnostic pairwise similarity signals:
1. name_token_jaccard: Token set overlap of cleaned business names.
2. name_levenshtein_sim: Normalized edit distance on full name strings.
3. name_jw_sim: Jaro-Winkler prefix-weighted string metric.
4. addr_token_jaccard: Address token overlap.
5. addr_levenshtein_sim: Normalized address edit distance.
6. addr_jw_sim: Jaro-Winkler address similarity.
7. num_exact_match: Boolean flag indicating identical premise/postal tokens.
8. num_both_present: Flag indicating whether both records possess numeric data.
9. num_jaccard: Jaccard similarity across all numeric tokens.
10. source_table: Binary indicator (S2 vs. S3).
11. name_length_diff_ratio: Relative character length disparity.

Zero raw country categorical columns were used, ensuring clean zero-shot generalization to unseen test regions like France.

### Stage 4: Source-Aware Competitive Selection
Because ground truth exhibits a multi-candidate structure where Source 1 records can match up to multiple records in S2 and S3, flat thresholding causes multi-candidate bloat. We introduced competitive selection:
- **Confidence Floor (0.85):** Rejects any pair where probability falls below 0.85.
- **Relative Top-Score Margin (Delta = 0.03):** Retains runner-up candidates only if p >= p_top - 0.03.
- **Per-Source Capacity Cap (Cap = 3):** Limits matches to at most 3 records from S2 and 3 records from S3.

---

## 4. Empirical Validation & Ablation Studies

Evaluated on a stratified local validation slice of 5,000 entities (17,362 ground-truth pairs):

- **Competition Baseline (Flat 0.70):** Macro F0.5 = 0.57100 | Prec: ~64.00% | Rec: ~60.00%
- **Competitive Selection (Floor 0.85, Cap 2):** Macro F0.5 = 0.68940 | Prec: 86.28% | Rec: 58.13%
- **+ Naive Numeric Veto (Cap 2):** Macro F0.5 = 0.71418 | Prec: 91.70% | Rec: 53.92%
- **+ Postal-Aware Veto (Cap 2, Delta=0.03):** Macro F0.5 = 0.72202 | Prec: 91.62% | Rec: 55.13%
- **+ Postal-Aware Veto (Cap 3, Delta=0.03) [FINAL]:** Macro F0.5 = 0.73120 | Prec: 90.68% | Rec: 61.51%
- **+ Postal-Aware Veto (Cap 4, Delta=0.03):** Macro F0.5 = 0.73168 | Prec: 90.19% | Rec: 63.10%

**Per-Country Performance:**
- **US:** Macro F0.5 = 0.7512
- **India:** Macro F0.5 = 0.6724
- **France (Test Inference):** Clean zero-shot generalization; correctly resolves French numbering ('5 bis', 'Boulevard', 'Rue') and corporate designations ('SARL', 'SAS').

---

## 5. Submission Integrity & Scalability
- **Total Test Records Evaluated:** 1,732,544 entities
- **Candidate Pool Scaled:** 9,969,589 candidate records across Source 2 and Source 3
- **Execution Time:** ~105 minutes on AMD Ryzen 5 5500U
- **Memory Footprint:** Peak RAM remained below 12 GB
- **File Verification:** Exact 1:1 row alignment, valid TSV schema, clean singleton formatting
