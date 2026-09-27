# Business Entity Resolution — Methodology & Approach

## 1. Methodology Used
Our entity resolution architecture uses a multi-stage funnel designed for the asymmetric macro $F_{0.5}$ metric, which penalizes false positive mergers twice as severely as false negatives:
* **Multi-key Blocking:** Generates high-recall candidate pairs from Source 2 and Source 3 for each Source 1 reference entity.
* **Deterministic Premise & Postal Veto:** Eliminates spurious candidate matches early by enforcing strict premise/street number and postal code compatibility.
* **Gradient-Boosted Decision Trees:** Scores surviving pairs using a calibrated LightGBM classifier trained on multi-attribute similarity features.
* **Precision-Biased Assignment:** Employs a conservative probability threshold ($0.85$), a cap of at most 3 matches per reference entity, and a delta margin ($\Delta = 0.03$) to safeguard against precision collapse.

---

## 2. Candidate Generation / Blocking Strategy
Given ~1.73M Source 1 records and ~9.97M combined records across Source 2 and Source 3, pairwise evaluation is computationally intractable ($O(N \times M)$). We implemented multi-key inverted index blocking:
* **Country Partitioning:** Blocking searches are strictly bounded within the matching country partition (India, US, and France).
* **Normalized Name Prefix & Token Keys:** Inverted indices index 3-character prefixes of legal-cleaned entity names, core significant tokens (removing common stopwords and legal suffixes such as *Ltd, Pvt, Corp, LLC, SA, SAS*), and phonetically standardized tokens.
* **Geographic / Postal Anchors:** Address tokens and postal codes are indexed to retrieve localized candidate subsets.
* **Reduction Ratio & Recall:** The candidate generation stage achieves a >99.9% reduction ratio while preserving a candidate recall ceiling exceeding 70% prior to veto filtering.

---

## 3. Model Architecture and Feature Engineering

### Model Architecture
* **LightGBM Classifier:** Configured with `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, and trained under binary cross-entropy on balanced positive and hard-negative pairs.
* Complies strictly with the competition constraint of under 8 Billion parameters and open-source licensing (MIT/Apache 2.0).

### Feature Engineering
For each candidate pair `(Source 1, Candidate)`, an 18-dimensional feature vector is computed across name and address attributes:
* **String Distances:** Levenshtein similarity, Damerau-Levenshtein distance, and Jaro-Winkler distance on cleaned names.
* **Token Similarities:** Jaccard token overlap, Dice coefficient, and character n-gram cosine similarities.
* **Address Match Signals:** Street number exact match indicator, street name Jaccard overlap, postal prefix agreement, and city/locality token overlap.
* **Country Match:** Exact match indicator across open string labels.

---

## 4. Other Relevant Information & Handling Edge Cases

* **Postal-Aware Premise Veto:** If both records contain an identifiable street or premise number and they conflict, the pair is vetoed prior to scoring. Distinct 5-digit/6-digit numeric sequences are parsed separately as postal codes to prevent false conflicts between street numbers and zip codes.
* **Singleton Preservation:** Records where all candidates fall below the calibrated threshold ($0.85$) are emitted as empty strings, correctly maximizing the macro $F_{0.5}$ score for isolated entities.
* **No External Lookups:** The pipeline strictly avoids external APIs, web scraping, external databases, or geocoders, adhering entirely to the academic integrity and fair-play regulations.
