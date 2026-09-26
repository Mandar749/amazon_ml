from rapidfuzz import fuzz, distance
from src.blocking.normalize import clean_text, extract_name_tokens, extract_numbers_and_anchors

def compute_pair_features(s1_name: str, s1_addr: str, cand_name: str, cand_addr: str, cand_id: str, hit_count: int = 1) -> dict:
    """Computes a lightweight, highly predictive feature vector for a candidate pair."""
    # 1. Cleaned text representation
    c_s1_name = clean_text(s1_name)
    c_s1_addr = clean_text(s1_addr)
    c_cand_name = clean_text(cand_name)
    c_cand_addr = clean_text(cand_addr)

    # 2. Token extractions
    s1_tokens = set(extract_name_tokens(s1_name))
    cand_tokens = set(extract_name_tokens(cand_name))
    s1_nums, _ = extract_numbers_and_anchors(s1_addr)
    cand_nums, _ = extract_numbers_and_anchors(cand_addr)

    # 3. Name similarities (RapidFuzz normalized 0.0 - 1.0)
    name_sort_ratio = fuzz.token_sort_ratio(c_s1_name, c_cand_name) / 100.0
    name_set_ratio = fuzz.token_set_ratio(c_s1_name, c_cand_name) / 100.0
    name_jw = distance.JaroWinkler.similarity(c_s1_name, c_cand_name)
    name_exact = 1.0 if c_s1_name and c_s1_name == c_cand_name else 0.0

    # Token overlap
    name_token_jaccard = 0.0
    if s1_tokens and cand_tokens:
        name_token_jaccard = len(s1_tokens.intersection(cand_tokens)) / len(s1_tokens.union(cand_tokens))

    # 4. Address similarities
    addr_set_ratio = fuzz.token_set_ratio(c_s1_addr, c_cand_addr) / 100.0 if (c_s1_addr and c_cand_addr) else 0.0
    addr_jw = distance.JaroWinkler.similarity(c_s1_addr, c_cand_addr) if (c_s1_addr and c_cand_addr) else 0.0

    # 5. Number matching logic (vital discriminator)
    s1_num_set = set(s1_nums)
    cand_num_set = set(cand_nums)
    num_both_present = 1.0 if (s1_num_set and cand_num_set) else 0.0
    num_exact_match = 1.0 if (s1_num_set and cand_num_set and bool(s1_num_set.intersection(cand_num_set))) else 0.0

    # 6. Metadata features
    source_is_s2 = 1.0 if cand_id.startswith('S2') else 0.0

    return {
        'name_token_sort_ratio': name_sort_ratio,
        'name_token_set_ratio': name_set_ratio,
        'name_jaro_winkler': name_jw,
        'name_exact_clean': name_exact,
        'name_token_jaccard': name_token_jaccard,
        'addr_token_set_ratio': addr_set_ratio,
        'addr_jaro_winkler': addr_jw,
        'num_both_present': num_both_present,
        'num_exact_match': num_exact_match,
        'source_is_s2': source_is_s2,
        'hit_count': float(hit_count)
    }
