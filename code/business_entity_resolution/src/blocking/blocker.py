from collections import defaultdict
from src.blocking.normalize import clean_text, extract_name_tokens, extract_numbers_and_anchors

GENERIC_ADDR_WORDS = {
    'india', 'delhi', 'mumbai', 'bangalore', 'road', 'street', 'city', 'center',
    'national', 'international', 'general', 'global', 'near', 'opp', 'opposite',
    'main', 'cross', 'lane', 'avenue', 'drive', 'nagar', 'colony', 'sector',
    'bazaar', 'market', 'plaza', 'complex', 'tower', 'building', 'office', 'flat',
    'unit', 'suite', 'floor', 'block', 'dist', 'taluk', 'taluka', 'post'
}

def generate_blocking_keys(name: str, address: str) -> set[str]:
    keys = set()
    tokens = extract_name_tokens(name)
    numbers, anchors = extract_numbers_and_anchors(address)

    # 1. Address Numbers (Vital for aliased or translated names)
    for num in numbers:
        if len(num) >= 3:
            keys.add(f"num_{num}")

    # 2. Distinctive Address Landmark/Building Anchors
    for anc in anchors:
        if anc not in GENERIC_ADDR_WORDS and len(anc) >= 6:
            keys.add(f"anc_{anc}")

    # 3. Name Token Bigram (Order-invariant)
    if len(tokens) >= 2:
        sorted_pair = "_".join(sorted(tokens[:2]))
        keys.add(f"bgr_{sorted_pair}")
        # Acronym key (e.g. cmc for cardiology metro care)
        if len(tokens) >= 3:
            acronym = "".join(t[0] for t in tokens[:4])
            if len(acronym) >= 3:
                keys.add(f"acr_{acronym}")
    elif len(tokens) == 1 and len(tokens[0]) >= 3:
        keys.add(f"single_{tokens[0]}")

    # 4. Informative Unigrams (length >= 4 and non-generic)
    for tok in tokens:
        if len(tok) >= 4 and tok not in GENERIC_ADDR_WORDS:
            keys.add(f"tok_{tok}")

    return keys

class FastInvertedIndexBlocker:
    def __init__(self, max_bucket_size: int = 1000):
        self.index = defaultdict(list)
        self.max_bucket_size = max_bucket_size

    def index_candidates(self, candidate_df):
        """Indexes candidates into inverted key buckets."""
        for row in candidate_df.itertuples(index=False):
            cid = row.entity_id
            keys = generate_blocking_keys(str(row.business_name), str(row.business_address))
            for k in keys:
                self.index[k].append(cid)

    def retrieve_candidates(self, name: str, address: str, max_candidates_per_entity: int = 50) -> set[str]:
        keys = generate_blocking_keys(name, address)
        candidate_hit_counts = defaultdict(int)

        for k in keys:
            bucket = self.index.get(k)
            if bucket and len(bucket) <= self.max_bucket_size:
                for cid in bucket:
                    candidate_hit_counts[cid] += 1

        if not candidate_hit_counts:
            return set()

        if len(candidate_hit_counts) > max_candidates_per_entity:
            sorted_candidates = sorted(candidate_hit_counts.items(), key=lambda x: x[1], reverse=True)
            return {cid for cid, _ in sorted_candidates[:max_candidates_per_entity]}

        return set(candidate_hit_counts.keys())
