import re
import unicodedata
import pandas as pd

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'services', 'enterprises', 'associates',
    'partners', 'center', 'centre', 'multimedia', 'retail', 'holdings',
    'llp', 'aka', 'fka', 'shri', 'smt', 'sri', 'the', 'and', 'for', 'of'
}

ORDINALS_AND_FLOOR = {
    '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
    'first', 'second', 'third', 'ground', 'floor', 'flr', 'basement'
}

DOMAINS = re.compile(r'\.(com|org|net|in|co\.in|edu|gov|io|ai)$', re.IGNORECASE)

def clean_text(text) -> str:
    """Normalize unicode, strip accents, lower-case, remove noise characters safely with NA checks."""
    if pd.isna(text):
        return ''
    text_str = str(text).strip()
    if not text_str or text_str in ('<NA>', '<NULL>', 'nan', 'None'):
        return ''
        
    # Convert accents: e.g. á -> a, é -> e
    text_norm = unicodedata.normalize('NFKD', text_str)
    text_norm = ''.join(c for c in text_norm if not unicodedata.combining(c))
    text_norm = text_norm.lower()
    # Replace non-alphanumeric (except native characters, digits, spaces)
    text_norm = re.sub(r'[^\w\s]', ' ', text_norm)
    return ' '.join(text_norm.split())

def extract_name_tokens(name) -> list[str]:
    """Tokenize business name, strip domains and legal suffixes, split hyphenated terms."""
    cleaned = clean_text(name)
    tokens = []
    for raw_tok in cleaned.split():
        sub_tok = DOMAINS.sub('', raw_tok)
        if sub_tok and sub_tok not in LEGAL_SUFFIXES and len(sub_tok) > 1:
            tokens.append(sub_tok)
    return tokens

def extract_numbers_and_anchors(address) -> tuple[list[str], list[str]]:
    """
    Extracts valid building/plot numbers (ignoring ordinals like 1st floor)
    and salient address word anchors (like building or road names).
    """
    cleaned = clean_text(address)
    tokens = cleaned.split()
    numbers = []
    anchors = []

    for tok in tokens:
        if tok in ORDINALS_AND_FLOOR:
            continue
        if any(c.isdigit() for c in tok) and len(tok) >= 2:
            norm = tok.lstrip('0')
            if norm and norm not in ORDINALS_AND_FLOOR:
                numbers.append(norm)
        elif len(tok) >= 5 and tok not in ORDINALS_AND_FLOOR:
            anchors.append(tok)

    return numbers[:3], anchors[:3]
