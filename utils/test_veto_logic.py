import re

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
    # Different postal code -> VETO
    if len(post1) > 0 and len(post2) > 0 and len(post1 & post2) == 0:
        return True
    # Different street/premise number -> VETO
    if len(p1) > 0 and len(p2) > 0 and len(p1 & p2) == 0:
        return True
    return False

test_cases = [
    ("7916 Glendale Ave, 90210", "7920 Glendale Ave, 90210", "US", True),
    ("2204 Park St", "220 Park St", "US", True),
    ("42 MG Road, Bangalore 560001", "42 MG Road, Bangalore 560001", "India", False),
    ("15 Rue de la Paix, 75002 Paris", "17 Rue de la Paix, 75002 Paris", "France", True),
    ("15 Rue de la Paix, 75002 Paris", "15 Rue de la Paix, 75002 Paris", "France", False),
    ("Shop 4, Market Road", "Market Road", "India", False),
]

print("=" * 65)
print("RUNNING NUMERIC VETO UNIT TESTS")
print("=" * 65)
all_passed = True
for a1, a2, country, expected in test_cases:
    actual = numeric_veto(a1, a2)
    status = "PASS" if actual == expected else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"[{status}] Country: {country:<6} | Veto: {actual} (Expected: {expected})")
    print(f"       Addr1: {a1}")
    print(f"       Addr2: {a2}")
    print("-" * 65)

assert all_passed, "Unit tests failed!"
print("ALL UNIT TESTS PASSED!")
