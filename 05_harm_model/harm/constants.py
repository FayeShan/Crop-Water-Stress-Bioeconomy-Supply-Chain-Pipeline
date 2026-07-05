"""Centralised crop and mapping constants for the HARM pipeline."""

ALL_CROPS = [
    'aff', 'bar', 'bea', 'cas', 'ckp', 'coc', 'cof', 'cot', 'cwp',
    'mai', 'mil', 'nut', 'oac', 'pdc', 'pec',
    'plm', 'pot', 'rap', 'ri1', 'ri2', 'sgb', 'sgc',
    'sor', 'soy', 'sun', 'vgt', 'wh',
]

# Unweighted pipeline (G → G+R → Extreme on G+R)
UNWEIGHTED_CROPS = {'soy', 'sor', 'sgb', 'rap'}

# Weighted pipeline (GW → GW+RW → Extreme on GW+RW) — every crop except the unweighted set
WEIGHTED_CROPS = [c for c in ALL_CROPS if c not in UNWEIGHTED_CROPS]

# ── 30-region (Köppen) crops — use koppen data dirs ──
KOPPEN_CROPS = {'wh', 'mai'}

# ── Crops without SPAM harvest area mapping (monthly mask only) ──
CROPS_NO_SPAM = {'aff', 'oac', 'pec', 'pdc'}

# ── ACEA crop code → SPAM 2020 crop code mapping ──
ACEA_TO_SPAM = {
    'wh':  ['WHEA'], 'ri1': ['RICE'], 'ri2': ['RICE'], 'mai': ['MAIZ'],
    'bar': ['BARL'], 'mil': ['MILL', 'PMIL'], 'sor': ['SORG'],
    'pot': ['POTA'], 'cas': ['CASS'], 'bea': ['BEAN'], 'ckp': ['CHIC'],
    'cwp': ['COWP'], 'soy': ['SOYB'], 'nut': ['GROU'], 'plm': ['OILP'],
    'sun': ['SUNF'], 'rap': ['RAPE'], 'sgc': ['SUGC'], 'sgb': ['SUGB'],
    'cot': ['COTT'], 'cof': ['COFF', 'RCOF'], 'coc': ['COCO'],
    'vgt': ['TOMA', 'ONIO', 'VEGE'],
}

KOPPEN_LABELS = {
    1: "Af",  2: "Am",  3: "Aw",
    4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc",
    11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc",
    17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd",
    25: "Dfa", 26: "Dfb", 27: "Dfc", 28: "Dfd",
    29: "ET",  30: "EF",
}


def is_weighted(crop: str) -> bool:
    """Return True if crop uses the weighted pipeline."""
    return crop not in UNWEIGHTED_CROPS


def is_koppen(crop: str) -> bool:
    """Return True if crop uses the 30-region (Köppen) data."""
    return crop in KOPPEN_CROPS


def get_model_subdirs(crop: str):
    """Return (global_subdir, regional_subdir, extreme_subdir) for a crop."""
    if is_weighted(crop):
        return "global_weighted", "regional_weighted", "extreme_weighted"
    return "global", "regional", "extreme"
