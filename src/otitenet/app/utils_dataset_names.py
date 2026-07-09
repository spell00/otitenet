"""
Utility for mapping full dataset names to short names for display.
"""

DATASET_SHORT_NAME_MAP = {
    "Banque_Viscaino_Chili_2020": "Chili",
    "Banque_Comert_Turquie_2020_jpg": "Turquie",
    "Banque_Calaman_USA_2020_trie_CM": "USA",
    "GMFUNL_jan2023": "GMFUNL",
}

def get_short_dataset_name(name: str) -> str:
    """Return the short name for a dataset, or the original if not mapped."""
    if not name:
        return ""
    return DATASET_SHORT_NAME_MAP.get(name, name)

def get_short_dataset_names(names: str) -> str:
    """Map comma-separated dataset names to short names, joined by commas."""
    if not names:
        return ""
    return ",".join(get_short_dataset_name(n.strip()) for n in names.split(",") if n.strip())