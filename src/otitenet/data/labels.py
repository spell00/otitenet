"""Canonical label mapping for otoscopy datasets."""

from __future__ import annotations


FOUR_CLASS_LABEL_MAP = {
    # Normal
    "Normal": "Normal",
    "normal": "Normal",
    "NotNormal": "NotNormal",
    "notNormal": "NotNormal",
    # Abnormal finding / needs attention
    "Effusion": "NotNormal",
    "Aom": "NotNormal",
    "AOM": "NotNormal",
    "CSOM": "NotNormal",
    "Anormal": "NotNormal",
    "anormal": "NotNormal",
    "Ano": "NotNormal",
    "Chronic": "NotNormal",
    "Chornic": "NotNormal",
    "Chronic otitis media": "NotNormal",
    "OtitExterna": "NotNormal",
    "OtitisEksterna": "NotNormal",
    "Pseudomebrane": "NotNormal",
    "PseduoMembran": "NotNormal",
    "tympanoskleros": "NotNormal",
    "Myringosclerosis": "NotNormal",
    "myringosclerosis": "NotNormal",
    "foreign": "NotNormal",
    "Foreign": "NotNormal",
    # Wax
    "earwax": "Wax",
    "Earwax": "Wax",
    "Wax": "Wax",
    "earwax plug": "Wax",
    "Earwax plug": "Wax",
    # Tube
    "Tube": "Tube",
    "Earventulation": "Tube",
}


LABEL_SCHEMES = {
    "binary": {
        "label": "Binary: Normal / NotNormal",
        "labels": ("Normal", "NotNormal"),
    },
    "four_class": {
        "label": "Four class: Normal / NotNormal / Wax / Tube",
        "labels": ("Normal", "NotNormal", "Wax", "Tube"),
    },
}

DEFAULT_LABEL_SCHEME = "four_class"
CANONICAL_LABELS = LABEL_SCHEMES[DEFAULT_LABEL_SCHEME]["labels"]
TASK_LABEL_SCHEMES = {
    "notNormal": "binary",
    "otite_four_class": "four_class",
    "otitis_four_class": "four_class",
}
DEFAULT_LABEL_TASK = "notNormal"


def normalize_label(label: object, *, scheme: str = DEFAULT_LABEL_SCHEME, strict: bool = True) -> str:
    """Map a raw dataset label to the canonical training label."""
    if scheme not in LABEL_SCHEMES:
        raise ValueError(f"Unknown label scheme {scheme!r}. Expected one of: {', '.join(LABEL_SCHEMES)}")

    raw = str(label).strip()
    if raw in FOUR_CLASS_LABEL_MAP:
        mapped = FOUR_CLASS_LABEL_MAP[raw]
        if scheme == "binary" and mapped != "Normal":
            return "NotNormal"
        return mapped
    if strict:
        known = ", ".join(sorted(FOUR_CLASS_LABEL_MAP))
        raise ValueError(f"Unknown otoscopy label {raw!r}. Add it to LABEL_MAP. Known labels: {known}")
    return raw


def label_scheme_labels(scheme: str) -> tuple[str, ...]:
    if scheme not in LABEL_SCHEMES:
        raise ValueError(f"Unknown label scheme {scheme!r}. Expected one of: {', '.join(LABEL_SCHEMES)}")
    return LABEL_SCHEMES[scheme]["labels"]


def label_scheme_display_name(scheme: str) -> str:
    if scheme not in LABEL_SCHEMES:
        return str(scheme)
    return LABEL_SCHEMES[scheme]["label"]


def label_scheme_for_task(task: object) -> str:
    text = str(task or DEFAULT_LABEL_TASK)
    if text.startswith("SMOKE_TEST_"):
        return "binary"
    return TASK_LABEL_SCHEMES.get(text, "binary")


def task_display_name(task: object) -> str:
    text = str(task or DEFAULT_LABEL_TASK)
    if text == "notNormal":
        return "Binary: Normal / NotNormal"
    if text in {"otite_four_class", "otitis_four_class"}:
        return "Four class: Normal / NotNormal / Wax / Tube"
    return text


def labels_for_task(task: object) -> tuple[str, ...]:
    return label_scheme_labels(label_scheme_for_task(task))
