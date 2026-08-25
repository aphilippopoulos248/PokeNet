"""Canonical class names for the original 151.

Two image datasets merged together will not agree on spelling: `Nidoran-f`,
`NidoranF`, `Nidoran(female)`, `Mr. Mime` vs `MrMime`, `Farfetch'd` vs
`Farfetchd`. Left alone that silently splits one Pokemon into two classes, which
quietly caps your accuracy and poisons the confusion matrix.

This module folds every spelling onto one canonical name, using the official
list from data/metadata/Gen1_Pokemon.csv when it is present.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from src.utils import ROOT

METADATA_CSV = ROOT / "data" / "metadata" / "Gen1_Pokemon.csv"

# Spellings that do not survive naive normalisation.
ALIASES = {
    "nidoranf": "nidoranfemale",
    "nidoranfe": "nidoranfemale",
    "nidoranfemale": "nidoranfemale",
    "nidoranm": "nidoranmale",
    "nidoranmale": "nidoranmale",
    "mrmime": "mrmime",
    "mimejr": "mrmime",
    "farfetchd": "farfetchd",
    "farfetch": "farfetchd",
}


def normalize(name: str) -> str:
    """Lowercase, strip everything that is not a letter or digit."""
    key = re.sub(r"[^a-z0-9]", "", name.strip().lower())
    return ALIASES.get(key, key)


def load_canonical(csv_path: Path = METADATA_CSV) -> dict[str, str]:
    """normalized-key -> official display name. Empty dict if the CSV is absent."""
    if not Path(csv_path).exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    names = [r["Name"].strip() for r in rows if r.get("Name")]
    return {normalize(n): n for n in names}


def resolve(folder_names: list[str], canonical: dict[str, str] | None = None):
    """Map every folder name to a canonical class name.

    Returns (mapping, unmatched). `unmatched` are folders whose normalised key is
    not in the official 151 - usually junk directories ('images', 'train') or a
    spelling worth adding to ALIASES. Inspect them before training.
    """
    canonical = load_canonical() if canonical is None else canonical
    mapping, unmatched = {}, []
    for folder in folder_names:
        key = normalize(folder)
        if canonical:
            if key in canonical:
                mapping[folder] = canonical[key]
            else:
                unmatched.append(folder)
        else:
            # No official list available - fold spellings together anyway.
            mapping[folder] = key
    return mapping, unmatched
