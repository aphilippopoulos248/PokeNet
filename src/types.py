"""Pokemon type lookup - a join, not a prediction.

A Pokemon's type is a deterministic property of its species: Charizard is always
Fire/Flying, in every image that has ever existed of it. So the model has no
business predicting it. Once the CNN says "Charizard", the type is a dictionary
lookup, and a lookup cannot contradict the species the way a second prediction
head could (a type classifier could happily output "Water" for a Charizard).

The general rule this file exists to illustrate: do not ask a model to predict
what you can derive. A learned output can be wrong; a lookup is correct by
construction.

The data comes from data/metadata/Gen1_Pokemon.csv - the stats table from the
first Kaggle dataset, which had no images and so is useless for training, but
carries the official 151 names and their types. names.py already uses it as the
canonical label list; this is its second job.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from src.names import METADATA_CSV, normalize

# Gen 1 only has 15 of these, but the full modern 18 are here so the map still
# works if this project is ever pointed at a later generation.
TYPE_COLORS = {
    "Normal": "#A8A77A", "Fire": "#EE8130", "Water": "#6390F0",
    "Electric": "#F7D02C", "Grass": "#7AC74C", "Ice": "#96D9D6",
    "Fighting": "#C22E28", "Poison": "#A33EA1", "Ground": "#E2BF65",
    "Flying": "#A98FF3", "Psychic": "#F95587", "Bug": "#A6B91A",
    "Rock": "#B6A136", "Ghost": "#735797", "Dragon": "#6F35FC",
    "Dark": "#705746", "Steel": "#B7B7CE", "Fairy": "#D685AD",
}


@lru_cache(maxsize=1)
def _type_table(csv_path: str = str(METADATA_CSV)) -> dict[str, list[str]]:
    """normalized-name -> ["Fire"] or ["Fire", "Flying"]. Empty if the CSV is gone.

    Keyed by names.normalize() rather than the raw name so the awkward spellings
    line up automatically: the CSV says "Nidoran(female)" and "Mr. Mime", the
    model's class names come from folded folder names, and normalize() collapses
    both onto the same key. Cached because it is read once per process but hit
    once per prediction.
    """
    path = Path(csv_path)
    if not path.exists():
        return {}
    table: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("Name") or "").strip()
            if not name:
                continue
            # "Type 2" is an empty string for single-type Pokemon, not a missing
            # column - filter on truthiness, not on key presence.
            types = [(row.get("Type 1") or "").strip(), (row.get("Type 2") or "").strip()]
            table[normalize(name)] = [t for t in types if t]
    return table


def types_for(name: str) -> list[str]:
    """Types for one class name. [] if the name isn't in the metadata CSV."""
    return list(_type_table().get(normalize(name), []))


def type_map(names: list[str]) -> dict[str, list[str]]:
    """Build a {class_name: [types]} map for a whole class list.

    Used at ONNX-export time to bake the lookup into meta.json, so the deployed
    serverless function is self-contained and never needs the CSV or src/.
    """
    return {n: types_for(n) for n in names}


def describe(name: str) -> str:
    """'Fire / Flying', or '' if unknown. For CLI output."""
    return " / ".join(types_for(name))
