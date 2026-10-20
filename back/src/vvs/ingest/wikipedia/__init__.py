"""Ingestion des listes de sondages présidentiels de Wikipédia FR.

Usage :

    from vvs.ingest.wikipedia import parse
    polls, poll_results = parse(2022, html)

`html` est un HTML brut stocké par `common.store_raw` (avec son en-tête vvs). Ajouter une
élection : écrire `e<AAAA>.py` avec une sous-classe de `engine.PageAdapter`, l'enregistrer
dans `ADAPTERS`, compléter `data/registry/aliases_wikipedia.yaml`.

Scrutins de liste et législatives (européennes 2014-2024, législatives 2022-2024) : module
`scrutins` (élection en texte, entités = listes de `data/registry/lists.yaml`).
"""

from __future__ import annotations

import polars as pl

from vvs.ingest.wikipedia import e2002, e2007, e2012, e2017, e2022, e2027
from vvs.ingest.wikipedia.common import load_aliases, read_meta
from vvs.ingest.wikipedia.engine import PageAdapter, ParseReport

ADAPTERS: dict[int, PageAdapter] = {
    2002: e2002.Adapter2002(),
    2007: e2007.Adapter2007(),
    2012: e2012.Adapter2012(),
    2017: e2017.Adapter2017(),
    2022: e2022.Adapter2022(),
    2027: e2027.Adapter2027(),
}


def adapter_for(election: int) -> PageAdapter:
    try:
        return ADAPTERS[election]
    except KeyError:
        raise ValueError(
            f"pas d'adaptateur Wikipédia pour {election} (disponibles : {sorted(ADAPTERS)})"
        ) from None


def parse_with_report(election: int, html: str) -> tuple[pl.DataFrame, pl.DataFrame, ParseReport]:
    """Comme `parse`, avec le journal des tableaux et lignes ignorés."""
    adapter = adapter_for(election)
    return adapter.parse(html, read_meta(html, election), load_aliases(election))


def parse(election: int, html: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(polls, poll_results) d'une page, validés par `vvs.schemas`."""
    polls, results, _ = parse_with_report(election, html)
    return polls, results
