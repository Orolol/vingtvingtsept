"""Séries INSEE par l'API SDMX ouverte de la BDM (sans clé).

https://bdm.insee.fr/series/sdmx/data/SERIES_BDM/<idbank> renvoie un message SDMX-ML 2.1
« StructureSpecificData » : un élément <Series> (attributs IDBANK, FREQ, TITLE_FR,
LAST_UPDATE…) et des <Obs TIME_PERIOD=… OBS_VALUE=…/>. On télécharge tout l'historique ; le
parseur ne garde que les périodes demandées.
"""

from __future__ import annotations

import re
from datetime import date

import httpx
from lxml import etree

from vvs.ingest.macro import raw
from vvs.ingest.macro.registry import Series

SDMX_URL = "https://bdm.insee.fr/series/sdmx/data/SERIES_BDM/{idbank}"
# FREQ SDMX de l'INSEE -> fréquence curated
FREQ_MAP = {"M": "M", "T": "Q", "A": "A"}
XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)


def fetch(http: httpx.Client, series: Series) -> bytes:
    url = SDMX_URL.format(idbank=series.source_id)
    content = raw.get(http, url).content
    # Valide avant d'écrire : un message d'erreur SDMX arrive aussi en HTTP 200 parfois.
    parse(content, series.source_id)
    return content


def ingest(http: httpx.Client, series: Series) -> raw.RawFile | None:
    content = fetch(http, series)
    return raw.write_raw(series.raw_dir, "xml", content, SDMX_URL.format(idbank=series.source_id))


def parse(content: bytes, idbank: str) -> tuple[str, list[tuple[date, float]]]:
    """Renvoie (fréquence curated, [(premier jour de la période, valeur)]) triés par date.

    Les observations sans valeur (OBS_VALUE absent ou « NaN ») sont ignorées.
    """
    root = etree.fromstring(content, XML_PARSER)
    if etree.QName(root).localname == "Error":
        texts = " ".join(t.strip() for t in root.itertext() if t.strip())
        raise ValueError(f"INSEE {idbank}: erreur SDMX : {texts}")
    series_nodes = list(root.iter("{*}Series"))
    if len(series_nodes) != 1:
        raise ValueError(f"INSEE {idbank}: {len(series_nodes)} séries dans la réponse, 1 attendue")
    node = series_nodes[0]
    if node.get("IDBANK") != idbank:
        raise ValueError(f"INSEE {idbank}: la réponse porte sur {node.get('IDBANK')}")
    sdmx_freq = node.get("FREQ")
    if sdmx_freq not in FREQ_MAP:
        raise ValueError(f"INSEE {idbank}: fréquence SDMX inconnue {sdmx_freq!r}")
    freq = FREQ_MAP[sdmx_freq]
    obs = []
    for o in node.iterchildren("{*}Obs"):
        value = o.get("OBS_VALUE")
        if value is None or value.strip().lower() in ("nan", ""):
            continue
        obs.append((period_start(o.get("TIME_PERIOD"), freq), float(value)))
    obs.sort()
    dates = [d for d, _ in obs]
    if len(dates) != len(set(dates)):
        raise ValueError(f"INSEE {idbank}: périodes en double")
    return freq, obs


def period_start(period: str, freq: str) -> date:
    """« 2026-09 » -> 2026-09-01 ; « 2026-Q2 » -> 2026-04-01 ; « 2026 » -> 2026-01-01."""
    if freq == "M":
        m = re.fullmatch(r"(\d{4})-(\d{2})", period)
        if m:
            return date(int(m[1]), int(m[2]), 1)
    elif freq == "Q":
        m = re.fullmatch(r"(\d{4})-Q([1-4])", period)
        if m:
            return date(int(m[1]), 3 * (int(m[2]) - 1) + 1, 1)
    elif freq == "A":
        m = re.fullmatch(r"(\d{4})", period)
        if m:
            return date(int(m[1]), 1, 1)
    raise ValueError(f"période INSEE {period!r} illisible pour la fréquence {freq}")
