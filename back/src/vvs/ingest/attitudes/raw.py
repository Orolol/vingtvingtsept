"""Stockage immuable des documents dans data/raw/attitudes/<groupe>/<document>.<ext>.

Chaque document est écrit une fois, avec <document>.meta.json (URL, heure UTC, taille, sha256) ;
un fichier présent n'est jamais réécrit ni retéléchargé. La lecture vérifie l'empreinte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from vvs import paths
from vvs.ingest.attitudes.registry import Document
from vvs.ingest.macro import raw as macro_raw

# Certains hébergeurs (HubSpot de Verian, portail de la Commission européenne) refusent les
# agents qui ne ressemblent pas à un navigateur.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) vingtvingtsept/0.0.1 (+https://github.com/Orolol)"


@dataclass(frozen=True)
class RawDoc:
    path: Path
    sha256: str
    retrieved_at: datetime
    url: str


def root() -> Path:
    return paths.RAW / "attitudes"


def path_of(doc: Document) -> Path:
    return root() / doc.group / f"{doc.id}.{doc.ext}"


def client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=macro_raw.TIMEOUT, follow_redirects=True
    )


def download(http: httpx.Client, doc: Document) -> str:
    """Télécharge le document s'il manque ; renvoie « téléchargé » ou « présent »."""
    path = path_of(doc)
    if path.exists():
        read(doc)  # vérifie l'empreinte du brut existant
        return "présent"
    r = macro_raw.get(http, doc.url)
    content = r.content
    if doc.ext == "pdf" and not content.startswith(b"%PDF"):
        raise ValueError(f"{doc.id} : la réponse n'est pas un PDF ({r.headers.get('content-type')})")
    if doc.ext in ("zip", "xlsx") and not content.startswith(b"PK"):
        raise ValueError(f"{doc.id} : la réponse n'est pas une archive zip/xlsx")
    macro_raw.write_once(path, content, doc.url)
    return "téléchargé"


def read(doc: Document) -> RawDoc:
    path = path_of(doc)
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lancer `vvs ingest attitudes`")
    meta = macro_raw.read_meta(path)  # lève si le sha256 ne correspond plus
    digest = json.loads(macro_raw.meta_path(path).read_text(encoding="utf-8"))["sha256"]
    return RawDoc(path=path, sha256=digest, retrieved_at=meta.retrieved_at, url=meta.url)
