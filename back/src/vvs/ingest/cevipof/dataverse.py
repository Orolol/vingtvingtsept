"""Client minimal de l'API native Dataverse de data.sciencespo.fr (version 5.14).

Le jeton d'API est lu dans la variable d'environnement `DATAVERSE_API_TOKEN` et passé dans
l'en-tête `X-Dataverse-key` ; il n'apparaît jamais dans un message, un journal ni un fichier.
Sans jeton, seules les métadonnées et les fichiers publics sont accessibles.

Points d'API utilisés (https://guides.dataverse.org/en/5.14/api/) :
- `GET /api/search` (recherche de jeux) ;
- `GET /api/dataverses/{alias}/contents` (contenu d'une collection) ;
- `GET /api/datasets/:persistentId/?persistentId=doi:…` (métadonnées et fichiers) ;
- `GET /api/access/datafile/{id}?format=original` (fichier déposé, pas la version tabulée) ;
- `PUT /api/access/datafile/{id}/requestAccess` (demande d'accès à un fichier restreint).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

BASE_URL = "https://data.sciencespo.fr"
TOKEN_ENV = "DATAVERSE_API_TOKEN"
USER_AGENT = "vingtvingtsept/0.0.1"


class DataverseError(RuntimeError):
    pass


class AccessDenied(DataverseError):
    """Fichier restreint et accès non accordé à ce compte (HTTP 401 ou 403)."""


def token() -> str | None:
    t = os.environ.get(TOKEN_ENV, "").strip()
    return t or None


def _headers(auth: bool = True) -> dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    t = token() if auth else None
    if t:
        h["X-Dataverse-key"] = t
    return h


def _json(r: httpx.Response) -> dict:
    try:
        d = r.json()
    except ValueError as exc:
        raise DataverseError(
            f"{r.request.url.path} : réponse non JSON (HTTP {r.status_code})"
        ) from exc
    if d.get("status") != "OK":
        raise DataverseError(f"{r.request.url.path} : {d.get('message', d)} (HTTP {r.status_code})")
    return d["data"]


def get(path: str, params: dict | None = None, retries: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(BASE_URL + path, params=params, headers=_headers(), timeout=60)
            if r.status_code >= 500:
                raise httpx.HTTPStatusError("serveur", request=r.request, response=r)
            return _json(r)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise DataverseError(f"GET {path} : {last}")


def whoami() -> str | None:
    """Identifiant du compte associé au jeton, None sans jeton."""
    if token() is None:
        return None
    return get("/api/users/:me").get("identifier")


def dataset(pid: str) -> dict:
    """Métadonnées d'un jeu (dernière version visible), `pid` = « doi:10.21410/7E4/XXXXXX »."""
    return get("/api/datasets/:persistentId/", {"persistentId": pid})


def collection(alias: str) -> list[dict]:
    return get(f"/api/dataverses/{alias}/contents")


def search(q: str, per_page: int = 100) -> list[dict]:
    items, start = [], 0
    while True:
        d = get("/api/search", {"q": q, "type": "dataset", "per_page": per_page, "start": start})
        items += d["items"]
        start += per_page
        if start >= d["total_count"]:
            return items


@dataclass(frozen=True)
class Download:
    bytes: int
    content_type: str


def download(file_id: int, dest: Path, original: bool = True, retries: int = 4) -> Download:
    """Télécharge un fichier vers `dest` (via un `.part` renommé à la fin).

    `original` : format déposé (Stata, SPSS) plutôt que la version tabulée par Dataverse ; sans
    effet sur un fichier non tabulaire. Lève `AccessDenied` sur un fichier restreint.
    """
    url = f"{BASE_URL}/api/access/datafile/{file_id}"
    params = {"format": "original"} if original else None
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            # Dataverse peut répondre par une redirection vers le stockage.
            with httpx.stream(
                "GET", url, params=params, headers=_headers(), follow_redirects=True, timeout=600
            ) as r:
                if r.status_code in (401, 403) or (
                    r.status_code == 400 and "permission" in r.read().decode("utf-8", "replace")
                ):
                    raise AccessDenied(f"fichier {file_id} : accès refusé (HTTP {r.status_code})")
                r.raise_for_status()
                expected = r.headers.get("content-length")
                ctype = r.headers.get("content-type", "")
                with part.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            size = part.stat().st_size
            if expected is not None and size != int(expected):
                raise OSError(f"taille {size} ≠ content-length {expected}")
            part.rename(dest)
            return Download(size, ctype)
        except AccessDenied:
            part.unlink(missing_ok=True)
            raise
        except (httpx.HTTPError, OSError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
    raise DataverseError(f"échec du téléchargement du fichier {file_id} : {last}")


def can_download(file_id: int) -> bool:
    """Vrai si le compte peut lire le fichier (requête sur le premier octet)."""
    r = httpx.get(
        f"{BASE_URL}/api/access/datafile/{file_id}",
        headers={**_headers(), "Range": "bytes=0-0"},
        follow_redirects=True,
        timeout=60,
    )
    return r.status_code in (200, 206)


def request_access(file_id: int) -> str:
    """Demande l'accès à un fichier restreint au nom du compte du jeton.

    Renvoie le message du serveur. Une demande déjà en attente renvoie une erreur explicite
    du serveur, rapportée telle quelle (pas de nouvelle demande).
    """
    if token() is None:
        raise DataverseError(f"{TOKEN_ENV} absent : impossible de demander un accès")
    r = httpx.put(
        f"{BASE_URL}/api/access/datafile/{file_id}/requestAccess", headers=_headers(), timeout=60
    )
    try:
        d = r.json()
    except ValueError:
        return f"HTTP {r.status_code}"
    msg = d.get("message") or (d.get("data") or {}).get("message") or ""
    return f"{d.get('status')} : {msg}".strip()
