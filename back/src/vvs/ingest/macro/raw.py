"""Client HTTP et stockage immuable dans data/raw/macro/.

Chaque téléchargement daté est écrit dans data/raw/macro/<raw_dir>/<AAAA-MM-JJ>.<ext>,
accompagné de <AAAA-MM-JJ>.meta.json (URL, heure UTC, taille, sha256). Un fichier existant
n'est jamais réécrit : relancer l'ingestion le même jour ne fait rien pour cette série.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from vvs import paths

USER_AGENT = "vingtvingtsept/0.0.1 (+https://github.com/Orolol)"
TIMEOUT = httpx.Timeout(120.0, connect=30.0)
RETRIES = 3


def macro_dir() -> Path:
    return paths.RAW / "macro"


def client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True)


def get(http: httpx.Client, url: str, **kwargs) -> httpx.Response:
    """GET avec quelques reprises sur erreur réseau ou 5xx ; lève sur toute autre erreur."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = http.get(url, **kwargs)
            if r.status_code < 500:
                r.raise_for_status()
                return r
            last = httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
        except httpx.TransportError as e:
            last = e
        time.sleep(2 * (attempt + 1))
    assert last is not None
    raise last


@dataclass(frozen=True)
class RawFile:
    path: Path
    retrieved_at: datetime
    url: str


def now_utc() -> datetime:
    """Heure UTC sans fuseau (convention des colonnes *_at des tables curated)."""
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def write_raw(
    raw_dir: str, ext: str, content: bytes, url: str, day: date | None = None
) -> RawFile | None:
    """Écrit le fichier daté du jour ; renvoie None s'il existe déjà (rien n'est réécrit)."""
    day = day or date.today()
    folder = macro_dir() / raw_dir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{day.isoformat()}.{ext}"
    if path.exists():
        return None
    return _write_with_meta(path, content, url)


def write_once(path: Path, content: bytes, url: str) -> RawFile:
    """Écrit un document unique (PDF d'archive) ; l'appelant vérifie qu'il n'existe pas."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_with_meta(path, content, url)


def _write_with_meta(path: Path, content: bytes, url: str) -> RawFile:
    retrieved_at = now_utc()
    meta = {
        "url": url,
        "retrieved_at": retrieved_at.isoformat(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(content)
    tmp.rename(path)
    meta_path(path).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return RawFile(path=path, retrieved_at=retrieved_at, url=url)


def meta_path(path: Path) -> Path:
    return path.with_name(path.stem + ".meta.json")


def read_meta(path: Path) -> RawFile:
    meta = json.loads(meta_path(path).read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != meta["sha256"]:
        raise ValueError(f"{path}: sha256 {digest} différent de la méta {meta['sha256']}")
    return RawFile(
        path=path, retrieved_at=datetime.fromisoformat(meta["retrieved_at"]), url=meta["url"]
    )


def latest_raw(raw_dir: str, ext: str) -> RawFile:
    """Dernier fichier daté <AAAA-MM-JJ>.<ext> d'une série."""
    folder = macro_dir() / raw_dir
    files = sorted(p for p in folder.glob(f"????-??-??.{ext}")) if folder.exists() else []
    if not files:
        raise FileNotFoundError(f"aucun fichier brut {folder}/*.{ext} : lancer `vvs ingest macro`")
    return read_meta(files[-1])
