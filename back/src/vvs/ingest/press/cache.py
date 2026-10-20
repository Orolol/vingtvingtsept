"""Cache disque des réponses d'API de presse, sous data/raw/press/<source>/.

Une réponse est écrite une fois, avec à côté un `.meta.json` (requête, heure UTC, taille,
sha256), puis n'est plus jamais réécrite : relire le cache redonne exactement les octets
reçus. La clé est l'empreinte des paramètres de la requête. Une fenêtre qui finit moins de
`VOLATILE_DAYS` jours avant aujourd'hui peut encore bouger (articles indexés en retard) : sa
clé inclut alors la date du jour, pour qu'une relance le lendemain interroge à nouveau la
source au lieu de figer une réponse incomplète.

Le journal `requests.jsonl` de chaque source garde une ligne par requête réseau (succès,
429, erreurs, attentes), en ajout seul.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from vvs import paths

VOLATILE_DAYS = 3


def press_dir(source: str) -> Path:
    return paths.RAW / "press" / source


def now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def params_key(params: dict) -> str:
    canon = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def is_volatile(end: date, today: date | None = None) -> bool:
    today = today or date.today()
    return end > today - timedelta(days=VOLATILE_DAYS)


@dataclass(frozen=True)
class Cached:
    path: Path
    content: bytes
    meta: dict


class DiskCache:
    """Cache immuable d'une source ; `root` vaut press_dir(source) par défaut."""

    def __init__(self, source: str, root: Path | None = None):
        self.source = source
        self.root = root or press_dir(source)

    def path_for(self, kind: str, year: int, name: str, ext: str = "json") -> Path:
        return self.root / kind / str(year) / f"{name}.{ext}"

    def get(self, path: Path) -> Cached | None:
        if not path.exists():
            return None
        content = path.read_bytes()
        meta = json.loads(meta_path(path).read_text(encoding="utf-8"))
        digest = hashlib.sha256(content).hexdigest()
        if digest != meta["sha256"]:
            raise ValueError(f"{path}: sha256 {digest} différent de la méta {meta['sha256']}")
        return Cached(path, content, meta)

    def put(self, path: Path, content: bytes, request: dict) -> Cached:
        if path.exists():
            raise FileExistsError(f"{path} existe déjà : le cache brut n'est jamais réécrit")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "request": request,
            "retrieved_at": now_utc().isoformat(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(content)
        tmp.rename(path)
        meta_path(path).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return Cached(path, content, meta)

    def log(self, **event) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        event = {"at": now_utc().isoformat(), **event}
        with (self.root / "requests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def iter_cached(self, kind: str) -> list[Path]:
        folder = self.root / kind
        if not folder.exists():
            return []
        return sorted(p for p in folder.rglob("*.json") if not p.name.endswith(".meta.json"))


def meta_path(path: Path) -> Path:
    return path.with_name(path.stem + ".meta.json")
