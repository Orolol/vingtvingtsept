"""Téléchargement des enquêtes du Cevipof vers data/raw/cevipof/<key>/.

- Documentation publique (`fetch: true` dans cevipof_datasets.yaml) : codebooks DDI,
  questionnaires, documentations, rapports techniques.
- Microdonnées (`role: data`, plus les rares fichiers publics marqués `fetch`) : téléchargées
  au format déposé (Stata, SPSS) dès que le compte du jeton y a accès ; sinon statut
  `restricted`, sans erreur.

Un fichier présent n'est jamais réécrit ; son empreinte sha256 est vérifiée contre le
registre (`sha256`) ou, à défaut, contre le `<nom>.source.json` écrit au premier
téléchargement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from vvs import paths
from vvs.ingest.cevipof import dataverse as dv
from vvs.ingest.cevipof.registry import DataFile, Dataset, Inventory, default_inventory

RAW_DIRNAME = "cevipof"
# Extension du fichier déposé selon son type MIME (les fichiers tabulés sont servis en .tab)
ORIGINAL_EXT = {
    "application/x-stata": ".dta",
    "application/x-stata-13": ".dta",
    "application/x-stata-14": ".dta",
    "application/x-spss-sav": ".sav",
    "application/x-spss-por": ".por",
    "text/csv": ".csv",
}


class ConsistencyError(ValueError):
    pass


def raw_dir(key: str | None = None) -> Path:
    d = paths.RAW / RAW_DIRNAME
    return d if key is None else d / key


def local_name(f: DataFile) -> str:
    """Nom local : celui de l'entrepôt sans espaces parasites, extension d'origine pour les
    fichiers tabulés (« fr_cdsp_ddi_enef2017_v16-1.tab » déposé en Stata -> « ….dta »)."""
    name = f.name.strip()
    ext = ORIGINAL_EXT.get(f.format)
    if ext and name.endswith(".tab"):
        name = name[: -len(".tab")] + ext
    elif ext and not Path(name).suffix:
        name += ext
    return name


def local_path(ds: Dataset, f: DataFile) -> Path:
    return raw_dir(ds.key) / local_name(f)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".source.json")


def _check(path: Path, f: DataFile) -> str:
    digest = sha256_of(path)
    expected = f.sha256
    if expected is None and _sidecar(path).exists():
        expected = json.loads(_sidecar(path).read_text(encoding="utf-8")).get("sha256")
    if expected and digest != expected:
        raise ConsistencyError(f"{path} : sha256 {digest} ≠ {expected} attendu")
    return digest


@dataclass
class IngestReport:
    rows: list[tuple[str, str, str, str]] = field(
        default_factory=list
    )  # key, fichier, statut, détail

    def add(self, key: str, name: str, status: str, detail: str = "") -> None:
        self.rows.append((key, name, status, detail))

    def count(self, status: str) -> int:
        return sum(1 for r in self.rows if r[2] == status)

    def summary(self, verbose: bool = False) -> str:
        statuses = sorted({r[2] for r in self.rows})
        lines = ["fichiers : " + ", ".join(f"{s} {self.count(s)}" for s in statuses)]
        for key, name, status, detail in self.rows:
            if verbose or status not in ("present", "downloaded"):
                lines.append(f"  {status:12s} {key} {name} {detail}".rstrip())
        return "\n".join(lines)


def _fetch(ds: Dataset, f: DataFile, report: IngestReport) -> None:
    dest = local_path(ds, f)
    if dest.exists():
        report.add(ds.key, dest.name, "present", _check(dest, f)[:12])
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        got = dv.download(f.id, dest, original=True)
    except dv.AccessDenied:
        report.add(ds.key, dest.name, "restricted")
        return
    try:
        digest = _check(dest, f)
    except ConsistencyError:
        dest.unlink()  # fichier tout juste téléchargé et invalide : on ne le garde pas
        raise
    if got.bytes != f.size:
        # la taille du registre est celle du fichier déposé ; un écart signale une nouvelle
        # version côté entrepôt (on garde le fichier, l'écart est rapporté)
        report.add(ds.key, dest.name, "size_mismatch", f"{got.bytes} octets ≠ {f.size} au registre")
    meta = {
        "dataset": ds.pid,
        "dataset_version": ds.version,
        "file_id": f.id,
        "file_name": f.name,
        "url": f"{dv.BASE_URL}/api/access/datafile/{f.id}?format=original",
        "sha256": digest,
        "bytes": got.bytes,
        "content_type": got.content_type,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _sidecar(dest).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report.add(ds.key, dest.name, "downloaded", digest[:12])


def ingest(
    inventory: Inventory | None = None,
    keys: list[str] | None = None,
    data: bool = True,
) -> IngestReport:
    """Télécharge la documentation publique et, si `data`, les microdonnées accessibles des
    jeux `core` et `useful` (ou des `keys` données)."""
    inv = inventory or default_inventory()
    report = IngestReport()
    for ds in inv.datasets:
        if keys is not None:
            if ds.key not in keys:
                continue
        elif ds.priority == "listed":
            continue
        for f in ds.files:
            if f.fetch or (data and f.role == "data"):
                _fetch(ds, f, report)
    return report


def request_access(inventory: Inventory | None = None, keys: list[str] | None = None):
    """Demande l'accès au fichier `data` des jeux `request: true` (ou des `keys` données) que
    le compte ne peut pas encore lire. Produit (key, file_id, réponse du serveur) au fil de
    l'eau (une requête par fichier, le serveur peut être lent)."""
    inv = inventory or default_inventory()
    for ds in inv.datasets:
        if (keys is not None and ds.key not in keys) or (keys is None and not ds.request):
            continue
        f = ds.data_file
        if f is None or not f.restricted:
            continue
        try:
            if dv.can_download(f.id):
                yield ds.key, f.id, "accès déjà accordé"
                continue
            yield ds.key, f.id, dv.request_access(f.id)
        except Exception as exc:  # noqa: BLE001 - rapporté, on continue avec les autres
            yield ds.key, f.id, f"échec : {type(exc).__name__} {exc}"


def access_status(inventory: Inventory | None = None) -> list[tuple[str, int, bool]]:
    """(key, file_id, lisible) pour le fichier `data` de chaque jeu `request: true`."""
    inv = inventory or default_inventory()
    return [
        (ds.key, ds.data_file.id, dv.can_download(ds.data_file.id))
        for ds in inv.datasets
        if ds.request and ds.data_file is not None
    ]


def check(inventory: Inventory | None = None) -> list[str]:
    """Écarts entre le registre et l'API : version, fichiers ajoutés ou retirés, accès."""
    inv = inventory or default_inventory()
    issues = []
    for ds in inv.datasets:
        try:
            meta = dv.dataset(ds.pid)
        except dv.DataverseError as exc:
            issues.append(f"{ds.key} : {exc}")
            continue
        v = meta["latestVersion"]
        version = f"{v['versionNumber']}.{v['versionMinorNumber']}"
        if version != ds.version:
            issues.append(f"{ds.key} : version {version} sur l'API, {ds.version} au registre")
        live = {f["dataFile"]["id"]: bool(f.get("restricted")) for f in v["files"]}
        mine = {f.id: f.restricted for f in ds.files}
        for fid in sorted(set(live) - set(mine)):
            issues.append(f"{ds.key} : fichier {fid} nouveau sur l'API")
        for fid in sorted(set(mine) - set(live)):
            issues.append(f"{ds.key} : fichier {fid} retiré de l'API")
        for fid in sorted(set(mine) & set(live)):
            if mine[fid] != live[fid]:
                issues.append(f"{ds.key} : fichier {fid} restricted={live[fid]} sur l'API")
    return issues
