"""Duels de second tour des législatives 2022 et 2024, par circonscription.

Étude du duel RN contre LFI (docs/results/2026-11-24-duel-rn-lfi.md). Pilotée par
data/registry/legislative_duels_sources.yaml :

- `ingest()` télécharge les résultats par circonscription du ministère de l'Intérieur
  (data.gouv.fr, un fichier par tour), le CSV de contrôle de l'accord NFP et les pages
  Wikipédia figées par révision (oldid) dans data/raw/legislative_duels/ ; un fichier présent
  n'est jamais réécrit, son empreinte est vérifiée ;
- `parse()` écrit data/curated/legislative_candidates.parquet (une ligne par candidat et par
  tour, avec le parti des candidats d'union de la gauche) et
  data/curated/legislative_duels.parquet (une ligne par second tour à deux candidats).

Parti des candidats d'union (nuance NUP en 2022, UG en 2024) : le ministère ne le donne pas.
2022 : tableau de l'article Wikipédia de la NUPES (par circonscription), contrôlé par les pages
départementales ; 2024 : pages départementales (par candidat, appariement par les voix de
premier tour), contrôlées par le CSV de l'accord NFP. Les désaccords entre source et contrôle
sont gardés dans la table (`party_check`), jamais corrigés à la main.

Électorats : chaque nuance est rangée dans un bloc du modèle (`NUANCE_BLOC`), les candidats
d'union selon leur parti (LFI et PCF en gauche radicale, comme dans blocs.yaml ; PS,
écologistes et autres en gauche ; parti inconnu : gauche radicale, bloc de la coalition dans
lists.yaml).
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from vvs import paths
from vvs.ingest.interieur import _download, sha256_of
from vvs.ingest.interieur.geo import OVERSEAS

SOURCES_FILE = paths.REGISTRY / "legislative_duels_sources.yaml"
WIKI_API = "https://fr.wikipedia.org/w/api.php"
USER_AGENT = "vingtvingtsept/0.0.1 (research)"
WIKI_BATCH = 10  # révisions par requête (contenu complet : réponses de quelques Mo)
QUALIFY_SHARE = 0.125  # part des inscrits pour se maintenir au second tour
# Écart toléré (voix) entre somme des voix et exprimés d'une circonscription : le fichier 2024
# en contient de quelques voix (somme nationale −2 au T1, +1 au T2, interieur_sources.yaml)
MAX_VOTE_GAP = 10

UNION_NUANCES = {"NUP", "UG"}
FAR_RIGHT_FINALIST = {"leg2022": {"RN"}, "leg2024": {"RN", "UXD"}}
# Nuance -> bloc du modèle (hors candidats d'union, rangés selon leur parti)
_COMMON = {
    "DXG": "far_left", "EXG": "far_left", "COM": "far_left", "FI": "far_left",
    "SOC": "left", "VEC": "left", "ECO": "left", "DVG": "left", "RDG": "left",
    "ENS": "centre", "HOR": "centre", "DVC": "centre",
    "LR": "right", "DVD": "right",
    "RN": "far_right", "UXD": "far_right", "REC": "far_right", "DSV": "far_right",
    "EXD": "far_right", "DXD": "far_right",
    "DIV": "other", "REG": "other",
}  # fmt: skip
# UDI : dans l'UDC avec LR en 2022, dans Ensemble en 2024 (lists.yaml)
NUANCE_BLOC = {"leg2022": {**_COMMON, "UDI": "right"}, "leg2024": {**_COMMON, "UDI": "centre"}}
PARTY_BLOC = {"lfi": "far_left", "pcf": "far_left", "ps": "left", "ecologists": "left",
              "generation_s": "left", "other_left": "left", "unknown": "far_left"}  # fmt: skip
BLOCS = ("far_left", "left", "centre", "right", "far_right", "other")
CENTRE_NUANCES = {"ENS", "HOR", "DVC"}
RIGHT_NUANCES = {"LR", "DVD"}


class SourceError(ValueError):
    pass


# --- sources


@dataclass(frozen=True)
class ResultFile:
    election: str
    name: str
    round: int
    format: str
    url: str
    sha256: str | None
    dataset: str = ""


def load_sources(path: Path | None = None) -> dict:
    return yaml.safe_load((path or SOURCES_FILE).read_text(encoding="utf-8"))


def result_files(src: dict) -> list[ResultFile]:
    out = []
    for key, e in src["results"].items():
        for f in e["files"]:
            out.append(ResultFile(key, f["name"], int(f["round"]), f["format"], f["url"],
                                  f.get("sha256"), f.get("dataset", "")))  # fmt: skip
    return out


def raw_dir() -> Path:
    return paths.RAW / "legislative_duels"


def _slug(title: str) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _check(path: Path, sha: str | None) -> str:
    digest = sha256_of(path)
    if sha and digest != sha:
        raise SourceError(f"{path}: sha256 {digest} ≠ {sha} attendu")
    return digest


def _fetch_file(url: str, dest: Path, sha: str | None, dataset: str) -> str:
    if dest.exists():
        _check(dest, sha)
        return "present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _download(url, dest)
    try:
        digest = _check(dest, sha)
    except SourceError:
        dest.unlink()
        raise
    meta = {"url": url, "dataset": dataset, "sha256": digest, "bytes": dest.stat().st_size,
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    dest.with_name(dest.name + ".source.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return "downloaded"


def _fetch_wikitext(title: str, revid: int, dest: Path, section: int | None = None) -> str:
    """Wikitexte d'une révision figée (oldid), une section si demandé."""
    if dest.exists():
        return "present"
    import httpx

    params = {"action": "parse", "oldid": revid, "prop": "wikitext|revid", "format": "json"}
    if section is not None:
        params["section"] = section
    for attempt in range(8):
        try:
            r = httpx.get(WIKI_API, params=params, timeout=60,
                          headers={"User-Agent": USER_AGENT})  # fmt: skip
            r.raise_for_status()
            d = r.json()["parse"]
            break
        except (httpx.HTTPError, KeyError) as exc:
            if attempt == 7:
                raise SourceError(f"échec Wikipédia {title} ({revid}) : {exc}") from exc
            wait = 10 * (attempt + 1)  # 429 : l'API demande de ralentir
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.headers.get("retry-after", "").isdigit():
                wait = max(wait, int(exc.response.headers["retry-after"]))
            time.sleep(wait)
    if int(d["revid"]) != int(revid):
        raise SourceError(f"{title} : révision {d['revid']} au lieu de {revid}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(d["wikitext"]["*"], encoding="utf-8")
    meta = {"title": title, "revid": int(revid), "section": section,
            "url": f"https://fr.wikipedia.org/w/index.php?oldid={revid}",
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    dest.with_name(dest.name + ".source.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    time.sleep(2.0)  # politesse envers l'API (429 en dessous)
    return "downloaded"


def _page_path(election: str, title: str) -> Path:
    return raw_dir() / "wikipedia" / election / f"{_slug(title)}.wikitext"


def ingest(src: dict | None = None) -> list[tuple[str, str]]:
    """Télécharge tout ce que liste le registre (en série). Renvoie [(fichier, statut)]."""
    src = src or load_sources()
    report = []
    for f in result_files(src):
        dest = raw_dir() / f.election / f.name
        report.append((f"{f.election}/{f.name}", _fetch_file(f.url, dest, f.sha256, f.dataset)))
    alloc = src["parties"]["leg2024"]["nfp_allocation"]
    dest = raw_dir() / "leg2024" / alloc["name"]
    report.append((f"leg2024/{alloc['name']}",
                   _fetch_file(alloc["url"], dest, alloc["sha256"], alloc["dataset"])))  # fmt: skip
    nupes = src["parties"]["leg2022"]["nupes_table"]
    dest = _page_path("leg2022", nupes["title"] + " section " + str(nupes["section"]))
    report.append((dest.name, _fetch_wikitext(nupes["title"], nupes["revid"], dest, nupes["section"])))
    for key in ("leg2022", "leg2024"):
        todo = [p for p in src["parties"][key]["department_pages"]
                if not _page_path(key, p["title"]).exists()]  # fmt: skip
        report += [(_page_path(key, p["title"]).name, "present")
                   for p in src["parties"][key]["department_pages"] if p not in todo]  # fmt: skip
        for i in range(0, len(todo), WIKI_BATCH):
            report += _fetch_revisions(key, todo[i : i + WIKI_BATCH])
    return report


def _fetch_revisions(election: str, pages: list[dict]) -> list[tuple[str, str]]:
    """Wikitexte de plusieurs révisions figées en une requête (prop=revisions&revids=…) :
    une requête `parse` par page déclenche des 429 de l'API."""
    import httpx

    params = {"action": "query", "prop": "revisions", "rvprop": "ids|content",
              "rvslots": "main", "format": "json", "formatversion": 2,
              "revids": "|".join(str(p["revid"]) for p in pages)}  # fmt: skip
    for attempt in range(8):
        try:
            r = httpx.get(WIKI_API, params=params, timeout=120,
                          headers={"User-Agent": USER_AGENT})  # fmt: skip
            r.raise_for_status()
            got = {rev["revid"]: rev["slots"]["main"]["content"]
                   for pg in r.json()["query"]["pages"] for rev in pg.get("revisions", [])}  # fmt: skip
            break
        except (httpx.HTTPError, KeyError) as exc:
            if attempt == 7:
                raise SourceError(f"échec Wikipédia ({election}) : {exc}") from exc
            time.sleep(10 * (attempt + 1))
    out = []
    for p in pages:
        if int(p["revid"]) not in got:
            raise SourceError(f"{p['title']} : révision {p['revid']} absente de la réponse")
        dest = _page_path(election, p["title"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(got[int(p["revid"])], encoding="utf-8")
        meta = {"title": p["title"], "revid": int(p["revid"]), "section": None,
                "url": f"https://fr.wikipedia.org/w/index.php?oldid={p['revid']}",
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
        dest.with_name(dest.name + ".source.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        out.append((dest.name, "downloaded"))
    time.sleep(2.0)
    return out


# --- résultats du ministère


def norm_dept(code: str) -> str:
    """Code département normalisé : '1' -> '01', ZA -> 971 ; ZX et ZZ restent tels quels."""
    s = code.strip().strip('"').upper()
    if s.isdigit() and len(s) == 1:
        s = "0" + s
    if s in OVERSEAS and s != "ZZ":
        return OVERSEAS[s]
    return s


def _int(s: str) -> int:
    s = s.strip().strip('"').replace(" ", "")
    return int(s) if s else 0


def parse_results_text(text: str, fmt: str, election: str, round_: int) -> pl.DataFrame:
    """Une ligne par candidat : résultats d'une circonscription répétés sur ses candidats."""
    rows = []
    reader = csv.reader(io.StringIO(text), delimiter=";")
    next(reader)
    for rec in reader:
        if not rec or not rec[0].strip():
            continue
        if fmt == "leg2022_cirlg":
            dept, label, circ = rec[0], rec[1], rec[2]
            registered, voters, blank, null, expressed = (
                _int(rec[5]), _int(rec[8]), _int(rec[10]), _int(rec[13]), _int(rec[16])
            )
            start, width = 19, 9
            fields = ("panel", "sex", "last_name", "first_name", "nuance", "votes")
        elif fmt == "leg2024_circ":
            dept, label, circ = rec[0], rec[1], rec[2]
            registered, voters, expressed, blank, null = (
                _int(rec[4]), _int(rec[5]), _int(rec[9]), _int(rec[12]), _int(rec[15])
            )
            start, width = 18, 9
            fields = ("panel", "nuance", "last_name", "first_name", "sex", "votes")
        else:
            raise SourceError(f"format inconnu {fmt}")
        d = norm_dept(dept)
        circ_no = int(circ.strip().strip('"')[-2:])
        total = 0
        n_cands = 0
        for k in range(start, len(rec) - width + 2, width):
            block = [x.strip().strip('"') for x in rec[k : k + width]]
            if len(block) < 6 or not block[0]:
                continue
            c = dict(zip(fields, block[:6]))
            c["votes"] = _int(c["votes"])
            total += c["votes"]
            n_cands += 1
            elected_raw = block[8] if len(block) > 8 else ""
            rows.append({
                "election": election, "round": round_, "dept": d, "dept_label": label.strip('"'),
                "circ": circ_no, "circ_code": f"{d}-{circ_no:02d}",
                "registered": registered, "voters": voters, "blank": blank, "null": null,
                "expressed": expressed, "panel": int(c["panel"]), "last_name": c["last_name"],
                "first_name": c["first_name"], "nuance": c["nuance"], "votes": c["votes"],
                "elected": elected_raw.lower().startswith("él") or elected_raw.lower() == "elu",
            })  # fmt: skip
        # écart interne à la source entre somme des voix et exprimés : gardé, borné
        gap = total - expressed
        if abs(gap) > MAX_VOTE_GAP:
            raise SourceError(f"{election} T{round_} {d}-{circ_no}: voix {total} ≠ exprimés {expressed}")
        for r in rows[len(rows) - n_cands :]:
            r["vote_sum_gap"] = gap
    return pl.DataFrame(rows)


def read_results(src: dict | None = None) -> pl.DataFrame:
    src = src or load_sources()
    parts = []
    for f in result_files(src):
        path = raw_dir() / f.election / f.name
        _check(path, f.sha256)
        enc = "latin-1" if f.format == "leg2022_cirlg" else "utf-8-sig"
        parts.append(parse_results_text(path.read_text(encoding=enc), f.format, f.election, f.round))
    return pl.concat(parts)


# --- partis des candidats d'union (Wikipédia)


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z]+", " ", s).strip()


_PARTY_PATTERNS = [
    ("lfi", r"\b(LFI|FI|France insoumise|insoumis)\b"),
    ("pcf", r"\b(PCF|Parti communiste|communiste)\b"),
    ("ps", r"\b(PS|Parti socialiste|socialiste)\b"),
    ("generation_s", r"(G\.s|Génération\.?s|Générations?)"),
    ("ecologists", r"\b(EELV|EÉLV|LÉ|LE|Les Écologistes|Écologistes?|Europe Écologie|Pôle écologiste|PE)\b"),
]


def party_code(text: str) -> str:
    """Parti d'un candidat d'union depuis un libellé Wikipédia (premier motif reconnu)."""
    t = text.strip()
    if not t:
        return "unknown"
    for code, pat in _PARTY_PATTERNS:
        if re.search(pat, t):
            return code
    return "other_left"


def _strip_wiki(cell: str) -> str:
    """Texte lisible d'une cellule : liens [[a|b]] -> b, modèles et balises retirés."""
    s = re.sub(r"<ref[^>]*/>", "", cell)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.DOTALL)
    s = re.sub(r"<small>\s*\(.*?\)\s*</small>", "", s, flags=re.DOTALL)  # coalition entre parenthèses
    s = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\{\{\s*(?:[Aa]brd|[Aa]bréviation discrète)\s*\|([^|}]*)\|[^}]*\}\}", r"\1", s)
    s = re.sub(r"\{\{\s*[Ff]ormatnum:([^}]*)\}\}", r"\1", s)
    s = re.sub(r"\{\{\s*(?:nowrap|Nowrap|nobr|Nobr)\s*\|(.*)\}\}", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("'''", "").replace("''", "")
    s = re.sub(r"^\s*(align|style|scope|colspan|rowspan|bgcolor)[^|]*\|", "", s)
    return s.strip()


def _cells(row: str) -> list[str]:
    """Cellules d'une ligne de tableau wiki (une cellule par ligne commençant par « | »)."""
    cells: list[str] = []
    for line in row.split("\n"):
        line = line.rstrip()
        if line.startswith("|") and not line.startswith("|-") and not line.startswith("|}") \
                and not line.startswith("|+"):  # fmt: skip
            body = line[1:]
            # cellules multiples sur une ligne : « || »
            cells.extend(body.split("||"))
        elif cells and line and not line.startswith("!") and not line.startswith("{|"):
            cells[-1] += "\n" + line
    return cells


# Nuances du ministère (2022 et 2024) reconnues dans la colonne « Nuance » des pages
KNOWN_NUANCES = {*_COMMON, "UDI", "NUP", "UG", "ENS"}


def _nuance_of(cell: str) -> str | None:
    """Nuance d'une cellule dont le texte lisible n'est qu'un code de nuance connu (écrit nu,
    en modèle d'abréviation ou en lien, selon les pages)."""
    t = _strip_wiki(cell).strip()
    return t if t in KNOWN_NUANCES else None


def parse_department_page(text: str) -> list[dict]:
    """Candidats des tableaux de résultats d'une page départementale : nuance, parti (texte),
    nom, voix du premier tour. Une ligne est lue si une de ses cellules, à partir de la
    troisième, ne contient qu'un code de nuance ; parti = cellule précédente, nom = celle
    d'avant ; voix = première cellule entière qui suit."""
    out = []
    for row in re.split(r"\n\|-", text):
        cells = _cells(row)
        k = next((i for i, c in enumerate(cells) if i >= 2 and _nuance_of(c)), None)
        if k is None:
            continue
        nuance = _nuance_of(cells[k])
        votes = None
        for c in cells[k + 1 :]:
            v = _strip_wiki(c).replace(" ", "").replace("\xa0", "").replace(" ", "")
            if re.fullmatch(r"\d+", v):
                votes = int(v)
                break
        if votes is None:
            continue
        out.append({"nuance": nuance, "party_text": _strip_wiki(cells[k - 1]),
                    "name": _strip_wiki(cells[k - 2]), "votes1": votes})  # fmt: skip
    return out


_ORDINAL = re.compile(r"\{\{\s*(\d+)(?:re|er|e|ère)\s*\}\}")
_COLOR = re.compile(r"couleurs\|([^}|]+)\}\}")
# Code couleur du tableau NUPES -> parti ; POI (3 candidats investis dans le quota LFI), DVG,
# PLR, MDM : `other_left` (non LFI au sens strict du code couleur)
NUPES_COLOR_PARTY = {"LFI": "lfi", "FI": "lfi", "PS": "ps", "EELV": "ecologists",
                     "PE": "ecologists", "GE": "ecologists", "ECO": "ecologists", "PCF": "pcf",
                     "G.s": "generation_s", "G·s": "generation_s", "GS": "generation_s"}  # fmt: skip


def parse_nupes_table(text: str) -> list[dict]:
    """Lignes du tableau NUPES 2022 : département (libellé), numéro, parti, candidat."""
    out = []
    for row in re.split(r"\n\|-", text):
        cells = _cells(row)
        if len(cells) < 4:
            continue
        m_ord = _ORDINAL.search(cells[0])
        m_col = _COLOR.search(cells[1])
        if not m_ord or not m_col:
            continue
        dept_label = _strip_wiki(cells[0].split("(")[0])
        code = m_col.group(1).strip()
        cand_cell = cells[3]
        out.append({
            "dept_label": dept_label, "circ": int(m_ord.group(1)), "color": code,
            "party": NUPES_COLOR_PARTY.get(code, "other_left" if code != "NUPES" else "unknown"),
            "party_text": _strip_wiki(cells[2]), "name": _strip_wiki(cand_cell),
        })  # fmt: skip
    return out


def _label_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]+", " ", s).strip()


def _surname_match(full: str, last: str) -> bool:
    """Nom du ministère contenu dans le nom Wikipédia, à l'orthographe près (tirets, espaces,
    une lettre d'écart : « Courrière Calmon » / « Courrière-Calmont »)."""
    import difflib

    a, b = norm_name(full), norm_name(last)
    if not b:
        return False
    if all(tok in a.split() for tok in b.split()):
        return True
    ac, bc = a.replace(" ", ""), b.replace(" ", "")
    if bc in ac:
        return True
    return any(difflib.SequenceMatcher(None, bc, ac[i : i + len(bc)]).ratio() >= 0.85
               for i in range(max(1, len(ac) - len(bc) + 1)))  # fmt: skip


def union_parties(results: pl.DataFrame, src: dict) -> pl.DataFrame:
    """Parti des candidats d'union de premier tour : election, circ_code, panel, party,
    party_source, party_check (contrôle : `agree`, `disagree`, `missing`), notes."""
    r1 = results.filter((pl.col("round") == 1) & pl.col("nuance").is_in(list(UNION_NUANCES)))
    rows = []
    for key in ("leg2022", "leg2024"):
        cand = r1.filter(pl.col("election") == key)
        dept_rows: dict[str, list[dict]] = {}
        for p in src["parties"][key]["department_pages"]:
            path = _page_path(key, p["title"])
            if path.exists():
                dept_rows[norm_dept(str(p["dept"]))] = parse_department_page(
                    path.read_text(encoding="utf-8")
                )
        # parti selon les pages départementales : appariement par nuance d'union et voix T1
        by_page: dict[tuple[str, int], tuple[str, str]] = {}
        for c in cand.iter_rows(named=True):
            hits = [x for x in dept_rows.get(c["dept"], [])
                    if x["nuance"] in UNION_NUANCES and x["votes1"] == c["votes"]]  # fmt: skip
            if len(hits) != 1:
                hits = [x for x in dept_rows.get(c["dept"], []) if x["votes1"] == c["votes"]]
            if len(hits) == 1:
                by_page[(c["circ_code"], c["panel"])] = (party_code(hits[0]["party_text"]),
                                                         hits[0]["party_text"])  # fmt: skip
        if key == "leg2022":
            nupes = src["parties"]["leg2022"]["nupes_table"]
            path = _page_path("leg2022", nupes["title"] + " section " + str(nupes["section"]))
            table = parse_nupes_table(path.read_text(encoding="utf-8"))
            labels = {_label_key(lab): d for d, lab in
                      cand.select("dept", "dept_label").unique().iter_rows()}  # fmt: skip
            by_circ = {}
            for t in table:
                d = labels.get(_label_key(t["dept_label"]))
                if d is not None:
                    by_circ[f"{d}-{t['circ']:02d}"] = t
            for c in cand.iter_rows(named=True):
                t = by_circ.get(c["circ_code"])
                page = by_page.get((c["circ_code"], c["panel"]))
                party, source, note = "unknown", "none", ""
                if t is not None:
                    if _surname_match(t["name"], c["last_name"]):
                        party, source = t["party"], "wikipedia_nupes_table"
                    else:
                        note = f"tableau NUPES : autre candidat ({t['name']})"
                if party == "unknown" and page is not None:
                    party, source = page[0], "wikipedia_department_page"
                check = "missing" if page is None or source != "wikipedia_nupes_table" else (
                    "agree" if page[0] == party else "disagree")
                rows.append({"election": key, "circ_code": c["circ_code"], "panel": c["panel"],
                             "party": party, "party_source": source, "party_check": check,
                             "party_note": note if check != "disagree" else
                             f"page départementale : {page[1]}"})  # fmt: skip
        else:
            alloc_path = raw_dir() / "leg2024" / src["parties"]["leg2024"]["nfp_allocation"]["name"]
            alloc = _read_allocation(alloc_path.read_text(encoding="utf-8"))
            for c in cand.iter_rows(named=True):
                page = by_page.get((c["circ_code"], c["panel"]))
                party, source = (page[0], "wikipedia_department_page") if page else ("unknown", "none")
                a = alloc.get(c["circ_code"])
                if a is None or party == "unknown":
                    check = "missing"
                else:
                    check = "agree" if a == party else "disagree"
                rows.append({"election": key, "circ_code": c["circ_code"], "panel": c["panel"],
                             "party": party, "party_source": source, "party_check": check,
                             "party_note": "" if check != "disagree" else f"accord NFP : {a}"})  # fmt: skip
    return pl.DataFrame(rows)


ALLOC_PARTY = {"FI": "lfi", "PS": "ps", "PE": "ecologists", "PCF": "pcf"}
# Corrections signalées par l'auteur du CSV (investitures finales FI au lieu du PS)
ALLOC_FIXES = {"64-01": "lfi", "64-03": "lfi", "66-01": "lfi"}


def _read_allocation(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    reader = csv.reader(io.StringIO(text))
    next(reader)
    for circ, label in reader:
        d, n = circ.split("-")
        code = f"{'ZZ' if d == '99' else norm_dept(d)}-{int(n):02d}"  # 99 : Français de l'étranger
        if code == "49-01" and code in out:
            code = "49-02"  # erreur signalée par l'auteur : deux 49-01, pas de 49-02
        out[code] = ALLOC_PARTY.get(label.strip(), "other_left")
    out.update(ALLOC_FIXES)
    return out


# --- tables propres


def candidate_bloc(election: str, nuance: str, party: str | None) -> str:
    if nuance in UNION_NUANCES:
        return PARTY_BLOC[party or "unknown"]
    return NUANCE_BLOC[election].get(nuance, "other")


def duel_type(election: str, fr_nuance: str | None, opp_nuance: str, opp_party: str | None) -> str:
    if fr_nuance is None:
        return "no_far_right"
    if opp_nuance in UNION_NUANCES:
        if opp_party == "lfi":
            return "rn_lfi"
        return "rn_left_unknown" if opp_party in (None, "unknown") else "rn_left_other"
    if opp_nuance in CENTRE_NUANCES or (election == "leg2024" and opp_nuance == "UDI"):
        return "rn_centre"
    if opp_nuance in RIGHT_NUANCES or (election == "leg2022" and opp_nuance == "UDI"):
        return "rn_right"
    return "rn_other"


def build_tables(results: pl.DataFrame, parties: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    cands = results.join(parties, on=["election", "circ_code", "panel"], how="left")
    # le parti d'un candidat d'union vaut pour ses deux tours (même panneau)
    p1 = cands.filter(pl.col("round") == 1).select(
        "election", "circ_code", "panel", pl.col("party").alias("_p"),
        pl.col("party_source").alias("_s"), pl.col("party_check").alias("_c"),
        pl.col("party_note").alias("_n"),
    )
    cands = (
        cands.drop("party", "party_source", "party_check", "party_note")
        .join(p1, on=["election", "circ_code", "panel"], how="left")
        .rename({"_p": "party", "_s": "party_source", "_c": "party_check", "_n": "party_note"})
    )
    cands = cands.with_columns(
        pl.struct("election", "nuance", "party")
        .map_elements(lambda r: candidate_bloc(r["election"], r["nuance"], r["party"]),
                      return_dtype=pl.Utf8)
        .alias("bloc")
    )  # fmt: skip
    duels = []
    for (election, circ_code), g2 in cands.filter(pl.col("round") == 2).group_by(
        ["election", "circ_code"], maintain_order=True
    ):
        if g2.height != 2:
            continue
        g1 = cands.filter((pl.col("election") == election) & (pl.col("circ_code") == circ_code)
                          & (pl.col("round") == 1))  # fmt: skip
        if g1.is_empty():
            continue
        fin = g2.to_dicts()
        fr = [c for c in fin if c["nuance"] in FAR_RIGHT_FINALIST[election]]
        if len(fr) == 1:
            a = fr[0]
            b = fin[1] if fin[0] is a else fin[0]
        else:
            a, b = (None, None)
        r1 = g1.to_dicts()
        reg1 = r1[0]["registered"]
        quals = [c for c in r1 if c["votes"] >= QUALIFY_SHARE * reg1]
        if len(quals) < 2:
            quals = sorted(r1, key=lambda c: -c["votes"])[:2]
        in2 = {c["panel"] for c in fin}
        withdrawn = [c for c in quals if c["panel"] not in in2]
        row = {
            "election": election, "year": int(election[-4:]), "circ_code": circ_code,
            "dept": fin[0]["dept"], "circ": fin[0]["circ"],
            "duel_type": duel_type(election, a and a["nuance"], (b or fin[1])["nuance"],
                                   (b or fin[1])["party"]),
            "registered1": reg1, "voters1": r1[0]["voters"], "expressed1": r1[0]["expressed"],
            "registered2": fin[0]["registered"], "voters2": fin[0]["voters"],
            "expressed2": fin[0]["expressed"], "n_candidates1": len(r1),
            "n_qualified1": len(quals), "withdrawn_nuances": ",".join(c["nuance"] for c in withdrawn),
            "triangular_withdrawn": bool(withdrawn),
        }  # fmt: skip
        if a is not None:
            others = [c for c in r1 if c["panel"] not in (a["panel"], b["panel"])]
            a1 = next(c for c in r1 if c["panel"] == a["panel"])
            b1 = next(c for c in r1 if c["panel"] == b["panel"])
            row |= {
                "fr_name": f"{a['first_name']} {a['last_name']}", "fr_nuance": a["nuance"],
                "opp_name": f"{b['first_name']} {b['last_name']}", "opp_nuance": b["nuance"],
                "opp_party": b["party"], "opp_party_check": b["party_check"],
                "opp_bloc": b["bloc"], "fr_votes1": a1["votes"], "opp_votes1": b1["votes"],
                **{f"votes1_{bl}": sum(c["votes"] for c in others if c["bloc"] == bl) for bl in BLOCS},
                "fr_votes2": a["votes"], "opp_votes2": b["votes"],
                "rn_share2": 100.0 * a["votes"] / (a["votes"] + b["votes"]),
                "fr_elected": a["elected"],
            }  # fmt: skip
        duels.append(row)
    return cands, pl.DataFrame(duels, infer_schema_length=None)


def parse(src: dict | None = None, curated: Path | None = None) -> tuple[Path, Path]:
    src = src or load_sources()
    curated = curated or paths.CURATED
    results = read_results(src)
    parties = union_parties(results, src)
    cands, duels = build_tables(results, parties)
    p1, p2 = curated / "legislative_candidates.parquet", curated / "legislative_duels.parquet"
    cands.write_parquet(p1)
    duels.write_parquet(p2)
    return p1, p2


def coverage(cands: pl.DataFrame) -> pl.DataFrame:
    """Couverture du parti des candidats d'union au premier tour, par élection."""
    return (
        cands.filter((pl.col("round") == 1) & pl.col("nuance").is_in(list(UNION_NUANCES)))
        .group_by("election", "party_source", "party_check", "party")
        .len()
        .sort("election", "party_source", "party_check", "party")
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "parse"
    if cmd == "ingest":
        for name, status in ingest():
            if status != "present":
                print(f"{status:10s} {name}")
        print("ingestion terminée")
    elif cmd == "parse":
        p1, p2 = parse()
        cands = pl.read_parquet(p1)
        duels = pl.read_parquet(p2)
        with pl.Config(tbl_rows=60):
            print(coverage(cands))
            print(duels.group_by("election", "duel_type").len().sort("election", "duel_type"))
        print(f"écrit : {p1}, {p2}")
    else:
        raise SystemExit("usage : python -m vvs.ingest.legislative_duels [ingest|parse]")


if __name__ == "__main__":
    main()
