"""Eurobaromètre standard : « les deux problèmes les plus importants » pour la France, par vague.

Source : les classeurs « Volume A » (tableaux pondérés par pays) de chaque vague standard,
publiés sans compte sur le portail open data de l'UE (data.europa.eu, téléchargements
webgate.ec.europa.eu). Un document du registre = un classeur (`member` : fichier lu dans le
zip). Les microdonnées GESIS exigent un compte : non utilisées.

Question (STD82 et suivantes) : « QA3a. What do you think are the two most important issues
facing (OUR COUNTRY) at the moment? (MAX. 2 ANSWERS) » ; numéros différents avant (Q.26,
QA28a, QA6a…). La question « personnelle » (« And personally… ») est écartée.

Deux mises en page :

- depuis 2005 (STD63) : une feuille par question, ligne d'en-tête des pays (« FR »), ligne
  « Total » (effectif pondéré), puis pour chaque item une ligne en français (effectif) et une
  ligne en anglais (part, en fraction arrondie au centième) ;
- 2003-2004 (STD60 à 62) : une seule feuille texte, colonne « F », item « 6.Unemployment »
  (effectif) suivi de la part des répondants en % à une décimale.

Valeur retenue : effectif pondéré de l'item / effectif pondéré total × 100, arrondi à 0,1
(méthode `workbook`) ; elle doit concorder avec la part imprimée (écart ≤ 0,75 point pour une
part arrondie au centième, 0,1 pour une part à une décimale : les effectifs pondérés publiés
sont eux-mêmes arrondis), sinon la lecture s'arrête. Deux
demi-échantillons (QA6a1 / QA6a2…) aux mêmes items sont additionnés ; la liste des feuilles à
lire peut être fixée par document (`sheets`). Les classeurs de 2003-2004 ne portent pas leurs
dates de terrain : elles viennent de la fiche JSON de l'enquête (`survey_metadata`, API du
portail Eurobaromètre).

Les classeurs .xls (jusqu'en 2020) sont convertis en .xlsx par LibreOffice (`soffice
--headless`), dans un cache temporaire indexé par l'empreinte du brut ; le brut n'est pas
modifié.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import date
from pathlib import Path

from unidecode import unidecode

from vvs.ingest.attitudes import Obs, raw
from vvs.ingest.attitudes.registry import AttitudeRegistry, Extraction
from vvs.ingest.macro import xlsx

QUESTION_RE = re.compile(r"two most important issues facing \(?our country\)?", re.IGNORECASE)
PERSONAL_RE = re.compile(r"personally|personnellement", re.IGNORECASE)

# item canonique -> motif sur le libellé anglais (ou français pour STD93), premier qui gagne
ITEMS: list[tuple[str, re.Pattern]] = [
    ("rising_prices", re.compile(r"rising prices|inflation|cost of living|hausse des prix")),
    ("economic_situation", re.compile(r"economic situation|situation economique")),
    ("unemployment", re.compile(r"^unemployment|^le chomage")),
    ("crime", re.compile(r"^crime|^l'insecurite")),
    ("immigration", re.compile(r"^immigration|^l'immigration")),
    ("terrorism", re.compile(r"^terrorism|^le terrorisme")),
    ("health", re.compile(r"health|sante")),
    ("pensions", re.compile(r"pension|retraite")),
    ("housing", re.compile(r"^housing|^le logement")),
    ("taxation", re.compile(r"^taxation|^les impots")),
    ("government_debt", re.compile(r"government debt|public debt|dette publique")),
    ("education", re.compile(r"educat|systeme educatif")),
    ("environment_climate", re.compile(r"environment|climat")),
    ("energy", re.compile(r"^energy|energetique")),
]


class EurobarometerError(ValueError):
    pass


# items distincts qui tomberaient sous un motif plus large : non retenus
EXCLUDED = re.compile(r"^(climate change|le changement climatique)")


def canonical(label: str) -> str | None:
    t = unidecode(label).lower().replace("\\", "/").strip()
    t = re.sub(r"^\(only to split [ab]\)\s*", "", re.sub(r"^\d+\.\s*", "", t))
    if EXCLUDED.search(t):
        return None
    for key, pat in ITEMS:
        if pat.search(t):
            return key
    return None


# ------------------------------------------------------------------------------- classeurs


def workbook_bytes(path: Path, member: str | None, sha: str) -> bytes:
    """Contenu xlsx du classeur (extrait du zip, converti depuis .xls si besoin)."""
    if member is None:
        content, name = path.read_bytes(), path.name
    else:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.split("/")[-1] == member]
            if len(names) != 1:
                raise EurobarometerError(f"{path.name} : membre {member!r} introuvable")
            content, name = z.read(names[0]), member
    if name.lower().endswith(".xlsx"):
        return content
    if not name.lower().endswith(".xls"):
        raise EurobarometerError(f"{name} : format non géré")
    cache = Path(tempfile.gettempdir()) / "vvs_xls_cache"
    cache.mkdir(exist_ok=True)
    out = cache / f"{sha[:16]}_{re.sub(r'[^A-Za-z0-9]+', '_', name)}x"
    if not out.exists():
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if soffice is None:
            raise EurobarometerError("LibreOffice (soffice) requis pour lire les classeurs .xls")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.xls"
            src.write_bytes(content)
            subprocess.run(
                [soffice, "--headless", "--convert-to", "xlsx", "--outdir", tmp, str(src)],
                capture_output=True, check=True, timeout=300,
            )
            shutil.move(Path(tmp) / "in.xlsx", out)
    return out.read_bytes()


def _text(row: list) -> str:
    return " ".join(str(c) for c in row if c not in (None, ""))


def _num(v) -> float | None:
    if isinstance(v, float):
        return v
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


FIELDWORK_RE = re.compile(r"Fieldwork\s*:?\s*([\d/\s-]+\d)", re.IGNORECASE)


def fieldwork(rows: list[list]) -> tuple[date | None, date | None, date | None]:
    """(début, fin, premier du mois de début) lus dans l'en-tête « Terrain/Fieldwork : … ».

    Formes rencontrées : « 9/10 - 5/11/2025 », « 12-27/05/2012 », « 08 - 17/11/2014 »,
    « 05-06/2005 » (mois seulement), « 10/2008 » (mois seulement)."""
    for r in rows[:6]:
        m = FIELDWORK_RE.search(_text(r))
        if not m:
            continue
        t = re.sub(r"\s+", "", m.group(1))
        if g := re.fullmatch(r"(\d{1,2})/(\d{1,2})-(\d{1,2})/(\d{1,2})/(\d{4})", t):
            d1, m1, d2, m2, y = map(int, g.groups())
            y1 = y - 1 if m1 > m2 else y
            return date(y1, m1, d1), date(y, m2, d2), date(y1, m1, 1)
        if g := re.fullmatch(r"(\d{1,2})-(\d{1,2})/(\d{1,2})/(\d{4})", t):
            d1, d2, mo, y = map(int, g.groups())
            return date(y, mo, d1), date(y, mo, d2), date(y, mo, 1)
        if g := re.fullmatch(r"(\d{1,2})-(\d{1,2})/(\d{4})", t):
            m1, _m2, y = map(int, g.groups())
            return None, None, date(y, m1, 1)
        if g := re.fullmatch(r"(\d{1,2})/(\d{4})", t):
            mo, y = map(int, g.groups())
            return None, None, date(y, mo, 1)
        raise EurobarometerError(f"date de terrain illisible : {m.group(0)!r}")
    return None, None, None


def _has_header(rows: list[list]) -> bool:
    for r in rows:
        cells = {str(c).strip() for c in r if c is not None}
        if "FR" in cells or ("F" in cells and "DK" in cells):
            return True
    return False


def _candidate_sheets(content: bytes) -> set[str] | None:
    """Feuilles dont le XML cite une chaîne partagée contenant la question (pré-filtre rapide :
    un classeur compte jusqu'à 200 feuilles) ; None si le classeur n'a pas de chaînes partagées."""
    from lxml import etree

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        if "xl/sharedStrings.xml" not in z.namelist():
            return None
        sst = etree.fromstring(z.read("xl/sharedStrings.xml"), xlsx.XML_PARSER)
        hits = []
        for i, si in enumerate(sst.iterfind("m:si", xlsx.NS)):
            t = "".join(x.text or "" for x in si.iter(f"{{{xlsx.NS['m']}}}t"))
            if QUESTION_RE.search(t):
                hits.append(f"<v>{i}</v>".encode())
        wb = etree.fromstring(z.read("xl/workbook.xml"), xlsx.XML_PARSER)
        rels = etree.fromstring(z.read("xl/_rels/workbook.xml.rels"), xlsx.XML_PARSER)
        target = {r.get("Id"): r.get("Target") for r in rels.iterfind("rel:Relationship", xlsx.NS)}
        out = set()
        for sh in wb.iterfind("m:sheets/m:sheet", xlsx.NS):
            t = target[sh.get(f"{{{xlsx.NS['r']}}}id")].lstrip("/")
            xml = z.read(t if t.startswith("xl/") else f"xl/{t}")
            if any(h in xml for h in hits):
                out.add(sh.get("name"))
        return out


def question_rows(content: bytes, only: list[str] | None = None) -> list[tuple[str, int]]:
    """(feuille, ligne) de chaque occurrence de la question « pays » (pas « personnellement »)."""
    out = []
    candidates = _candidate_sheets(content)
    for name in xlsx.sheet_names(content):
        if only and name not in only:
            continue
        if candidates is not None and name not in candidates:
            continue
        rows = xlsx.read_sheet(content, name)
        for i, r in enumerate(rows):
            t = _text(r)
            if QUESTION_RE.search(t) and not PERSONAL_RE.search(t) and _has_header(rows[i : i + 40]):
                out.append((name, i))  # pas les sommaires (« Index », « Content »)
                break
    return out


def parse_modern(rows: list[list]) -> tuple[float, dict[str, tuple[float, float, str]]]:
    """(total, {item: (effectif, part imprimée en %, libellé)}) depuis la ligne de la question
    (mise en page depuis 2004)."""
    hdr = next(
        (i for i, r in enumerate(rows[:30]) if "FR" in [str(c).strip() for c in r if c is not None]),
        None,
    )
    if hdr is None:
        raise EurobarometerError("ligne d'en-tête « FR » introuvable")
    col = [str(c).strip() if c is not None else "" for c in rows[hdr]].index("FR")
    total = None
    items: dict[str, tuple[float, float, str]] = {}
    i = hdr + 1
    while i < len(rows):
        r = rows[i]
        label = str(r[1]).strip() if len(r) > 1 and r[1] is not None else ""
        v = _num(r[col]) if len(r) > col else None
        if label.upper() == "TOTAL" and total is None:
            total = v
            i += 1
            continue
        if label and re.match(r"^(QA?\.?\s?\d|D\d)", label) and i > hdr + 3:
            break  # question suivante dans une feuille à plusieurs questions
        if label and v is not None and i + 1 < len(rows):
            nxt = rows[i + 1]
            en = str(nxt[1]).strip() if len(nxt) > 1 and nxt[1] is not None else ""
            if not en and "\n" in label:  # STD93 : « L'insécurité\nCrime » dans une cellule
                label, en = label.split("\n", 1)[0].strip(), label.split("\n", 1)[1].strip()
            share = _num(nxt[col]) if len(nxt) > col else None
            if nxt[col] is not None and str(nxt[col]).strip() == "-":
                share = 0.0
            if en.upper() in ("DK", "DON'T KNOW"):
                break  # après « NSP / DK » : regroupements ajoutés (STD92), hors liste posée
            if en and share is not None:
                pct = share * 100 if share <= 1.0 else share
                key = canonical(en) or canonical(label)
                if key is not None:
                    split = "SPLIT" in en.upper() and "SPLIT" in items.get(key, (0, 0, ""))[2].upper()
                    if key in items and split:
                        # item posé sous deux libellés à deux demi-échantillons disjoints : les
                        # parts imprimées portent chacune sur un demi-échantillon, leur moyenne
                        # approche la part de l'ensemble
                        c0, p0, l0 = items[key]
                        items[key] = (c0 + v, (p0 + pct) / 2, f"{l0} + {en}")
                        i += 2
                        continue
                    if key in items:  # deux libellés pour un même item : à trancher à la main
                        raise EurobarometerError(
                            f"item {key} : deux libellés ({items[key][2]!r} et {en or label!r})"
                        )
                    items[key] = (v, pct, en or label)
                i += 2
                continue
        i += 1
    if not total:
        raise EurobarometerError("ligne « Total » introuvable")
    return total, items


def parse_legacy(rows: list[list]) -> tuple[float, dict[str, tuple[float, float, str]]]:
    """Mise en page texte de 2003-2004 : (total, items) ; colonne « F », parts en %."""
    start = next(i for i, r in enumerate(rows) if QUESTION_RE.search(_text(r)))
    qm = re.search(r"Q\.?\s*(\d+)", _text(rows[start]))
    qnum = qm.group(1) if qm else ""
    hdr = next(
        i for i in range(start, start + 40)
        if "F" in [str(c).strip() for c in rows[i] if c is not None]
        and "DK" in [str(c).strip() for c in rows[i] if c is not None]
    )
    col = [str(c).strip() if c is not None else "" for c in rows[hdr]].index("F")
    total = None
    items: dict[str, tuple[float, float, str]] = {}
    i = hdr + 1
    while i < len(rows):
        first = str(rows[i][0]).strip() if rows[i] and rows[i][0] is not None else ""
        q = re.match(r"^Q\.?\s*(\d+)", first)
        if q and q.group(1) != qnum:
            break  # question suivante (la question en cours est répétée à chaque page)
        v = _num(rows[i][col]) if len(rows[i]) > col else None
        if first.upper().startswith("TOTAL") and total is None:
            total = v
        elif re.match(r"^\d+\.", first) and v is not None:
            nxt = rows[i + 1]
            cont = str(nxt[0]).strip() if nxt and nxt[0] is not None else ""
            label = first + (" " + cont if cont and not cont.startswith("_") else "")
            pct = _num(nxt[col])
            key = canonical(label)
            if key is not None and pct is not None:
                items[key] = (v, pct, label)
        i += 1
        if i - hdr > 400:
            break
    if not total:
        raise EurobarometerError("ligne TOTAL introuvable (mise en page 2003-2004)")
    return total, items


# ------------------------------------------------------------------------------ extraction


def extract(reg: AttitudeRegistry, e: Extraction) -> list[Obs]:
    sp = e.spec
    wanted = set(sp["series"])
    prefix = sp.get("series_prefix", "eb_mii_")
    overrides = sp.get("sheets") or {}
    out: list[Obs] = []
    for doc_id in sp["documents"]:
        doc = reg.documents[doc_id]
        rd = raw.read(doc)
        content = workbook_bytes(rd.path, doc.member, rd.sha256)
        found = question_rows(content, overrides.get(doc_id))
        if not found:
            raise EurobarometerError(f"{doc_id} : question introuvable")
        pooled: dict[str, list] = {}
        start = end = month = None
        sheets = [name for name, _ in found]
        for name, qrow in found:
            rows = xlsx.read_sheet(content, name)
            body = rows[qrow:]
            legacy = not any(
                "FR" in [str(c).strip() for c in r if c is not None] for r in body[:30]
            )
            if legacy:
                t, items = parse_legacy(rows)
                tol = 0.1
            else:
                t, items = parse_modern(body)
                tol = 0.75
            s, en, m = fieldwork(rows)
            start, end, month = start or s, end or en, month or m
            for k, (c, pct, label) in items.items():
                share = 100 * c / t
                if abs(share - pct) > tol + 1e-9:
                    raise EurobarometerError(
                        f"{doc_id} {name} {k} : {share:.2f} % recalculé contre {pct:g} % imprimé"
                    )
                acc = pooled.setdefault(k, [0.0, 0.0, label, []])
                acc[0] += c
                acc[1] += t
                acc[3].append(name)
        meta_id = (sp.get("survey_metadata") or {}).get(doc_id)
        if meta_id and start is None:  # 2003-2004 : dates de la fiche de l'enquête (API)
            meta = json.loads(raw.read(reg.documents[meta_id]).path.read_text(encoding="utf-8"))
            start = date.fromisoformat(meta["fieldworkStartDate"])
            end = date.fromisoformat(meta["fieldworkEndDate"])
        if start is None and month is None:
            raise EurobarometerError(f"{doc_id} : date de terrain introuvable")
        wave = (start or month).strftime("%Y-%m")
        note = sp.get("notes", {}).get(doc_id, "")
        for k, (c, t, label, names) in pooled.items():
            sid = prefix + k
            if sid not in wanted:
                continue
            parts = [note, f"libellé : {label}"]
            if len(names) > 1:
                parts.append(f"demi-échantillons additionnés ({', '.join(names)})")
            elif len(sheets) > 1:
                parts.append(f"posé au seul demi-échantillon {names[0]}")
            out.append(
                Obs(sid, wave, round(100 * c / t, 1), "workbook", doc_id, None,
                    "; ".join(x for x in parts if x), start=start, end=end or doc.fieldwork_end)
            )
    return out
