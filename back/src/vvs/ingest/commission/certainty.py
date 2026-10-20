"""Part d'électeurs « sûrs de leur choix » par candidat, lue dans le texte d'une notice.

Extraction indépendante du parseur des intentions de vote (`notice`), qu'elle ne modifie pas :
elle ne lit que les pages qui posent la question de la sûreté du choix et rend, pour chaque
candidat reconnu, la part de ses électeurs certains de leur choix (0 à 100).

Ce que publient les notices (relevé du 2026-10-20 sur 2017, 2022 et 2027) : la question est
posée par presque tous les instituts (« Votre choix est-il définitif ou peut-il encore
changer ? », « Diriez-vous que vous êtes sûr(e) de votre choix ou que vous pourriez changer
d'avis ? »), mais la ventilation par candidat n'est déposée que par certains ; les autres ne
déposent que le total (Ifop, Elabe, Kantar 2022) ou la seule non-réponse (Toluna Harris
Interactive). Formats de tableau reconnus, une ligne par candidat :

- **complément** : « Jean-Luc Mélenchon 76 24 » — sûrs puis « peut encore changer » (et
  parfois une non-réponse), somme 100 à 2 points près ; l'ordre des colonnes est lu dans la
  légende de la page (ligne qui nomme les deux modalités), sûrs en premier par défaut. Une
  évolution signée (« +6 ») en fin de ligne est ignorée. Ipsos (Cevipof, Le Parisien), Kantar.
- **part directe** : « Jean-Luc Mélenchon 66 +5 », une seule valeur par candidat, page titrée
  « part des électeurs … sûrs de leur choix » ; les valeurs doivent être ≥ 20 et ne pas sommer
  à 100 (sinon c'est un tableau d'intentions de vote). Kantar 2017.
- **ratio** : « Jean-Luc Mélenchon 12 18 », colonne « % sûrs de leur choix » en points des
  exprimés à côté de l'intention de vote (Kantar 2017) : part = 100 · 12 / 18. Accepté
  seulement si la seconde colonne somme à 100 (± 15) et domine la première sur toutes les
  lignes.
- **électorat potentiel** (BVA) : « Jean-Luc Mélenchon 9% De 7,1% à 10,9% +1,5 74% »,
  intention de vote, fourchette, évolution, puis la part des sûrs en dernière colonne, sous
  l'en-tête « % de l'électorat potentiel de chaque candidat sûrs de leur choix » ; « * »
  (effectif trop faible) : pas de valeur. La mise en page renvoie parfois les valeurs sur la
  ligne au-dessus du nom.

La valeur peut aussi être sur la ligne suivant le nom (mise en page en colonnes). Les pages
de second tour sont écartées. Une ligne « Ensemble » / « Total » donne la part de l'ensemble
(`TOTAL_ID`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from unidecode import unidecode

from vvs.ingest.commission.aliases import candidate_key

TOTAL_ID = "_all"

# Indices de la question, cherchés dans le texte ASCII minuscule sans espaces ni ponctuation
# (sauf les parenthèses de « sûr(e) ») : les notices collent parfois tous les mots.
_CUE = re.compile(
    r"sur\(?e?s?\)?s?de(votre|mon|leur|son)(choix|vote)"
    r"|sur(e|es|s)?\(?e?s?\)?deleurchoix"
    r"|surete(du|de)choix"
    r"|choix(devote)?estildefinitif"
    r"|choixdefinitif"
    r"|certitudeduchoix"
    r"|certainsdeleurchoix"
    r"|definitivementfixe"
)
_ROUND2 = re.compile(r"\b(2nd|second|2d|2e|2eme|deuxieme) tour\b")
_ROUND1 = re.compile(r"\b(1er|premier) tour\b")
# Modalités, sur le texte normalisé avec espaces
_SURE_WORD = re.compile(r"definitif|sur e de|sur de|surs de|sures de|sure de|sur e s de|certains de leur")
_UNSURE_WORD = re.compile(r"peut (il )?encore changer|pourr\w* (encore )?changer|peut changer|changer d avis")
_SHARE_HEADER = re.compile(r"(part|proportion|%) des electeurs|pour chaque candidat|electorat")
_IV_HEADER = re.compile(r"intentions? de vote")
_NUM = re.compile(r"^(\d{1,3}(?:[,.]\d+)?)%?$")
_SIGNED = re.compile(r"^[+\-−–=]\s?\d*(?:[,.]\d+)?%?$")
_TOTAL_LABEL = re.compile(r"^(ensemble|total|ensemble des \w+)$")
_POTENTIAL = re.compile(r"electorat potentiel de chaque")
# Ligne de la modalité « sûr » dans un tableau d'ensemble (Ifop, Elabe, Kantar 2022)
_SURE_LINE = re.compile(
    r"^(je suis sur|vous etes sur|vous etes tout a fait sur|sur e de|sur de son|surs de"
    r"|il est definitif|choix definitif|tout a fait sur)"
)


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9()]", "", unidecode(text).lower())


def _n(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%+]+", " ", unidecode(text).lower())).strip()


def is_certainty_page(page: str) -> bool:
    return bool(_CUE.search(_squash(page)))


def _is_second_round_page(page: str) -> bool:
    head = _n(" ".join(page.splitlines()[:8]))
    return bool(_ROUND2.search(head)) and not _ROUND1.search(head)


@dataclass
class CertaintyRow:
    candidate_id: str
    certain_share: float
    fmt: str  # complement, direct, ratio
    page: int
    values: tuple[float, ...] = ()


@dataclass
class CertaintyTable:
    page: int
    fmt: str
    rows: list[CertaintyRow] = field(default_factory=list)
    total: float | None = None


def _values(tokens: list[str]) -> list[float]:
    """Valeurs numériques en tête de `tokens`, évolutions signées ignorées, arrêt au premier
    mot."""
    out: list[float] = []
    for t in tokens:
        t = t.strip("*")
        if not t:
            continue
        if _SIGNED.match(t):
            continue
        m = _NUM.match(t)
        if m:
            out.append(float(m.group(1).replace(",", ".")))
            continue
        if t == "%":
            continue
        break
    return out


def _find_names(
    tokens: list[str], lookup: Callable[[str], str | None]
) -> list[tuple[str, int, int]]:
    """Candidats nommés dans une ligne : (id, début, fin) en indices de jetons."""
    out = []
    i = 0
    while i < len(tokens):
        for j in range(min(len(tokens), i + 4), i, -1):
            cid = lookup(" ".join(tokens[i:j]))
            if cid:
                out.append((cid, i, j))
                i = j
                break
        else:
            i += 1
    return out


def _raw_rows(
    page: str, lookup: Callable[[str], str | None]
) -> tuple[list[tuple[str, list[float]]], list[float]]:
    """(candidat, valeurs) dans l'ordre de la page, plus les valeurs de la ligne de total."""
    lines = page.splitlines()
    rows: list[tuple[str, list[float]]] = []
    total: list[float] = []
    for k, line in enumerate(lines):
        tokens = line.replace("%", " % ").split()
        names = _find_names(tokens, lookup)
        if not names:
            label = _n(re.sub(r"[\d,.%+\-−–]+", " ", line))
            if _TOTAL_LABEL.match(label) and not total:
                vals = _values([t for t in tokens if _n(t) not in ("ensemble", "total")])
                if not vals and k + 1 < len(lines):
                    vals = _values(lines[k + 1].replace("%", " % ").split())
                total = vals
            continue
        for idx, (cid, _, end) in enumerate(names):
            stop = names[idx + 1][1] if idx + 1 < len(names) else len(tokens)
            vals = _values(tokens[end:stop])
            if not vals and idx == len(names) - 1 and k + 1 < len(lines):
                nxt = lines[k + 1].replace("%", " % ").split()
                if nxt and not _find_names(nxt, lookup):
                    vals = _values(nxt)
                    if len(vals) != len([t for t in nxt if t != "%" and not _SIGNED.match(t)]):
                        vals = []  # ligne suivante qui n'est pas une ligne de valeurs
            if vals:
                rows.append((cid, vals))
    return rows, total


def _numbers(tokens: list[str]) -> list[float]:
    """Toutes les valeurs non signées de `tokens` (« De 7,1% à 10,9% » compris)."""
    out = []
    for t in tokens:
        m = _NUM.match(t.strip("*"))
        if m and not _SIGNED.match(t):
            out.append(float(m.group(1).replace(",", ".")))
    return out


def _drop_ranges(tokens: list[str]) -> list[str]:
    """Retire les fourchettes « De 7,1% à 10,9% » (marge d'erreur, BVA)."""
    out, i = [], 0
    while i < len(tokens):
        w = [unidecode(t).lower() for t in tokens[i : i + 4]]
        if len(w) == 4 and w[0] == "de" and w[2] == "a" and _NUM.match(w[1]) and _NUM.match(w[3]):
            i += 4
            continue
        out.append(tokens[i])
        i += 1
    return out


def _potential_rows(page: str, lookup: Callable[[str], str | None]) -> list[tuple[str, float]]:
    """Format « électorat potentiel » (BVA) : dernière valeur de la ligne du candidat, ou de
    la ligne de valeurs au-dessus quand la mise en page l'y a renvoyée ; « * » final : rien."""
    # « 57%* » (base faible) garde sa valeur ; « * » ou « ** » seul : effectif trop faible.
    lines = [re.sub(r"(?<=\d)%?\*+", "% ", ln).replace("%", " % ") for ln in page.splitlines()]
    toks = [ln.split() for ln in lines]
    names = [_find_names(t, lookup) for t in toks]
    out = []
    for k, tokens in enumerate(toks):
        if len(names[k]) != 1:
            continue
        cid, _, end = names[k][0]
        after = _drop_ranges([t for t in tokens[end:] if t != "%"])
        cand = after
        if len(_numbers(after)) < 2 and k > 0 and not names[k - 1]:
            prev = _drop_ranges([t for t in toks[k - 1] if t != "%"])
            if prev and all(_NUM.match(t) or _SIGNED.match(t) or set(t) == {"*"} for t in prev):
                cand = prev
        if not cand or set(cand[-1]) == {"*"}:
            continue
        nums = _numbers(cand)
        # sur la ligne du nom, l'intention de vote précède la part des sûrs
        if len(nums) >= (1 if cand is not after else 2) and 10 <= nums[-1] <= 100:
            out.append((cid, nums[-1]))
    return out


def _overall(page: str) -> float | None:
    """Part des sûrs dans l'ensemble (dernière colonne : redressé ou publié), hors question."""
    for line in page.splitlines():
        if "?" in line:
            continue
        t = _n(line)
        if _SURE_LINE.match(t):
            vals = _values(line.replace("%", " % ").split()[::-1])[::-1]
            if vals and 20 <= vals[-1] <= 100:
                return vals[-1]
    return None


def _sure_first(page: str) -> bool:
    """Ordre des colonnes : la légende qui nomme les deux modalités, hors question."""
    for line in page.splitlines():
        t = _n(line)
        if "?" in line:
            continue
        s, u = _SURE_WORD.search(t), _UNSURE_WORD.search(t)
        if s and u:
            return s.start() < u.start()
    return True


def parse_certainty_page(
    page: str, page_no: int, lookup: Callable[[str], str | None]
) -> CertaintyTable | None:
    """Tableau de sûreté du choix d'une page, ou None (voir les formats du module)."""
    if not is_certainty_page(page) or _is_second_round_page(page):
        return None
    norm = _n(page)
    if _POTENTIAL.search(norm):
        pot: dict[str, float] = {}
        for cid, v in _potential_rows(page, lookup):
            pot.setdefault(cid, v)
        if len(pot) < 2:
            return None
        rows = [CertaintyRow(cid, v, "potential", page_no, (v,)) for cid, v in pot.items()]
        return CertaintyTable(page_no, "potential", rows, None)
    raw, total_vals = _raw_rows(page, lookup)
    seen: dict[str, list[float]] = {}
    for cid, vals in raw:
        seen.setdefault(cid, vals)
    if len(seen) < 2:
        overall = _overall(page)
        return None if overall is None else CertaintyTable(page_no, "total", [], overall)
    items = list(seen.items())
    sure_first = _sure_first(page)

    def complement(vals: list[float]) -> float | None:
        for n in (2, 3):
            if len(vals) >= n and abs(sum(vals[:n]) - 100) <= 2:
                return vals[0] if sure_first else vals[1]
        return None

    comp = {cid: complement(v) for cid, v in items}
    if sum(x is not None for x in comp.values()) >= max(2, (len(items) + 1) // 2):
        rows = [
            CertaintyRow(cid, comp[cid], "complement", page_no, tuple(v))
            for cid, v in items
            if comp[cid] is not None
        ]
        total = complement(total_vals) if total_vals else None
        return CertaintyTable(page_no, "complement", rows, total)
    singles = [(cid, v) for cid, v in items if len(v) == 1]
    if (
        len(singles) == len(items)
        and _SHARE_HEADER.search(norm)
        and all(20 <= v[0] <= 100 for _, v in singles)
        and sum(v[0] for _, v in singles) > 110
    ):
        rows = [CertaintyRow(cid, v[0], "direct", page_no, tuple(v)) for cid, v in singles]
        total = total_vals[0] if len(total_vals) == 1 else None
        return CertaintyTable(page_no, "direct", rows, total)
    pairs = [(cid, v) for cid, v in items if len(v) == 2]
    if (
        len(pairs) >= 3
        and _IV_HEADER.search(norm)
        and all(0 < v[0] <= v[1] for _, v in pairs)
        and 85 <= sum(v[-1] for _, v in items if len(v) <= 2) <= 115
    ):
        rows = [
            CertaintyRow(cid, round(100 * v[0] / v[1], 1), "ratio", page_no, tuple(v))
            for cid, v in pairs
            if v[1] >= 1
        ]
        return CertaintyTable(page_no, "ratio", rows, None)
    overall = _overall(page)
    return None if overall is None else CertaintyTable(page_no, "total", [], overall)


def find_certainty_tables(
    pages: list[str], lookup: Callable[[str], str | None]
) -> list[CertaintyTable]:
    out = []
    for k, page in enumerate(pages, start=1):
        t = parse_certainty_page(page, k, lookup)
        if t is not None and (t.rows or t.total is not None):
            out.append(t)
    return out


def candidate_lookup(labels: dict[str, list[str]]) -> Callable[[str], str | None]:
    """Fonction libellé -> id à partir de {id: [libellés]}, insensible aux accents, à la casse
    et aux séparateurs (`aliases.candidate_key`). Accepte aussi l'ordre « NOM Prénom »."""
    index: dict[str, str] = {}
    for cid, names in labels.items():
        for name in names:
            index[candidate_key(name)] = cid
            parts = name.split()
            if len(parts) >= 2:
                index[candidate_key(" ".join(parts[1:] + parts[:1]))] = cid

    def lookup(label: str) -> str | None:
        key = candidate_key(label)
        return index.get(key) if len(key) >= 6 else None

    return lookup


def merge_tables(tables: list[CertaintyTable]) -> tuple[dict[str, CertaintyRow], float | None]:
    """Une valeur par candidat : la première rencontrée (le tableau principal précède les
    rappels et les détails dans les notices vues), format complément de préférence."""
    order = {"complement": 0, "potential": 1, "direct": 2, "ratio": 3, "total": 4}
    out: dict[str, CertaintyRow] = {}
    total = None
    for t in sorted(tables, key=lambda t: (order[t.fmt], t.page)):
        for r in t.rows:
            out.setdefault(r.candidate_id, r)
        if total is None and t.total is not None:
            total = t.total
    return out, total
