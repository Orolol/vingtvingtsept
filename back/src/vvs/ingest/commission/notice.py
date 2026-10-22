"""Lecture d'une notice PDF : texte, métadonnées et tableaux d'intentions de vote.

Le parseur travaille sur le texte extrait par pdfplumber, ligne à ligne. Il n'y a pas de
gabarit commun aux instituts : on repère des « lignes de données » (libellé suivi de valeurs
numériques), on les regroupe en blocs contigus, puis on décide pour chaque bloc, d'après son
contexte (les lignes qui le précèdent) et son contenu, s'il s'agit d'une intention de vote
présidentielle 2027 et quelles colonnes donnent le brut et le redressé.

Règles de décision d'un bloc (dans l'ordre) :

1. il contient au moins deux candidats connus ; sinon ce n'est pas un tableau de candidats ;
2. il ne contient pas Emmanuel Macron et les 10 lignes au-dessus ne parlent ni de
   reconstitution, de reports, de transferts, d'évolutions, de primaire, de potentiel… :
   sinon c'est un rappel du vote 2022 ou une autre question (`_SKIP_CONTEXT`) ;
3. les 14 lignes au-dessus évoquent un vote (tour, hypothèse, intention, voteriez…) ;
4. tous les autres libellés du bloc sont des candidats connus ou des libellés non candidats
   connus (`aliases.NON_CANDIDATE_PREFIXES`) ; sinon `UnknownLabelError` ;
5. le nombre de valeurs par ligne correspond à un format connu (voir `_column_spec`), ou le
   tableau OpinionWay publie deux bases (`dual_columns`) ;
6. la somme des parts des candidats est comprise entre 94 et 106 (base exprimés) ;
7. deux candidats : second tour ; au moins `MIN_FIRST_ROUND_CANDIDATES` : premier tour ;
   entre les deux, bloc écarté.

Un bloc rejeté à une autre étape que la 4 est ignoré (et compté dans le rapport). Une même
hypothèse (tour, candidats, population) vue plusieurs fois dans une notice (page détaillée
puis récapitulatif, ou publié puis brut/redressé) n'est gardée qu'une fois, de préférence avec
son brut ; la même configuration publiée sur deux bases donne deux hypothèses.

Population d'un tableau (`notice_population`, `block_population`, règles communes avec les
archives) : la ligne « Base : … » du tableau, sauf quand la notice déclare que les résultats
publiés portent sur les certains d'aller voter (note Ifop « sur toutes les personnes inscrites
sur les listes électorales et certaines d'aller voter interrogées », pied des tableaux Toluna
Harris Interactive depuis septembre 2026 « Base : (Aux) inscrits sur les listes électorales
certains d'aller voter… ») : la ligne « inscrits » décrit alors l'échantillon, pas la colonne
publiée.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pdfplumber

from vvs import schemas
from vvs.ingest.commission.aliases import candidate_id, is_non_candidate, norm_label
from vvs.ingest.commission.listing import MONTHS_FR


class UnknownLabelError(ValueError):
    """Libellé de ligne inconnu dans un tableau d'intentions de vote."""


@dataclass
class NoticeText:
    pages: list[str]
    # Premières pages découpées en colonnes (tiers puis moitiés) : certaines méthodologies
    # sont mises en page sur trois colonnes et l'extraction pleine page mélange les lignes.
    columns: list[str]


def extract_text(path: Path, column_pages: int = 4) -> NoticeText:
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
        columns = []
        for page in pdf.pages[:column_pages]:
            w, h = page.width, page.height
            for k, n in ((0, 3), (1, 3), (2, 3), (0, 2), (1, 2)):
                columns.append(page.crop((w * k / n, 0, w * (k + 1) / n, h)).extract_text() or "")
    return NoticeText(pages, columns)


def _n(text: str) -> str:
    """Normalisation pour les recherches : ASCII minuscule, espaces simples."""
    return re.sub(r"\s+", " ", norm_label(text))


# --------------------------------------------------------------------------- métadonnées

_MONTH_RE = "(" + "|".join(MONTHS_FR) + ")"
_DAY_RE = r"(1 ?er|\d{1,2})"
# « du 14 au 16 octobre 2026 », « du 30 septembre au 1er octobre 2026 »,
# « les 18 et 19 août 2026 », « le 5 septembre 2026 » (texte normalisé par _n)
_RANGE_RE = re.compile(
    rf"\b(?:du|entre le|des|les)\s+{_DAY_RE}(?:\s+{_MONTH_RE})?(?:\s+(20\d\d))?"
    rf"\s+(?:au|et le|et)\s+{_DAY_RE}\s+{_MONTH_RE}\s+(20\d\d)\b"
)
_SINGLE_RE = re.compile(rf"\b(?:le)\s+{_DAY_RE}\s+{_MONTH_RE}\s+(20\d\d)\b")
# Plage dont le mois est rejeté plus loin par une mise en page en colonnes :
# « réalisée en ligne du 28 au 30 Échantillon de 1 725 personnes … avril 2026 ».
_LOOSE_RANGE_RE = re.compile(
    rf"\b(?:du|les)\s+{_DAY_RE}\s+(?:au|et)\s+{_DAY_RE}\b.{{0,120}}?\b{_MONTH_RE}\s+(20\d\d)\b"
)
_RECALL_CUE = re.compile(r"rappel|vague precedente|precedente vague|precedentes vagues")
_GLUED = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")
_FIELDWORK_CUE = re.compile(
    r"(terrain|realis|interrog|interview|dates? de|enquete|recueilli|collecte|administre)"
)


def _day(s: str) -> int:
    return 1 if s.replace(" ", "") == "1er" else int(s)


def _n_dates(text: str) -> str:
    """Normalisation pour les dates : parenthèses retirées, chiffres décollés des lettres
    (« du10au11juin2026 » dans certaines notices OpinionWay)."""
    return _GLUED.sub(" ", _n(re.sub(r"\([^()]{0,250}\)", " ", text)))


def find_fieldwork(
    text: str, require_cue: bool = False, window: tuple[date, date] | None = None
) -> tuple[date, date] | None:
    """Dates de terrain : première plage de dates précédée d'un indice de terrain.

    À défaut d'indice (et si `require_cue` est faux), la première plage du texte. Les
    parenthèses sont retirées (« du 7 (après l'annonce du verdict…) au 8 juillet 2026 »). Les
    dates sont validées : début <= fin, durée inférieure à 60 jours.
    """
    t = _n_dates(text)
    candidates = []
    for m in _LOOSE_RANGE_RE.finditer(t):
        d1, d2, m2, y2 = m.groups()
        if not _FIELDWORK_CUE.search(t[max(0, m.start() - 60) : m.start()]):
            continue
        try:
            end = date(int(y2), MONTHS_FR[m2], _day(d2))
            start = date(end.year, end.month, _day(d1))
        except ValueError:
            continue
        if start <= end:
            # rang 1 : après les plages strictes, mais devant les plages sans indice
            candidates.append((0.5, m.start(), start, end))
    for m in _RANGE_RE.finditer(t):
        d1, m1, y1, d2, m2, y2 = m.groups()
        try:
            end = date(int(y2), MONTHS_FR[m2], _day(d2))
            start_month = MONTHS_FR[m1] if m1 else end.month
            start_year = int(y1) if y1 else (end.year if start_month <= end.month else end.year - 1)
            start = date(start_year, start_month, _day(d1))
        except ValueError:
            continue
        if not (0 <= (end - start).days <= 60):
            continue
        cue = bool(_FIELDWORK_CUE.search(t[max(0, m.start() - 160) : m.start()]))
        candidates.append((0 if cue else 1, m.start(), start, end))
    for m in _SINGLE_RE.finditer(t):
        if _FIELDWORK_CUE.search(t[max(0, m.start() - 60) : m.start()]):
            try:
                d = date(int(m.group(3)), MONTHS_FR[m.group(2)], _day(m.group(1)))
            except ValueError:
                continue
            candidates.append((0.5, m.start(), d, d))
    # Dates d'une vague précédente citée pour les rappels : en dernier recours seulement.
    candidates = [
        (3, *c[1:]) if _RECALL_CUE.search(t[max(0, c[1] - 250) : c[1]]) else c for c in candidates
    ]
    if require_cue:
        candidates = [c for c in candidates if c[0] < 1]
    if window:
        candidates = [c for c in candidates if window[0] <= c[3] <= window[1]]
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2], candidates[0][3]


_SIZE_RE = r"(\d{1,3}(?: \d{3})+|\d{3,6})"
_WHO_RE = r"(?:personnes|individus|francais|inscrits|repondants|electeurs|residents)"
_SAMPLE_RE = re.compile(rf"echantillon[a-z ]{{0,40}}?de {_SIZE_RE} {_WHO_RE}")
_SAMPLE_RE2 = re.compile(rf"{_SIZE_RE} {_WHO_RE}")
_HYPOTHETICAL = re.compile(r"(cas d un|pour un|taille de l|exemple)( echantillon de)?\s*$")


def find_sample_size(text: str) -> int | None:
    """Échantillon total : le plus grand « échantillon de N <personnes> » cité, à défaut le
    plus grand « N personnes ».

    Les notices citent souvent le sous-échantillon d'inscrits puis l'échantillon d'origine
    (« 1 404 inscrits issus d'un échantillon de 1 506 personnes ») : on garde le maximum,
    borné à 100 000. Les exemples de lecture des marges (« pour un échantillon de 1 000
    personnes… ») sont écartés.
    """
    t = _GLUED.sub(" ", _n(text))

    def sizes_of(rx: re.Pattern) -> list[int]:
        # « dans le cas d'un échantillon de 2 000 personnes » : exemple de lecture des marges
        return [
            int(m.group(1).replace(" ", ""))
            for m in rx.finditer(t)
            if not _HYPOTHETICAL.search(t[max(0, m.start() - 25) : m.start()])
        ]

    sizes = sizes_of(_SAMPLE_RE) or sizes_of(_SAMPLE_RE2)
    sizes = [s for s in sizes if 100 <= s <= 100_000]
    return max(sizes) if sizes else None


def find_mode(text: str) -> str:
    t = _n(text)
    online = bool(re.search(r"en ligne|internet|online|cawi|auto administre", t))
    phone = bool(re.search(r"telephon|cati\b", t))
    if online and phone:
        return "mixed"
    if online:
        return "online"
    if phone:
        return "phone"
    return "unknown"


_SPONSOR_RES = (
    re.compile(r"(?:nom du/des |nom des? )?commanditaires?(?:\(s\))?\s*:\s*(.+)", re.IGNORECASE),
    re.compile(
        r"(?:realis[ée]e?|r[ée]alis[ée]e?|men[ée]e) (?:par [\w\- ]+? )?pour (.+)", re.IGNORECASE
    ),
)


def find_sponsor(pages: list[str]) -> str | None:
    head = "\n".join(pages[:4])
    for rx in _SPONSOR_RES:
        m = rx.search(head)
        if m:
            s = re.split(
                r"\s+(?:aupr[eè]s|du \d|selon|par |sur un |et (?:publi|diffus))|[.;]", m.group(1)
            )[0]
            s = s.strip(" :-–,")
            if 1 < len(s) <= 120:
                return s
    return None


_PUBLISH_RE = re.compile(rf"publication.{{0,80}}?\ble\s+{_DAY_RE}\s+{_MONTH_RE}(?:\s+(20\d\d))?")


def find_publish_date(text: str, fallback_year: int) -> date | None:
    m = _PUBLISH_RE.search(_n(text))
    if not m:
        return None
    try:
        return date(int(m.group(3) or fallback_year), MONTHS_FR[m.group(2)], _day(m.group(1)))
    except ValueError:
        return None


# Définition de l'échelle de participation dans les questionnaires (Elabe : « 0 signifie que
# vous êtes tout à fait certain de ne pas aller voter, 10 que vous êtes tout à fait certain
# d'aller voter ») : elle décrit une question, pas la population des intentions de vote.
_TURNOUT_SCALE = re.compile(
    r"\b\d+ (?:signifie )?que vous etes tout a fait certaine?s? (?:de ne pas aller|d aller) voter"
)
_CERTAIN = re.compile(
    r"certain(?:e|s|es)? d aller voter|votants decides|tout a fait certains"
    r"|ayant l intention d aller voter"
)


def find_population(text: str) -> str:
    """Population des intentions de vote d'après le texte de la notice.

    `certain_voters` : sous-échantillon filtré sur la participation déclarée (« certains
    d'aller voter », « votants décidés », Elabe : « seules les personnes inscrites […] et
    ayant l'intention d'aller voter sont prises en compte ») ; `registered` : inscrits sans
    filtre ; sinon `unknown`. La définition de l'échelle de participation 0-10 du
    questionnaire est ignorée : elle ne dit rien de la base des intentions.
    """
    t = _TURNOUT_SCALE.sub(" ", _n(text))
    if _CERTAIN.search(t):
        return "certain_voters"
    if re.search(r"inscrit", t):
        return "registered"
    return "unknown"


_IV_QUESTION = re.compile(
    r"(premier|1er|second|2nd|2d|2e) tour de (l|la prochaine) election presidentielle"
    r"( de 2027)? (avait lieu|se deroulait)"
    r"|si dimanche prochain se deroulait le (premier|second) tour"
    r"|intentions? de vote (au|pour le|a l election|a la presidentielle)"
    r"|le premier tour de l election presidentielle en france oppose"
)


_IV_CHOICE = re.compile(r"pour (lequel|laquelle|quel|qui)|quel est celui|voteriez|votiez|votez")
_IV_POTENTIAL = re.compile(r"pourriez vous voter|pourrait voter|potentiel")


def is_vote_intention(text: str) -> bool:
    """Vrai si la notice pose une question d'intention de vote présidentielle.

    La question doit demander un choix de candidat (« pour lequel… voteriez-vous ») ; le
    potentiel électoral (« pourriez-vous voter pour chacune… si le premier tour… ») n'en est
    pas une.
    """
    t = _n(text)
    for m in _IV_QUESTION.finditer(t):
        window = t[max(0, m.start() - 150) : m.end() + 150]
        if _IV_CHOICE.search(window) and not _IV_POTENTIAL.search(window):
            return True
    return False


# ------------------------------------------------------------------ tableaux de résultats

_VALUE_TOKEN = re.compile(r"^(?:<\s*1%?|<|\d{1,3}(?:[,.]\d+)?\s*%?\*?|-|–|—|\*|%)$")
_MARGIN = re.compile(r"\+/-\s*\d+(?:[,.]\d+)?\s*points?")
_LEADERS = re.compile(r"[.…·]{2,}")
_NOT_TESTED = re.compile(r"\bnon (?:test[ée]e?s?|pr[ée]sente?s?)\b", re.IGNORECASE)


@dataclass
class Row:
    label: str
    values: list[float | None]  # None : cellule vide (« - », « Non testé »)
    line_no: int


def parse_row(line: str, line_no: int) -> Row | None:
    """Découpe « libellé valeurs… » ; None si la ligne ne finit pas par une valeur.

    « <1% » vaut 0.5 par convention ; « - », « Non testé », « Non présent » valent None.
    """
    line = _MARGIN.sub(" ", line)
    line = _NOT_TESTED.sub(" - ", _LEADERS.sub(" ", line))
    tokens = line.split()
    values: list[str] = []
    while tokens and _VALUE_TOKEN.match(tokens[-1]):
        values.insert(0, tokens.pop())
    # « < 1 » écrit en deux jetons
    merged: list[str] = []
    for v in values:
        if merged and merged[-1] == "<":
            merged[-1] = "<" + v
        else:
            merged.append(v)
    values = [v for v in merged if v not in ("%", "*")]
    label = " ".join(tokens).strip(" •▪-–:")
    if not values or not label or not re.search(r"[A-Za-zÀ-ÿ]", label):
        return None
    parsed: list[float | None] = []
    for v in values:
        v = v.replace("%", "").replace("*", "").replace(",", ".").strip()
        if v.startswith("<"):
            parsed.append(0.5 if v[1:].strip() in ("1", "") else None)
        elif v in ("-", "–", "—"):
            parsed.append(None)
        else:
            parsed.append(float(v))
    return Row(label=label, values=parsed, line_no=line_no)


@dataclass
class Block:
    rows: list[Row]
    context: list[str]  # lignes précédant le bloc, depuis la fin du bloc précédent
    page: int

    @property
    def candidate_rows(self) -> list[tuple[str, Row]]:
        out = []
        for r in self.rows:
            cid = candidate_id(r.label)
            if cid:
                out.append((cid, r))
        return out


# Pieds de page qui finissent par un numéro et passeraient pour des lignes de données.
_FOOTER = re.compile(r"^\s*(?:p+o+u+r+\b|©|notice technique|retrouvez)", re.IGNORECASE)


def _only_values(line: str) -> bool:
    tokens = _MARGIN.sub(" ", line).split()
    return bool(tokens) and all(_VALUE_TOKEN.match(t) for t in tokens)


def _join_label(a: str, b: str) -> str:
    a, b = a.strip(" •▪"), b.strip()
    return f"{a}{b}" if a.endswith("-") else f"{a} {b}"


def _is_label_only(line: str) -> bool:
    return bool(line.strip()) and parse_row(line, 0) is None and not _only_values(line)


def merge_wrapped_values(lines: list[str]) -> list[str]:
    """Recolle les lignes de candidats coupées par la mise en page.

    Cas vus (mises en page à colonnes : Elabe fin 2025, Ifop, Ipsos, Toluna Harris Interactive) :
    - « Jean-Luc MÉLENCHON » / « 12,5 » : la valeur passe à la ligne suivante ;
    - « Raphaël » / « 10,0 » / « Glucksmann » : la valeur s'intercale dans le nom ;
    - « Nicolas Dupont- » / « Aignan 2 » : le nom est coupé, la valeur suit la fin du nom ;
    - « 4% +/- 0,4 point » / « Eric Zemmour » : la valeur précède le nom, en fin de tableau.
    On ne recolle que si le libellé recomposé est un candidat connu.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        nxt2 = lines[i + 2] if i + 2 < len(lines) else ""
        if _is_label_only(line):
            if _only_values(nxt):
                if candidate_id(line.strip(" •▪-–:")):
                    out.append(f"{line} {nxt}")
                    i += 2
                    continue
                if _is_label_only(nxt2) and candidate_id(_join_label(line, nxt2)):
                    out.append(f"{_join_label(line, nxt2)} {nxt}")
                    i += 3
                    continue
            row = parse_row(nxt, 0)
            if row and not candidate_id(row.label) and candidate_id(_join_label(line, row.label)):
                out.append(_join_label(line, nxt))
                i += 2
                continue
        prev = parse_row(out[-1], 0) if out else None
        if (
            _only_values(line)
            and prev is not None
            and candidate_id(prev.label)
            and _is_label_only(nxt)
            and candidate_id(nxt.strip(" •▪-–:"))
        ):
            # « 4% +/- 0,4 point » / « Eric Zemmour » : valeur avant le nom (Ipsos), seulement
            # au milieu d'un tableau déjà commencé.
            out.append(f"{nxt} {line}")
            i += 2
            continue
        out.append(line)
        i += 1
    return out


def _split_side_by_side(line: str) -> tuple[str, str] | None:
    """« A - Parti 42% 48% B - Parti 60% 55% » -> deux lignes, si A et B sont candidats."""
    tokens = line.split()
    for k in range(1, len(tokens) - 1):
        if not _VALUE_TOKEN.match(tokens[k]) or _VALUE_TOKEN.match(tokens[k + 1]):
            continue
        left, right = " ".join(tokens[: k + 1]), " ".join(tokens[k + 1 :])
        lrow, rrow = parse_row(left, 0), parse_row(right, 0)
        if lrow and rrow and candidate_id(lrow.label) and candidate_id(rrow.label):
            return left, right
    return None


def split_two_column_page(page: str) -> tuple[str, str] | None:
    """Page à deux tableaux côte à côte (seconds tours YouGov) -> pages gauche et droite.

    Activé si au moins deux lignes se coupent en deux lignes de candidats. Les autres lignes
    sont coupées en deux moitiés identiques (« Bruts Pondérés Bruts Pondérés ») ou recopiées
    des deux côtés comme contexte.
    """
    lines = page.splitlines()
    splits = [_split_side_by_side(line) for line in lines]
    if sum(s is not None for s in splits) < 2:
        return None
    left, right = [], []
    for line, sp in zip(lines, splits, strict=True):
        if sp is None:
            words = line.split()
            half = len(words) // 2
            if len(words) % 2 == 0 and half and words[:half] == words[half:]:
                sp = (" ".join(words[:half]),) * 2
            else:
                sp = (line, line)
        left.append(sp[0])
        right.append(sp[1])
    return "\n".join(left), "\n".join(right)


def find_blocks(pages: list[str], max_context: int = 30) -> list[Block]:
    blocks: list[Block] = []
    context: list[str] = []
    current: list[Row] = []
    current_page = 0
    line_no = 0
    for page_idx, page in enumerate(pages):
        for line in merge_wrapped_values(page.splitlines()):
            line_no += 1
            if current and (not _MARGIN.sub("", line).strip() or _only_values(line)):
                # Marge d'erreur ou valeur de rappel orpheline au milieu d'un tableau (Ipsos).
                continue
            row = None if _FOOTER.match(line) else parse_row(line, line_no)
            if row and (candidate_id(row.label) or is_non_candidate(row.label) or current):
                if not current:
                    current_page = page_idx + 1
                current.append(row)
                continue
            if current:
                blocks.append(Block(current, context[-max_context:], current_page))
                current, context = [], []
            context.append(line)
    if current:
        blocks.append(Block(current, context[-max_context:], current_page))
    return blocks


def _split_names(line: str) -> list[str] | None:
    """Découpe une ligne d'en-tête en une suite de candidats connus, ou None.

    « Fabien Roussel Jean-Luc Mélenchon Raphaël Glucksmann … » -> ids dans l'ordre ; tous les
    mots doivent être consommés.
    """
    words = norm_label(line).split()
    ids: list[str] = []
    i = 0
    while i < len(words):
        for j in range(min(len(words), i + 4), i, -1):
            cid = candidate_id(" ".join(words[i:j]))
            if cid:
                ids.append(cid)
                i = j
                break
        else:
            return None
    return ids


def _header_ids(lines: list[str], n: int) -> list[str] | None:
    """Candidats d'un en-tête de colonnes : une ligne de noms complets, ou une ligne de
    prénoms au-dessus d'une ligne de noms (« Fabien Jean-Luc … » / « ROUSSEL MELENCHON … »)."""
    if lines:
        ids = _split_names(lines[-1])
        if ids and len(ids) == n:
            return ids
    if len(lines) >= 2:
        first, last = lines[-2].split(), lines[-1].split()
        if len(first) == len(last) == n:
            ids = [candidate_id(f"{a} {b}") for a, b in zip(first, last, strict=True)]
            if all(ids):
                return ids
    return None


def find_transposed_blocks(pages: list[str]) -> list[Block]:
    """Tableaux où les candidats sont en colonnes (tris croisés Verian).

    Deux formes : « Total v1 … vN » sous l'en-tête, ou « Ensemble » seul puis la ligne de
    valeurs. L'en-tête doit se lire exactement comme N candidats connus (`_header_ids`) ;
    les en-têtes plus mélangés par la mise en page ne sont pas lus (la ligne « Total » revient
    d'ordinaire sur une page suivante avec un en-tête lisible).
    """
    blocks = []
    for page_idx, page in enumerate(pages):
        lines = page.splitlines()
        for k, line in enumerate(lines):
            row = parse_row(line, 0)
            if row and norm_label(row.label) in ("total", "ensemble"):
                values, header = row.values, lines[max(0, k - 2) : k]
            elif (
                k >= 1 and norm_label(lines[k - 1]) in ("total", "ensemble") and _only_values(line)
            ):
                parsed = parse_row("x " + line, 0)
                if parsed is None:  # ligne réduite à « % » ou « * » (notices archivées)
                    continue
                values = parsed.values
                header = lines[max(0, k - 3) : k - 1]
            else:
                continue
            ids = _header_ids(header, len(values))
            if not ids or len(ids) < 2:
                continue
            rows = [Row(cid, [v], 0) for cid, v in zip(ids, values, strict=True)]
            blocks.append(Block(rows, lines[max(0, k - 14) : k], page_idx + 1))
    return blocks


@dataclass
class Hypothesis:
    round: int
    population: str
    results: list[tuple[str, float, float | None]]  # (candidate_id, share, raw_share)
    page: int
    label: str = ""


@dataclass
class BlockOutcome:
    page: int
    status: str  # accepted, not_candidates, recall, unknown_format, bad_sum, duplicate
    detail: str = ""


@dataclass
class NoticeResults:
    hypotheses: list[Hypothesis] = field(default_factory=list)
    outcomes: list[BlockOutcome] = field(default_factory=list)


# Aucune hypothèse de premier tour publiée n'a moins de 7 candidats ; en dessous de 5, un
# bloc est un morceau de graphique ou de tableau (légendes d'évolution Toluna Harris).
MIN_FIRST_ROUND_CANDIDATES = 5

_SKIP_CONTEXT = re.compile(
    r"reconstitution|evolutions?\b|report(s)? de voix|transfert|pour memoire|primaire|souvenir de vote"
    r"|vote au (1er|premier|2nd|second) tour de l election presidentielle (de )?2022"
    r"|chances d etre elu|meilleur president|potentiel electoral|pourriez voter"
    r"|pourrait voter|pourriez vous voter|bon candidat|mieux place|souhait\w* (vous )?(que|qu|voir)"
    r"|recapitulatif|plus dangereux|voir gagner|prefer|plus proche|incarne"
    r"|ferait le meilleur"
)
_IV_CONTEXT = re.compile(
    r"\b(tour|intentions? de vote|itv|iv|hypothese|configuration|voteriez|votiez|scenario)\b"
)
_BRUT = re.compile(r"\bbruts?\b")
_ROUND2 = re.compile(r"\b(2nd|second|2d|2e|2eme) tour\b")


_HEADER_WORD = re.compile(r"\b(bruts?|redresses?|ponderes?|publies?)\b")


def _header_roles(block: Block, n: int) -> tuple[int, int | None] | None:
    """Colonnes d'après une ligne d'en-tête « Brut Redressé [Publié] » juste au-dessus.

    Le redressé retenu est la colonne « Publié » si elle existe, sinon « Redressé » ou
    « Pondérés » ; le brut est la colonne « Brut(s) ». La ligne doit nommer exactement autant
    de colonnes que de valeurs.
    """
    for line in reversed(block.context[-4:]):
        words = [w.rstrip("s") for w in _HEADER_WORD.findall(_n(line))]
        if len(words) != n or "brut" not in words:
            continue
        share = (
            words.index("publie")
            if "publie" in words
            else next((i for i, w in enumerate(words) if w in ("redresse", "pondere")), None)
        )
        if share is None:
            continue
        return share, words.index("brut")
    return None


def _column_spec(block: Block, pollster: str) -> tuple[int, int | None] | None:
    """(colonne du redressé, colonne du brut) d'après le nombre de valeurs et l'en-tête.

    Formats reconnus :
    - en-tête « Brut / Redressé|Pondérés [/ Publié] » juste au-dessus, autant de mots que de
      valeurs (Cluster17, YouGov) : rôles lus dans l'en-tête ;
    - une valeur : le résultat publié ;
    - deux valeurs sans cet en-tête : la seconde colonne est un rappel de vague précédente
      (Toluna Harris Interactive, Ipsos) ; on ne garde que la première ;
    - neuf valeurs, Ifop (grandes enquêtes 2025) : « Résultats bruts », redressements
      successifs, « Résultats publiés » en dernière colonne ;
    - neuf valeurs, OpinionWay : « Publié » en première colonne puis, pour les inscrits puis
      les certains d'aller voter, brut et trois redressements ; le publié correspond aux
      certains d'aller voter, dont le brut est la sixième colonne.
    Tout autre nombre de colonnes (récapitulatifs, tris croisés, reports) est écarté.
    """
    counts = sorted(len(r.values) for _, r in block.candidate_rows)
    n = counts[len(counts) // 2]
    roles = _header_roles(block, n)
    if roles:
        return roles
    if n == 1:
        return 0, None
    if n == 2:
        if any(_BRUT.search(_n(line)) for line in block.context[-4:]):
            return None  # en-tête brut/redressé illisible : ne pas prendre le brut pour le publié
        return 0, None
    header = " ".join(_n(line) for line in block.context[-12:])
    if n == 9 and _BRUT.search(header):
        if pollster == "ifop":
            return 8, 0
        if pollster == "opinionway":
            return 0, 5
    return None


def _block_population(block: Block) -> str | None:
    """Population d'après la ligne « Base : … » au-dessus du tableau (et ses 2 suivantes).

    On ne lit pas tout le contexte : la note de bas de la page précédente (« … et certaines
    d'aller voter interrogées, 823 personnes… », Ifop) fausserait la détection.
    """
    ctx = block.context[-15:]
    for i in range(len(ctx) - 1, -1, -1):
        if _n(ctx[i]).startswith("base"):
            pop = find_population(" ".join(ctx[i : i + 3]))
            return None if pop == "unknown" else pop
    return None


_DUAL_HEADER = re.compile(r"certain")
_DUAL_OTHER = re.compile(r"ensemble|inscrit")


def dual_columns(block: Block) -> tuple[int, int, int, int] | None:
    """Tableau OpinionWay à deux bases -> (certains, inscrits, brut certains, brut inscrits).

    1 + 2k valeurs par ligne (k = 4 ou 5), en-tête citant les certains d'aller voter et
    l'ensemble des inscrits ou des répondants : « Publié » en première colonne, puis k colonnes
    pour l'ensemble des inscrits (brut et redressements) et k pour les certains d'aller voter,
    dans le même ordre. La colonne publiée est l'un des redressements des certains ; on la
    repère à la valeur (somme des écarts minimale à la première colonne) et on prend la colonne
    des inscrits au même rang : même redressement, seule la base change."""
    counts = sorted(len(r.values) for _, r in block.candidate_rows)
    n = counts[len(counts) // 2]
    if n not in (9, 11):
        return None
    head = _n(" ".join(block.context[-14:]))
    if not (_DUAL_HEADER.search(head) and _DUAL_OTHER.search(head)):
        return None
    k = (n - 1) // 2
    rows = [r.values for _, r in block.candidate_rows if len(r.values) == n]

    def gap(j: int) -> float:
        return sum(abs(v[j] - v[0]) for v in rows if v[j] is not None and v[0] is not None)

    j = min(range(k + 2, 2 * k + 1), key=gap)  # un redressement des certains, pas le brut
    return j, j - k, k + 1, 1


# Règles de population partagées par le parseur 2027 (`parse`) et les archives (`archive`).
#
# La colonne publiée est calculée sur les certains d'aller voter alors que l'en-tête du tableau
# dit « Base : ensemble des personnes inscrites » (Ifop : « les scores établis tiennent
# principalement compte des redressements … sur la base des électeurs certains d'aller
# voter » ; Elabe : « la colonne de référence … regroupant les individus ayant donné une note
# de 8 à 10 sur l'échelle d'intention d'aller voter »).
_REFERENCE_CERTAIN = re.compile(
    r"sur la base des electeurs certains d aller voter"
    r"|colonne de reference .{0,160}?note de \d+ a 10 sur l echelle d intention d aller voter"
    r"|colonne de reference .{0,200}?sur les certains d aller voter"
    r"|resultats? (presentes? )?aupres des personnes certaines d aller voter"
    r"|base\W{0,3}certains d aller voter"
    # Ifop 2022 : note sous chaque tableau, « Sur toutes les personnes inscrites sur les listes
    # électorales et certaines d'aller voter interrogées, N personnes ont exprimé… »
    r"|personnes inscrites sur les listes electorales et certaines d aller voter interrogees"
    # Toluna Harris Interactive 2027 (depuis septembre 2026) : base en pied des tableaux
    # détaillés, « Base : (Aux) inscrits sur les listes électorales certains d'aller voter et
    # ayant exprimé une intention de vote », alors que le récapitulatif dit « Aux inscrits sur
    # les listes électorales » pour les mêmes chiffres.
    r"|base (aux )?inscrits sur les listes electorales certains d aller voter et ayant exprime"
)


def notice_population(pages: list[str], head_population: str) -> str:
    """Population par défaut des tableaux à une base : `certain_voters` si la notice dit que
    les résultats publiés portent sur les certains d'aller voter, sinon celle de l'en-tête."""
    if _REFERENCE_CERTAIN.search(_n("\n".join(pages))):
        return "certain_voters"
    return head_population


def block_population(block: Block, default_population: str) -> str:
    """Ligne « Base : … » du tableau, sauf « inscrits » quand la notice déclare publier les
    certains d'aller voter (la ligne décrit alors l'échantillon, pas la colonne publiée)."""
    pop = _block_population(block)
    if pop is None or (pop == "registered" and default_population == "certain_voters"):
        return default_population
    return pop


def _block_versions(
    block: Block, pollster: str, default_population: str
) -> list[tuple[str, list[tuple[str, float, float | None]]]] | None:
    """(population, résultats) lus dans un tableau ; None si le format est inconnu.

    Tableau OpinionWay à deux bases (`dual_columns`) : deux versions, les certains d'aller voter
    sur la colonne publiée (première colonne, brut des certains) et les inscrits sur le
    redressement de même rang (brut des inscrits). Autres tableaux : une version, population de
    `block_population`."""
    cands = block.candidate_rows

    def read(share_col: int, raw_col: int | None) -> list[tuple[str, float, float | None]]:
        out = []
        for cid, row in cands:
            share = row.values[share_col] if share_col < len(row.values) else None
            raw = row.values[raw_col] if raw_col is not None and raw_col < len(row.values) else None
            if share is None:
                continue  # candidat non testé dans cette hypothèse (tableau récapitulatif)
            out.append((cid, share, raw))
        return out

    dual = dual_columns(block) if pollster == "opinionway" else None
    if dual:
        _, reg, raw_cert, raw_reg = dual
        return [("certain_voters", read(0, raw_cert)), ("registered", read(reg, raw_reg))]
    spec = _column_spec(block, pollster)
    if spec is None:
        return None
    return [(block_population(block, default_population), read(*spec))]


def classify_blocks(blocks: list[Block], pollster: str, default_population: str) -> NoticeResults:
    """Hypothèses des tableaux d'intentions de vote d'une notice.

    Une hypothèse est identifiée par (tour, candidats, population) : un tableau publié sur deux
    bases donne deux hypothèses, de même `label` (identifiant de la configuration publiée)."""
    out = NoticeResults()
    seen: dict[tuple, int] = {}
    for block in blocks:
        cands = block.candidate_rows
        if len(cands) < 2:
            out.outcomes.append(BlockOutcome(block.page, "not_candidates"))
            continue
        ids = [c for c, _ in cands]
        # Marqueurs cherchés dans les 10 lignes au-dessus du tableau seulement : les pieds de
        # page répétés (« … après la primaire du PS ») ne doivent pas disqualifier la page.
        near = _n(" ".join(block.context[-10:]))
        if "macron_emmanuel" in ids or _SKIP_CONTEXT.search(near):
            out.outcomes.append(BlockOutcome(block.page, "recall_or_other_question"))
            continue
        unknown = [
            r.label
            for r in block.rows
            if not candidate_id(r.label) and not is_non_candidate(r.label)
        ]
        if not _IV_CONTEXT.search(_n(" ".join(block.context[-14:]))):
            # Liste de personnalités hors question de vote (popularité, préférence…).
            out.outcomes.append(BlockOutcome(block.page, "other_question", str(unknown[:3])))
            continue
        if unknown:
            raise UnknownLabelError(
                f"page {block.page} : libellé(s) inconnu(s) dans un tableau de candidats : {unknown}"
            )
        versions = _block_versions(block, pollster, default_population)
        if versions is None:
            out.outcomes.append(
                BlockOutcome(block.page, "unknown_format", f"{len(cands[0][1].values)} valeurs")
            )
            continue
        # Configuration publiée (première version), commune aux deux bases d'un même tableau.
        label = schemas.hypothesis_id([c for c, _, _ in versions[0][1]])
        for population, results in versions:
            if len({c for c, _, _ in results}) != len(results):
                out.outcomes.append(BlockOutcome(block.page, "duplicate_candidate"))
                continue
            total = sum(s for _, s, _ in results)
            if not 94 <= total <= 106:
                out.outcomes.append(BlockOutcome(block.page, "bad_sum", f"somme {total:g}"))
                continue
            rnd = 2 if len(results) == 2 else 1
            if rnd == 1 and len(results) < MIN_FIRST_ROUND_CANDIDATES:
                out.outcomes.append(
                    BlockOutcome(block.page, "too_few_candidates", str(len(results)))
                )
                continue
            if (
                rnd == 1
                and _ROUND2.search(_n(" ".join(block.context[-6:])))
                and not re.search(r"\b(1er|premier) tour\b", _n(" ".join(block.context[-6:])))
            ):
                out.outcomes.append(BlockOutcome(block.page, "round_mismatch"))
                continue
            key = (rnd, tuple(sorted(c for c, _, _ in results)), population)
            has_raw = any(r is not None for _, _, r in results)
            hyp = Hypothesis(rnd, population, results, block.page, label=label)
            if key in seen:
                prev = out.hypotheses[seen[key]]
                prev_raw = any(r is not None for _, _, r in prev.results)
                if has_raw and not prev_raw:
                    out.hypotheses[seen[key]] = hyp
                out.outcomes.append(BlockOutcome(block.page, "duplicate"))
                continue
            seen[key] = len(out.hypotheses)
            out.hypotheses.append(hyp)
            out.outcomes.append(BlockOutcome(block.page, "accepted"))
    return out
