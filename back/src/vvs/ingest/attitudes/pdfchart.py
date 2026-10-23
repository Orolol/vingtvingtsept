"""Lecture des graphiques d'évolution et des tableaux des rapports PDF (Verian, OpinionWay).

Les rapports de baromètre publient leurs séries longues sous forme de courbes étiquetées : chaque
point porte sa valeur en texte, dans la couleur de sa courbe. On lit donc la couche texte du PDF
(pdfplumber), jamais l'image :

- `labels(page)` regroupe les chiffres consécutifs d'une même couleur, d'une même ligne, en
  étiquettes numériques (« 45 », « 45% », « 7,3 ») avec leur position et leur couleur ;
- `extract_chart(page, spec)` répartit les étiquettes d'un graphique entre séries (par couleur)
  et entre vagues (par colonne). Les colonnes viennent soit de l'axe des abscisses quand il est
  en texte (`columns: axis`, libellés « Déc. 2009 », « Janv. 2026 »), soit d'un regroupement des
  abscisses des étiquettes (`columns: cluster`) quand l'axe est dessiné en image : la liste
  des vagues est alors déclarée dans le registre, lue sur l'axe du graphique, et le nombre de
  colonnes trouvées doit lui être égal ;
- `extract_table(page, spec)` lit un tableau de séries (une ligne par item, une colonne par
  vague, en-têtes « Mai 2000 », « Nov. 2023 »).

Toute incohérence (nombre de colonnes, deux valeurs d'une même série dans une colonne, étiquette
trop loin de toute colonne) lève `ChartError` : un graphique mal lu n'est jamais complété.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from unidecode import unidecode

MONTHS = {
    "janv": 1, "janvier": 1, "jan": 1,
    "fevr": 2, "fevrier": 2, "fev": 2,
    "mars": 3,
    "avril": 4, "avr": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7, "juillet": 7,
    "aout": 8,
    "sept": 9, "septembre": 9,
    "oct": 10, "octobre": 10,
    "nov": 11, "novembre": 11,
    "dec": 12, "decembre": 12,
}  # fmt: skip
MONTH_RE = re.compile(r"^(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?$")
YEAR_RE = re.compile(r"^(19[5-9]\d|20[0-4]\d)\*?$")
NUM_CHARS = set("0123456789,%")


class ChartError(ValueError):
    """Graphique ou tableau dont la lecture n'est pas univoque."""


@dataclass(frozen=True)
class Label:
    value: float
    text: str
    x: float  # centre horizontal
    x0: float
    x1: float
    top: float
    color: tuple[float, ...]
    size: float


def norm(text: str) -> str:
    return unidecode(text).lower().strip()


def color_of(obj: dict) -> tuple[float, ...]:
    c = obj.get("non_stroking_color")
    if c is None:
        return ()
    if isinstance(c, (int, float)):
        c = (c,)
    return tuple(round(float(v), 3) for v in c)


def same_color(a: tuple[float, ...], b: tuple[float, ...], tol: float = 0.02) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def in_bbox(x: float, top: float, bbox: list[float] | None) -> bool:
    if not bbox:
        return True
    x0, t0, x1, t1 = bbox
    return x0 <= x <= x1 and t0 <= top <= t1


def labels(page, bbox: list[float] | None = None) -> list[Label]:
    """Étiquettes numériques de la page : chiffres contigus de même couleur sur une même ligne.

    Un « % » final est accepté, la virgule décimale française aussi. Deux étiquettes accolées
    (« 3739 ») sont séparées dès que leurs lignes de base diffèrent."""
    chars = [c for c in page.chars if c["text"] in NUM_CHARS and c.get("upright", True)]
    chars.sort(key=lambda c: (color_of(c), round(c["top"], 0), c["x0"]))
    groups: list[list[dict]] = []
    for c in chars:
        g = groups[-1] if groups else None
        if (
            g is not None
            and color_of(g[-1]) == color_of(c)
            and abs(g[-1]["top"] - c["top"]) < 1.0
            and -0.5 <= c["x0"] - g[-1]["x1"] < 1.5
        ):
            g.append(c)
        else:
            groups.append([c])
    out = []
    for g in groups:
        text = "".join(c["text"] for c in g)
        core = text.rstrip("%")
        if not re.fullmatch(r"\d{1,3}(,\d{1,2})?", core):
            continue
        x0, x1 = g[0]["x0"], g[-1]["x1"]
        if text.endswith("%"):  # centre du nombre, pas du symbole
            x1 = g[-2]["x1"]
        x = (x0 + x1) / 2
        top = min(c["top"] for c in g)
        if not in_bbox(x, top, bbox):
            continue
        out.append(
            Label(
                value=float(core.replace(",", ".")),
                text=text,
                x=x,
                x0=x0,
                x1=x1,
                top=top,
                color=color_of(g[0]),
                size=round(max(c["size"] for c in g), 1),
            )
        )
    return out


# ------------------------------------------------------------------------------- colonnes


def wave_of(month_word: str, year: str) -> str | None:
    m = MONTH_RE.match(norm(month_word))
    y = YEAR_RE.match(year)
    if not (m and y):
        return None
    return f"{y.group(1)}-{MONTHS[m.group(1)]:02d}"


def axis_columns(page, bbox: list[float]) -> list[tuple[str, float]]:
    """Vagues lues sur un axe des abscisses en texte, de gauche à droite : [(AAAA-MM, x)].

    Accepte « Déc. 2009 » sur une ligne, « Nov.2023 », ou le mois au-dessus de l'année."""
    words = [
        w
        for w in page.extract_words(keep_blank_chars=False, use_text_flow=False)
        if in_bbox((w["x0"] + w["x1"]) / 2, w["top"], bbox)
    ]
    cols: list[tuple[str, float]] = []
    used: set[int] = set()
    for i, w in enumerate(words):
        t = w["text"]
        glued = re.match(r"^([A-Za-zéèûô]+)\.?(\d{4}\*?)$", t)
        if glued and wave_of(glued.group(1), glued.group(2)):
            cols.append((wave_of(glued.group(1), glued.group(2)), (w["x0"] + w["x1"]) / 2))
            used.add(i)
            continue
        if not YEAR_RE.match(t):
            continue
        xc = (w["x0"] + w["x1"]) / 2
        best = None
        for j, m in enumerate(words):
            if j in used or j == i or not MONTH_RE.match(norm(m["text"])):
                continue
            mx = (m["x0"] + m["x1"]) / 2
            same_line = abs(m["top"] - w["top"]) < 2 and 0 <= w["x0"] - m["x1"] < 8
            above = 0 < w["top"] - m["top"] < 16 and abs(mx - xc) < 14
            if same_line or above:
                d = abs(mx - xc) + abs(w["top"] - m["top"])
                if best is None or d < best[0]:
                    best = (d, j, m)
        if best is None:
            continue
        _, j, m = best
        used.add(j)
        x = xc if not (abs(m["top"] - w["top"]) < 2) else (m["x0"] + w["x1"]) / 2
        cols.append((wave_of(m["text"], t), x))
    cols.sort(key=lambda c: c[1])
    return cols


def cluster(xs: list[float], gap: float) -> list[float]:
    """Centres des groupes d'abscisses séparés par plus de `gap` points."""
    if not xs:
        return []
    xs = sorted(xs)
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] > gap:
            groups.append([x])
        else:
            groups[-1].append(x)
    return [sum(g) / len(g) for g in groups]


# ------------------------------------------------------------------------------ graphique


def align(labels_x: list[float], cols_x: list[float]) -> list[int]:
    """Range des étiquettes triées dans des colonnes triées, dans l'ordre, une au plus par
    colonne, en minimisant la somme des écarts horizontaux (programmation dynamique). Une série
    incomplète saute ainsi les bonnes colonnes même quand ses étiquettes sont décalées."""
    m, n = len(labels_x), len(cols_x)
    inf = float("inf")
    # cost[i][j] : coût minimal des i premières étiquettes dans les j premières colonnes
    cost = [[inf] * (n + 1) for _ in range(m + 1)]
    take = [[False] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        cost[0][j] = 0.0
    for i in range(1, m + 1):
        for j in range(i, n + 1):
            skip = cost[i][j - 1]
            use = cost[i - 1][j - 1] + abs(labels_x[i - 1] - cols_x[j - 1])
            if use <= skip:
                cost[i][j], take[i][j] = use, True
            else:
                cost[i][j] = skip
    out = [0] * m
    i, j = m, n
    while i > 0:
        if take[i][j]:
            out[i - 1] = j - 1
            i -= 1
        j -= 1
    return out


@dataclass
class SeriesSpec:
    series_id: str
    color: tuple[float, ...]
    bbox: list[float] | None = None
    extra: list[dict] = field(default_factory=list)  # [{color, bbox}] : étiquettes isolées
    missing: list[str] = field(default_factory=list)  # vagues sans étiquette (vérifié à l'image)


@dataclass
class ChartSpec:
    series: list[SeriesSpec]
    columns: str = "cluster"  # cluster | axis
    waves: list[str] | None = None
    bbox: list[float] | None = None
    axis_bbox: list[float] | None = None
    col_gap: float = 5.0
    max_dist: float | None = None  # défaut : 90 % du pas médian entre colonnes


def extract_chart(page, spec: ChartSpec) -> dict[str, list[tuple[str, float]]]:
    """{series_id: [(vague AAAA-MM, valeur)]} pour un graphique d'évolution."""
    found = labels(page, spec.bbox)
    per_series: dict[str, list[Label]] = {}
    for s in spec.series:
        mine = [
            lab
            for lab in found
            if same_color(lab.color, s.color) and in_bbox(lab.x, lab.top, s.bbox)
        ]
        for e in s.extra:
            mine += [
                lab
                for lab in found
                if same_color(lab.color, tuple(e["color"])) and in_bbox(lab.x, lab.top, e["bbox"])
            ]
        if not mine:
            raise ChartError(f"{s.series_id} : aucune étiquette de couleur {s.color}")
        per_series[s.series_id] = mine
    if spec.columns == "axis":
        if not spec.axis_bbox:
            raise ChartError("columns: axis sans axis_bbox")
        cols = axis_columns(page, spec.axis_bbox)
        waves = [w for w, _ in cols]
        xs = [x for _, x in cols]
        if spec.waves is not None and spec.waves != waves:
            raise ChartError(f"axe lu {waves} différent des vagues déclarées {spec.waves}")
    else:
        if spec.waves is None:
            raise ChartError("columns: cluster exige la liste des vagues")
        waves = list(spec.waves)
        full = [sorted(lab.x for lab in labs) for labs in per_series.values() if len(labs) == len(waves)]
        if full:
            # une série complète fixe les colonnes (les étiquettes se décalent pour ne pas se
            # chevaucher : un regroupement brut fusionnerait ou scinderait des colonnes) ; ses
            # étiquettes, et celles de toute série complète, se lisent dans l'ordre
            xs = [sum(col) / len(col) for col in zip(*full, strict=True)]
        else:
            xs = cluster([lab.x for labs in per_series.values() for lab in labs], spec.col_gap)
        if len(xs) != len(waves):
            raise ChartError(
                f"{len(xs)} colonnes d'étiquettes pour {len(waves)} vagues déclarées "
                f"(abscisses {[round(x) for x in xs]})"
            )
    if len(set(waves)) != len(waves):
        raise ChartError(f"vagues en double : {waves}")
    max_dist = spec.max_dist
    if max_dist is None:
        steps = sorted(b - a for a, b in itertools.pairwise(xs))
        max_dist = 0.9 * steps[len(steps) // 2] if steps else 20.0
    out: dict[str, list[tuple[str, float]]] = {}
    for sid, labs in per_series.items():
        by_wave: dict[str, Label] = {}
        if len(labs) == len(waves):  # série complète : lecture dans l'ordre
            ordered = sorted(labs, key=lambda lab: lab.x)
            out[sid] = [(w, lab.value) for w, lab in zip(waves, ordered, strict=True)]
            continue
        if len(labs) > len(waves):
            raise ChartError(f"{sid} : {len(labs)} étiquettes pour {len(waves)} vagues")
        ordered = sorted(labs, key=lambda lab: lab.x)
        spec_s = next(s for s in spec.series if s.series_id == sid)
        allowed = [k for k, w in enumerate(waves) if w not in spec_s.missing]
        if len(ordered) > len(allowed):
            raise ChartError(f"{sid} : {len(ordered)} étiquettes pour {len(allowed)} vagues")
        cols = [allowed[k] for k in align([lab.x for lab in ordered], [xs[k] for k in allowed])]
        for lab, k in zip(ordered, cols, strict=True):
            if abs(xs[k] - lab.x) > max_dist:
                raise ChartError(
                    f"{sid} : étiquette {lab.text} à x={lab.x:.1f}, rangée en {waves[k]} "
                    f"(colonne à {xs[k]:.1f}, écart toléré {max_dist:.1f})"
                )
            by_wave[waves[k]] = lab
        out[sid] = [(w, by_wave[w].value) for w in waves if w in by_wave]
    return out


# -------------------------------------------------------------------------------- tableau


@dataclass
class TableSpec:
    rows: dict[str, str]  # series_id -> extrait distinctif du libellé de ligne
    header_bbox: list[float]
    label_x1: float  # bord droit de la colonne des libellés
    bbox: list[float] | None = None
    waves: list[str] | None = None
    header: str = "month_year"  # ou "year" : en-têtes d'années seules (« 1991 … 2015 »)


def year_columns(page, bbox: list[float]) -> list[tuple[str, float]]:
    words = [
        w for w in page.extract_words()
        if in_bbox((w["x0"] + w["x1"]) / 2, w["top"], bbox) and YEAR_RE.match(w["text"])
    ]
    return sorted(((w["text"][:4], (w["x0"] + w["x1"]) / 2) for w in words), key=lambda c: c[1])


def extract_table(page, spec: TableSpec) -> dict[str, list[tuple[str, float]]]:
    """{series_id: [(vague, valeur)]} pour un tableau « items × vagues ».

    Chaque mot de la colonne des libellés est rattaché à la ligne de valeurs la plus proche ;
    une ligne est reconnue par l'extrait distinctif de son libellé (`rows`)."""
    words = page.extract_words()
    if spec.header == "declared":
        # en-têtes illisibles (années écrites verticalement, un chiffre par ligne) : vagues
        # déclarées, colonnes prises sur les lignes complètes du tableau
        if not spec.waves:
            raise ChartError("header: declared exige la liste des vagues")
        waves = list(spec.waves)
        xs = []
    else:
        if spec.header == "year":
            cols = year_columns(page, spec.header_bbox)
        else:
            cols = axis_columns(page, spec.header_bbox)
        waves = [w for w, _ in cols]
        if spec.waves is not None and spec.waves != waves:
            raise ChartError(f"en-tête lu {waves} différent des vagues déclarées {spec.waves}")
        xs = [x for _, x in cols]
        if not xs:
            raise ChartError(f"aucune vague lue dans l'en-tête {spec.header_bbox}")
    body = [
        w for w in words if in_bbox((w["x0"] + w["x1"]) / 2, w["top"], spec.bbox)
        and w["top"] > spec.header_bbox[3]
    ]
    nums = [
        w for w in body if w["x0"] > spec.label_x1 and re.fullmatch(r"\d{1,3}", w["text"])
    ]
    text = [w for w in body if w["x1"] <= spec.label_x1 + 2]
    lines: list[list[dict]] = []
    for w in sorted(nums, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) < 3:
            lines[-1].append(w)
        else:
            lines.append([w])
    if spec.header == "declared":
        full = [sorted((w["x0"] + w["x1"]) / 2 for w in ln) for ln in lines if len(ln) == len(waves)]
        if not full:
            raise ChartError("aucune ligne complète pour situer les colonnes")
        xs = [sum(c) / len(c) for c in zip(*full, strict=True)]
    out: dict[str, list[tuple[str, float]]] = {}
    step = min(b - a for a, b in itertools.pairwise(xs)) if len(xs) > 1 else 50
    # chaque mot de libellé va à la ligne de valeurs la plus proche verticalement (un libellé
    # sur deux lignes encadre souvent sa ligne de valeurs)
    owned: dict[int, list[dict]] = {i: [] for i in range(len(lines))}
    for w in text:
        if not lines:
            break
        i = min(range(len(lines)), key=lambda i: abs(lines[i][0]["top"] - w["top"]))
        if abs(lines[i][0]["top"] - w["top"]) <= 16:
            owned[i].append(w)
    for i, line in enumerate(lines):
        label = " ".join(
            w["text"] for w in sorted(owned[i], key=lambda w: (round(w["top"]), w["x0"]))
        )
        key = _squash(label)
        hits = [sid for sid, anchor in spec.rows.items() if _squash(anchor) in key]
        if len(hits) > 1:
            raise ChartError(f"ligne « {label} » : plusieurs séries {hits}")
        if not hits:
            continue
        sid = hits[0]
        if sid in out:
            raise ChartError(f"{sid} : deux lignes de valeurs")
        vals: dict[str, float] = {}
        ordered = sorted(line, key=lambda w: w["x0"])
        ks = align([(w["x0"] + w["x1"]) / 2 for w in ordered], xs)
        for w, k in zip(ordered, ks, strict=True):
            xc = (w["x0"] + w["x1"]) / 2
            if abs(xs[k] - xc) > step / 2:
                raise ChartError(f"{sid} : valeur {w['text']} hors colonne (x={xc:.1f})")
            if waves[k] in vals:
                raise ChartError(f"{sid} : deux valeurs pour {waves[k]}")
            vals[waves[k]] = float(w["text"])
        out[sid] = [(wv, vals[wv]) for wv in waves if wv in vals]
    missing = set(spec.rows) - set(out)
    if missing:
        raise ChartError(f"lignes introuvables : {sorted(missing)}")
    return out


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unidecode(text).lower())
