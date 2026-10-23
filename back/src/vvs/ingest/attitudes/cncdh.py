"""Indice longitudinal de tolérance (CNCDH, V. Tiberj) : lecture du graphique vectoriel.

Le rapport annuel de la CNCDH ne publie pas de tableau de l'indice : la figure « L'indice
longitudinal de tolérance (1990-2025) » est une courbe vectorielle dont chaque année porte un
petit marqueur rond (3 points de diamètre, couleur de la courbe), et dont une vingtaine
d'années seulement ont une étiquette de valeur. On lit donc :

- les marqueurs (pdfplumber, courbes de moins de 4 points de côté dans la couleur déclarée),
  dédoublonnés (deux marqueurs superposés pour 2006 et 2025) ;
- l'échelle verticale, par les lignes de grille horizontales associées à leurs graduations
  (texte « 45 », « 47,5 »… à gauche du graphique), ajustée par régression linéaire ;
- l'année de chaque marqueur, par interpolation entre le premier marqueur (`first_year`) et le
  dernier (`last_year`), déclarés dans le registre d'après l'axe ; chaque marqueur doit tomber
  à moins d'un tiers de pas d'une année entière ;
- les étiquettes de valeur (pdftotext -bbox : pdfplumber ne voit pas ce texte) : chacune doit
  être égale à la valeur du marqueur le plus proche, arrondie à l'entier.

La valeur retenue est celle du marqueur, arrondie à l'entier quand elle en est à moins de
0,2 (l'indice est publié en entiers) ; sinon la lecture s'arrête. Méthode : `chart_marker`
(`chart_label` pour les années étiquetées, mêmes valeurs).
"""

from __future__ import annotations

import re
import subprocess

from vvs.ingest.attitudes import Obs, pdfchart, raw
from vvs.ingest.attitudes.registry import AttitudeRegistry, Extraction


class CncdhError(ValueError):
    pass


def _value_words(path, page: int, bbox: list[float]) -> list[tuple[float, float, float]]:
    """(x centre, y centre, valeur) des nombres du graphique selon pdftotext -bbox."""
    r = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), "-bbox", str(path), "-"],
        capture_output=True, text=True, check=True,
    )
    out = []
    pat = r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>'
    for m in re.finditer(pat, r.stdout):
        x0, y0, x1, y1 = (float(v) for v in m.groups()[:4])
        t = m.group(5)
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        if re.fullmatch(r"\d{2}", t) and pdfchart.in_bbox(xc, y0, bbox):
            out.append((xc, yc, float(t)))
    return out


def extract(reg: AttitudeRegistry, e: Extraction) -> list[Obs]:
    (sid,) = e.spec["series"]
    path = raw.read(reg.documents[e.document]).path
    years, labelled = read_chart(path, e.page, e.spec, e.id)
    note = e.spec.get("note", "")
    return [
        Obs(sid, str(yr), v, "chart_label" if yr in labelled else "chart_marker", e.document,
            e.page, note)
        for yr, v in sorted(years.items())
    ]


def read_chart(path, page_no: int, sp: dict, name: str = "") -> tuple[dict[int, float], set[int]]:
    """({année: valeur}, années recoupées par une étiquette) d'un graphique de l'indice."""
    import pdfplumber

    e_id = name
    bbox = sp["bbox"]
    color = tuple(sp["marker_color"])
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[page_no - 1]
        marks = sorted(
            {
                (round((c["x0"] + c["x1"]) / 2, 1), round((c["top"] + c["bottom"]) / 2, 1))
                for c in page.curves
                if c["x1"] - c["x0"] < 4
                and c["bottom"] - c["top"] < 4
                and pdfchart.same_color(pdfchart.color_of(c), color, tol=0.01)
                and pdfchart.in_bbox((c["x0"] + c["x1"]) / 2, c["top"], bbox)
            }
        )
        ticks = [
            lab for lab in pdfchart.labels(page, sp["ticks_bbox"])
        ]
        grid = sorted(
            {round(ln["top"], 2) for ln in page.lines if abs(ln["top"] - ln["bottom"]) < 0.5
             and ln["x1"] - ln["x0"] > 50  # lignes de grille, pas les petites graduations
             and pdfchart.in_bbox((ln["x0"] + ln["x1"]) / 2, ln["top"], bbox)}
        )
    # marqueurs superposés (même année dessinée deux fois)
    dedup: list[tuple[float, float]] = []
    for x, y in marks:
        if dedup and abs(dedup[-1][0] - x) < 1.0:
            if abs(dedup[-1][1] - y) > 0.5:
                raise CncdhError(f"{e_id} : deux marqueurs différents à x={x}")
            continue
        dedup.append((x, y))
    # échelle : chaque graduation va à la ligne de grille la plus proche
    pairs = []
    for t in ticks:
        yc = t.top + t.size / 2
        g = min(grid, key=lambda gy: abs(gy - yc))
        if abs(g - yc) > 3:
            raise CncdhError(f"{e_id} : graduation {t.text} sans ligne de grille")
        pairs.append((g, t.value))
    if len(pairs) < 3:
        raise CncdhError(f"{e_id} : {len(pairs)} graduation(s) seulement")
    n = len(pairs)
    my = sum(g for g, _ in pairs) / n
    mv = sum(v for _, v in pairs) / n
    slope = sum((g - my) * (v - mv) for g, v in pairs) / sum((g - my) ** 2 for g, _ in pairs)
    resid = max(abs(mv + slope * (g - my) - v) for g, v in pairs)
    if resid > 0.05:
        raise CncdhError(f"{e_id} : échelle non linéaire (résidu {resid:.3f})")
    first, last = int(sp["first_year"]), int(sp["last_year"])
    x0, x1 = dedup[0][0], dedup[-1][0]
    step = (x1 - x0) / (last - first)
    years: dict[int, float] = {}
    for x, y in dedup:
        pos = (x - x0) / step
        year = first + round(pos)
        if abs(pos - round(pos)) > 1 / 3:
            raise CncdhError(f"{e_id} : marqueur x={x} entre deux années ({first + pos:.2f})")
        if year in years:
            raise CncdhError(f"{e_id} : deux marqueurs pour {year}")
        v = mv + slope * (y - my)
        if abs(v - round(v)) > 0.2:
            raise CncdhError(f"{e_id} : {year} lu {v:.2f}, trop loin d'un entier")
        years[year] = float(round(v))
    # contrôle par les étiquettes imprimées
    labelled: set[int] = set()
    for xc, yc, val in _value_words(path, page_no, bbox):
        if yc > max(grid):  # sous l'axe : morceaux des années
            continue
        # l'étiquette est posée à côté de son marqueur : on cherche, à moins de deux pas, le
        # marqueur de même valeur le plus proche
        cands = [
            yr for yr in years
            if abs(x0 + (yr - first) * step - xc) <= 2 * step and years[yr] == val
        ]
        if not cands:
            raise CncdhError(f"{e_id} : étiquette {val:g} à x={xc:.1f} sans marqueur égal proche")
        labelled.add(min(cands, key=lambda yr: abs(x0 + (yr - first) * step - xc)))
    if len(labelled) < int(sp.get("min_labels", 1)):
        raise CncdhError(f"{e_id} : {len(labelled)} étiquette(s) recoupée(s) seulement")
    return years, labelled
