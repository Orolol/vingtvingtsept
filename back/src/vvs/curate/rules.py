"""Règles explicites de la curation des sondages (docs/03-donnees.md, « Curation »).

Chaque correction de la fusion passe par une règle de ce module, jamais par une retouche des
données. Les règles sont génériques ; les divergences connues entre sources
(`KNOWN_DIVERGENCES`) sont listées avec la règle qui les résout, et `check_known_divergences`
vérifie après chaque curation que le journal en garde la trace.

Règles de fusion (appliquées par `vvs.curate.merge`) :

- `date_match` : deux sondages du même institut, à la même élection, dont les débuts et les
  fins de terrain diffèrent d'au plus `DATE_TOLERANCE_DAYS` jour, sont le même sondage. Le
  commanditaire départage plusieurs candidats (Wikipédia ne le donne presque jamais, et les
  sources l'écrivent différemment : « LCI, le Figaro et Sud » / « LCI, Le Figaro, Sud
  Radio ») ; un désaccord franc est journalisé, il ne sépare pas deux sondages aux dates
  identiques.
- `same_results` : un sondage sans correspondant de dates dont toutes les hypothèses
  figurent à l'identique (`MATCH_SHARE_TOLERANCE`) dans un sondage du même institut terminé
  à moins de `SAME_RESULTS_WINDOW_DAYS` jours, avec le même ensemble d'hypothèses, est le
  même sondage aux dates erronées : les dates de la source prioritaire l'emportent.
- `multi_wave_split` : si ce sondage plus ancien a déjà un correspondant dans la même
  source, ou s'il contient plus d'hypothèses, c'est une notice qui publie deux vagues de
  terrain (Ifop pour Hexagone, avril 2025) : les hypothèses partagées passent dans un sondage
  à part, avec les dates et l'effectif de la vague, et les valeurs de la source prioritaire.
- hypothèses : même tour et même `hypothesis_id` = même hypothèse ; à défaut, une hypothèse
  qui ne diffère que par un candidat, à parts identiques (`label_swap`), est la même
  hypothèse au libellé décalé ; à défaut, une hypothèse présente dans une vague sœur (sondage
  fusionné qui partage une notice) y est rattachée ; sinon elle s'ajoute au sondage.
- précédence des valeurs en conflit : `PRECEDENCE` ; tout écart est journalisé. Seule
  exception : deux candidats aux parts échangées entre sources, quand la version prioritaire
  jure avec les autres hypothèses du même sondage (`prefer_swapped`, règle
  `swap_consistency`).
- population : la Commission prime ; une hypothèse sans population hérite de la population
  unique des hypothèses Commission du même sondage ; la population MieuxVoter n'est pas reprise
  pour un institut où elle contredit la Commission sur la plupart des sondages communs
  (`mieuxvoter_unreliable_pollsters`).
- population lue dans les notices (`vvs.ingest.commission.notice`, règles communes aux
  archives 2017-2022) : un tableau dont la ligne « Base » dit « inscrits » est compté sur les
  certains d'aller voter quand la notice déclare publier les certains (baromètres Ifop : « sur
  toutes les personnes inscrites sur les listes électorales et certaines d'aller voter
  interrogées » ; Toluna Harris Interactive depuis septembre 2026 : « Base : (Aux) inscrits sur
  les listes électorales certains d'aller voter et ayant exprimé une intention de vote »).
- `second_base` : une notice qui publie le même tableau sur deux bases (OpinionWay : inscrits
  et certains d'aller voter) donne deux hypothèses de même `hypothesis_id` et de `population`
  distincte, toutes deux gardées dans `poll_results` (clé d'unicité étendue à la population).
  La base prioritaire (`POPULATION_PRIORITY`, les certains d'aller voter d'abord, qui sont la
  colonne publiée et recopiée par Wikipédia et MieuxVoter) porte l'hypothèse fusionnée ; la
  seconde base suit son sondage sans être comparée aux autres sources, et reçoit les mêmes
  corrections de libellés (`swap_consistency`).
- `one_base` : tout usage qui compte des hypothèses (modèle via `load_polls`, parts par bloc,
  erreurs historiques) ne garde qu'une base par (sondage, tour, hypothèse), la première de
  `POPULATION_PRIORITY` présente ; une hypothèse publiée sur une seule base garde la sienne.
  La règle porte sur l'hypothèse et non sur le sondage entier : les populations reprises de
  MieuxVoter ou de Wikipédia mélangent parfois deux bases dans un même sondage (YouGov
  septembre 2026, Odoxa avril 2025) sans qu'aucune hypothèse soit en double.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl
from unidecode import unidecode

from vvs import schemas

PRECEDENCE: tuple[str, ...] = ("commission", "wikipedia", "mieuxvoter")
# Base retenue quand une même hypothèse est publiée sur plusieurs populations (`one_base`).
POPULATION_PRIORITY: tuple[str, ...] = ("certain_voters", "registered", "unknown")

DATE_TOLERANCE_DAYS = 1
SAME_RESULTS_WINDOW_DAYS = 45
# Écart de part au-delà duquel deux valeurs sont en conflit (journal) …
CONFLICT_TOLERANCE = 1e-6
# … et en deçà duquel deux hypothèses sont « identiques » pour rapprocher des sondages :
# « <0,5 % » vaut 0,25 sur Wikipédia et souvent 0,5 dans les notices.
MATCH_SHARE_TOLERANCE = 0.25

# Suspects (niveau hypothèse) : docs/03-donnees.md, « somme entre 97 et 103 ».
SHARE_SUM_BOUNDS = (97.0, 103.0)
MIN_FIRST_ROUND_CANDIDATES = 5
# Liste officielle réputée connue `OFFICIAL_LIST_LEAD_DAYS` jours avant le premier tour :
# le Conseil constitutionnel la publie trois à cinq semaines avant ; 14 jours laisse de la
# marge (en 2002, Pasqua est encore testé fin mars).
OFFICIAL_LIST_LEAD_DAYS = 14
# Au-delà de cette date, une hypothèse de premier tour à laquelle manquent au moins
# `MAX_MISSING_OFFICIAL` candidats officiels est incomplète (omettre un seul petit candidat,
# Schivardi en 2007, est une pratique d'institut, pas un défaut).
MAX_MISSING_OFFICIAL = 2

ROLLING_LABEL_RE = re.compile(r"rolling|presitrack", re.IGNORECASE)
ROLLING_MIN_CHAIN = 3
ROLLING_MAX_STEP_DAYS = 4
ROLLING_MAX_SAMPLE_RATIO = 1.5

MV_UNRELIABLE_MIN_POLLS = 2
# Une inversion de deux candidats entre sources n'est tranchée contre la précédence que si
# la version de la source moins prioritaire est plus proche, d'au moins cette marge (points),
# des parts médianes des deux candidats dans les autres hypothèses du même sondage.
SWAP_CONSISTENCY_MARGIN = 2.0

RESERVED = frozenset(schemas.RESERVED_CANDIDATE_IDS)
EXPRESSED_RESERVED = frozenset({"other"})  # réservé mais en base exprimés


# ---------------------------------------------------------------------------
# Commanditaires
# ---------------------------------------------------------------------------

_SPONSOR_STOP = frozenset({"le", "la", "les", "l", "et", "de", "du", "des", "pour", "journal", "radio"})


def sponsor_tokens(sponsor: str | None) -> frozenset[str]:
    """Mots significatifs d'un commanditaire (« L'Opinion et Sud Radio » -> {opinion, sud})."""
    if not sponsor:
        return frozenset()
    words = re.findall(r"[a-z0-9]+", unidecode(sponsor).lower())
    return frozenset(w for w in words if w not in _SPONSOR_STOP and len(w) > 1)


def sponsors_compatible(a: str | None, b: str | None) -> bool:
    """Vrai si l'un est inconnu ou s'ils partagent un mot significatif."""
    ta, tb = sponsor_tokens(a), sponsor_tokens(b)
    return not ta or not tb or bool(ta & tb)


# ---------------------------------------------------------------------------
# Hypothèses
# ---------------------------------------------------------------------------


def named(shares: dict[str, float]) -> set[str]:
    return {c for c in shares if c not in RESERVED}


def same_shares(a: dict[str, float], b: dict[str, float], tol: float = MATCH_SHARE_TOLERANCE) -> bool:
    """Mêmes candidats nommés et parts à `tol` près."""
    na, nb = named(a), named(b)
    return na == nb and all(abs(a[c] - b[c]) <= tol + 1e-9 for c in na)


def label_swap(
    high: dict[str, float], low: dict[str, float], tol: float = MATCH_SHARE_TOLERANCE
) -> tuple[str, str] | None:
    """(candidat de `high`, candidat de `low`) si les deux hypothèses ne diffèrent que par
    un candidat, à parts identiques : colonne décalée dans une source (Elabe, mars 2026)."""
    nh, nl = named(high), named(low)
    only_h, only_l = nh - nl, nl - nh
    if len(nh) != len(nl) or len(only_h) != 1 or len(only_l) != 1:
        return None
    ch, cl = next(iter(only_h)), next(iter(only_l))
    if abs(high[ch] - low[cl]) > tol + 1e-9:
        return None
    if any(abs(high[c] - low[c]) > tol + 1e-9 for c in nh & nl):
        return None
    return ch, cl


def swapped_pair(
    top: dict[str, float], alt: dict[str, float], tol: float = MATCH_SHARE_TOLERANCE
) -> tuple[str, str] | None:
    """(a, b) si `alt` ne diffère de `top` que par l'échange des parts de a et b."""
    if named(top) != named(alt):
        return None
    diff = sorted(c for c in named(top) if abs(top[c] - alt[c]) > tol + 1e-9)
    if len(diff) != 2:
        return None
    a, b = diff
    if abs(top[a] - alt[b]) <= tol + 1e-9 and abs(top[b] - alt[a]) <= tol + 1e-9:
        return a, b
    return None


def prefer_swapped(
    top: dict[str, float], alt: dict[str, float], pair: tuple[str, str],
    others: list[dict[str, float]],
) -> bool:
    """Règle `swap_consistency` : vrai si la version `alt` des deux candidats échangés est
    plus cohérente avec les autres hypothèses du sondage (parts médianes) que `top`, d'au
    moins `SWAP_CONSISTENCY_MARGIN`. Cas réel : OpinionWay, septembre et octobre 2026, une
    page de la notice inverse les libellés Le Pen (34 %) et Zemmour (4 %), que Wikipédia et
    MieuxVoter ont rétablis."""
    meds = []
    for c in pair:
        vals = sorted(o[c] for o in others if c in o)
        if not vals:
            return False
        mid = len(vals) // 2
        meds.append(vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2)
    dev_top = sum(abs(top[c] - m) for c, m in zip(pair, meds))
    dev_alt = sum(abs(alt[c] - m) for c, m in zip(pair, meds))
    return dev_alt + SWAP_CONSISTENCY_MARGIN < dev_top


def population_rank(population: str) -> int:
    return POPULATION_PRIORITY.index(population)


def one_base(results: pl.DataFrame) -> pl.DataFrame:
    """Règle `one_base` : une seule population par (sondage, tour, hypothèse), la première de
    `POPULATION_PRIORITY` présente. Colonnes : poll_id, round, hypothesis_id, population ; une
    table sans population (unités des scrutins de liste) est rendue telle quelle."""
    if results.is_empty() or "population" not in results.columns:
        return results
    rank = pl.col("population").replace_strict(
        {p: i for i, p in enumerate(POPULATION_PRIORITY)}, return_dtype=pl.Int8
    )
    return results.filter(rank == rank.min().over("poll_id", "round", "hypothesis_id"))


# ---------------------------------------------------------------------------
# Suspects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfficialRace:
    """Ce que le registre dit d'une élection passée : tours, candidats, finalistes."""

    round1: date
    round2: date | None
    candidates: frozenset[str]
    finalists: frozenset[str]


def official_races(registry) -> dict[int, OfficialRace]:
    out = {}
    for year, e in registry.elections.items():
        if not e.candidates:
            continue
        r1 = e.round_(1)
        r2 = e.round_(2) if len(e.rounds) == 2 else None
        finalists = frozenset(r.candidate_id for r in r2.results) if r2 is not None else frozenset()
        out[year] = OfficialRace(
            r1.date, r2.date if r2 is not None else None,
            frozenset(c.candidate_id for c in e.candidates), finalists,
        )
    return out


def suspect_reasons(
    round_: int,
    shares: dict[str, tuple[float, str]],
    start: date | None,
    end: date | None,
    race: OfficialRace | None,
    known_candidates: set[str] | frozenset[str],
) -> list[str]:
    """Raisons de suspicion d'une hypothèse (vide si saine).

    - `sum_out_of_range` : parts en base exprimés (candidats et `other`) hors de
      `SHARE_SUM_BOUNDS` ;
    - `incomplete_hypothesis` : moins de `MIN_FIRST_ROUND_CANDIDATES` candidats au premier tour,
      autre chose que deux au second ; ou, liste officielle connue (terrain commencé moins de
      `OFFICIAL_LIST_LEAD_DAYS` jours avant le premier tour), au moins
      `MAX_MISSING_OFFICIAL` candidats officiels absents ;
    - `undeclared_candidate` : candidat absent du registre ; ou, liste officielle connue, un
      candidat hors liste au premier tour ; ou, terrain après le premier tour, un duel qui
      n'est pas celui des deux qualifiés.
    """
    reasons: list[str] = []
    expressed = [s for c, (s, base) in shares.items() if base == "expressed"]
    total = sum(expressed)
    if not SHARE_SUM_BOUNDS[0] <= total <= SHARE_SUM_BOUNDS[1]:
        reasons.append("sum_out_of_range")
    cands = {c for c in shares if c not in RESERVED}
    incomplete = (round_ == 1 and len(cands) < MIN_FIRST_ROUND_CANDIDATES) or (
        round_ == 2 and len(cands) != 2
    )
    undeclared = bool(cands - set(known_candidates))
    if race is not None and start is not None and end is not None:
        list_known = start >= race.round1 - timedelta(days=OFFICIAL_LIST_LEAD_DAYS)
        if round_ == 1 and list_known and end < race.round1:
            incomplete |= len(race.candidates - cands) >= MAX_MISSING_OFFICIAL
            undeclared |= bool(cands - race.candidates)
        if round_ == 2 and race.finalists and start > race.round1:
            undeclared |= cands != race.finalists
    if incomplete:
        reasons.append("incomplete_hypothesis")
    if undeclared:
        reasons.append("undeclared_candidate")
    return reasons


# ---------------------------------------------------------------------------
# Rolling
# ---------------------------------------------------------------------------


def is_rolling_label(text: str | None) -> bool:
    return bool(text) and bool(ROLLING_LABEL_RE.search(unidecode(text)))


def _union_find(parent: dict[str, str]):
    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    return find


def rolling_by_regularity(polls: pl.DataFrame, labeled: set[str] | frozenset[str] = frozenset()) -> set[str]:
    """Vagues d'un sondage glissant : au moins `ROLLING_MIN_CHAIN` sondages d'un même institut
    (même élection) reliés de proche en proche par des terrains qui se chevauchent, la fin de
    l'un précédant celle de l'autre de 1 à `ROLLING_MAX_STEP_DAYS` jours, aux effectifs
    proches (rapport au plus `ROLLING_MAX_SAMPLE_RATIO`). Deux vagues ordinaires d'un institut
    ne se chevauchent pas ; les vagues d'un rolling, renouvelé par tiers, si (Ifop-Fiducial
    2017, publié tous les deux à quatre jours ; tracker quotidien OpinionWay 2022).

    Un sondage à libellé rolling (`labeled`) ne se relie qu'à d'autres sondages à libellé : une
    vague ordinaire (Ipsos-Cevipof, baromètre Ifop) dont le terrain chevauche un rolling
    étiqueté n'en devient pas une.

    Colonnes : poll_id, election, pollster, fieldwork_start, fieldwork_end, sample_size.
    """
    out: set[str] = set()
    for _, g in polls.group_by("election", "pollster"):
        rows = sorted(
            g.select("poll_id", "fieldwork_start", "fieldwork_end", "sample_size").rows(),
            key=lambda r: r[2],
        )
        parent = {r[0]: r[0] for r in rows}
        find = _union_find(parent)
        for i, (pid, start, end, n) in enumerate(rows):
            for qid, qstart, qend, qn in rows[i + 1:]:
                step = (qend - end).days
                if step > ROLLING_MAX_STEP_DAYS:
                    break
                close_n = not n or not qn or max(n, qn) <= ROLLING_MAX_SAMPLE_RATIO * min(n, qn)
                same_kind = (pid in labeled) == (qid in labeled)
                if step >= 1 and qstart <= end and close_n and same_kind:
                    parent[find(pid)] = find(qid)
        comps: dict[str, list[str]] = {}
        for pid in parent:
            comps.setdefault(find(pid), []).append(pid)
        for members in comps.values():
            if len(members) >= ROLLING_MIN_CHAIN:
                out.update(members)
    return out


# ---------------------------------------------------------------------------
# Divergences connues (docs/03-donnees.md) et règle qui les résout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownDivergence:
    name: str
    description: str
    pollster: str
    fieldwork_end: date  # fin de terrain retenue (source prioritaire)
    rule: str  # règle du journal attendue
    field: str  # champ du journal attendu


KNOWN_DIVERGENCES: tuple[KnownDivergence, ...] = (
    KnownDivergence(
        "elabe_2026_03_column_shift",
        "Elabe 25-27 mars 2026 : colonne décalée sur Wikipédia (Glucksmann et non Hollande à "
        "8,5) ; l'hypothèse Wikipédia est rattachée à celle de la Commission par `label_swap`.",
        "elabe", date(2026, 3, 27), "label_swap", "candidate_id",
    ),
    KnownDivergence(
        "ifop_2025_04_duel_inverted",
        "Ifop pour Hexagone, avril 2025 : duel Philippe–Le Pen inversé chez MieuxVoter (et "
        "rattaché à la première vague) ; la Commission l'emporte (vague sœur, précédence).",
        "ifop", date(2025, 4, 30), "sibling_wave", "share",
    ),
    KnownDivergence(
        "ifop_2025_04_two_waves",
        "Ifop pour Hexagone : une notice pour deux vagues (11-18 et 29-30 avril 2025) ; les "
        "hypothèses de la seconde vague forment un sondage à part.",
        "ifop", date(2025, 4, 30), "multi_wave_split", "hypotheses",
    ),
    KnownDivergence(
        "ifop_2025_03_dates",
        "Ifop mars 2025 : dates fausses sur Wikipédia (21-26 mars au lieu de 26-27) ; mêmes "
        "résultats que la notice, les dates de la Commission l'emportent.",
        "ifop", date(2025, 3, 27), "same_results", "fieldwork_start",
    ),
)


def check_known_divergences(log: list[dict]) -> dict[str, bool]:
    """Pour chaque divergence connue, vrai si le journal contient l'entrée attendue."""
    out = {}
    for d in KNOWN_DIVERGENCES:
        out[d.name] = any(
            e["rule"] == d.rule
            and e["field"] == d.field
            and e["key"]["pollster"] == d.pollster
            and e["key"]["fieldwork_end"] == d.fieldwork_end.isoformat()
            for e in log
        )
    return out
