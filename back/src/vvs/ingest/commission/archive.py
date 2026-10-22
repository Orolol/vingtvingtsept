"""Notices archivées (2017, 2022) et lecture « deux populations » des notices.

`parse_archive(election)` lit les notices téléchargées par `listing.ingest_archive` dans
`data/raw/commission/<élection>/` et écrit `polls_archive.parquet` et
`poll_results_archive.parquet` (schémas de `polls.parquet` et `poll_results.parquet`,
`source` = « commission », plus `file_id` et `matched_poll_id` dans `polls_archive`, le
sondage de `polls.parquet` décrivant la même collecte). Ces tables ne sont pas fusionnées
dans `polls.parquet` : la curation s'en chargera. Sorties communes aux trois élections :
`poll_turnout_intent.parquet` (part de certains d'aller voter dans l'échantillon) et
`poll_archive_coverage.parquet` (taux de lecture par institut). `parse_archive(2027)` relit les notices 2027 avec les mêmes règles et
écrit `polls_commission_dual.parquet` / `poll_results_commission_dual.parquet`, sans toucher
aux sorties de `vvs parse commission`.

Ce qui change par rapport au parseur 2027 (`parse`, `notice`), qu'on réutilise tel quel pour
le texte, les métadonnées, les blocs et les formats de colonnes :

- **candidats** : ceux du registre (toutes élections) s'ajoutent aux alias de la Commission
  le temps de la lecture (`aliases.archive_candidates`) ; Emmanuel Macron est candidat en
  2017 et 2022, la règle « bloc avec Macron = rappel 2022 » ne s'applique donc pas ; un bloc
  est un rappel quand les lignes au-dessus citent la présidentielle précédente (2012, 2017 ou
  2022) ;
- **deux populations** : une hypothèse est identifiée par (tour, candidats, population) et
  non plus (tour, candidats) ; le parseur 2027 garde la première population vue et écarte
  l'autre comme doublon ;
- **tableaux OpinionWay à deux bases** : « Publié » en première colonne, puis k colonnes pour
  l'ensemble des inscrits (brut et redressements) et k colonnes pour les certains d'aller
  voter, dans le même ordre (k = 4 en 2017 et 2027, 5 en 2022). La colonne publiée est l'un
  des redressements des certains ; on la repère à la valeur (somme des écarts minimale à la
  première colonne) et on prend la colonne des inscrits au même rang : même redressement,
  seule la base change. Le brut de chaque base est sa première colonne ;
- **ligatures** : les notices PrésiTrack 2017 d'OpinionWay rendent « ti » par un glyphe qui
  change d'une police à l'autre (« élec,on », « inten>ons », « échanCllon », « élecAon »,
  « No#ce ») ; on repère ces glyphes dans quelques mots-témoins et on les répare avant toute
  lecture ;
- **part de certains d'aller voter dans l'échantillon** (`turnout_intent`) : lue quand la
  notice la publie, voir `find_turnout_intent`.

Qui publie quoi (relevé du 2026-10-22 sur les notices archivées, détaillé dans
`runs/2026-10-22-turnout-gap/report.md`) : OpinionWay publie les deux bases dans le même
tableau (PrésiTrack 2017, baromètres et rolling 2022, 2027) ; Toluna Harris Interactive publie
les inscrits en 2022 et les certains dans ses tableaux détaillés depuis septembre 2026, avec
la part de certains d'aller voter au premier et au second tour ; Ifop publie les inscrits
jusqu'à fin mars 2017 puis les certains ; Elabe, Ipsos, BVA (graphiques non lus) et Kantar
publient les certains (Ipsos et Kantar donnent la part de certains exprimant un choix).
Règles de population propres aux archives : `notice_population` et `block_population`.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import polars as pl
from pdfminer.psparser import PSException

from vvs import paths, schemas
from vvs.ingest.commission import notice as nt
from vvs.ingest.commission.aliases import archive_candidates, candidate_id, norm_label
from vvs.ingest.commission.listing import RAW_DIR, archive_dir, load_index
from vvs.ingest.commission.parse import EXCLUDED_NOTICES as PARSE_EXCLUDED
from vvs.ingest.commission.parse import NoticeMeta, NoticeParse, notice_metadata

ARCHIVE_ELECTIONS = (2017, 2022)
POLLS_ARCHIVE_PATH = paths.CURATED / "polls_archive.parquet"
RESULTS_ARCHIVE_PATH = paths.CURATED / "poll_results_archive.parquet"
POLLS_DUAL_PATH = paths.CURATED / "polls_commission_dual.parquet"
RESULTS_DUAL_PATH = paths.CURATED / "poll_results_commission_dual.parquet"
TURNOUT_INTENT_PATH = paths.CURATED / "poll_turnout_intent.parquet"
# Taux de réussite par élection et institut (`ArchiveReport.by_pollster`).
COVERAGE_PATH = paths.CURATED / "poll_archive_coverage.parquet"
# Fenêtre de terrain retenue : de 18 mois avant le premier tour à la veille du second tour.
WINDOW_DAYS = 548

# Notices écartées par règle explicite : sous-populations, autres scrutins.
ARCHIVE_EXCLUDED: dict[tuple[int, int], str] = {
    (2017, 429): "sous-population : Français de l'étranger (BVA pour lepetitjournal.com)",
}

# Abréviations d'institut des titres 2016-2022 (« HI », « OW »), comme `ingest.certainty`.
ARCHIVE_POLLSTERS = {
    "hi": "Harris Interactive",
    "phi": "Harris Interactive",
    "ow": "OpinionWay",
    "op": "OpinionWay",
    "ipsoso": "Ipsos",
    "iposos": "Ipsos",
    "presitrack": "OpinionWay",
    "sofres": "Kantar",
    "tns": "Kantar",
    "ifopo": "Ifop",
    "roll": "Ifop",  # « 8293 Roll 28 » : rolling Ifop-Fiducial 2017 (Paris Match)
}

# Libellés de candidats des notices archivées absents du registre sous cette forme.
ARCHIVE_CANDIDATE_LABELS: dict[str, tuple[str, ...]] = {
    "lassalle_jean": ("Jean Lasalle",),
}
# Libellés non candidats propres aux notices archivées (rappels de vote, non-réponse),
# comparés par préfixe comme `aliases.NON_CANDIDATE_PREFIXES`.
ARCHIVE_NON_CANDIDATE_PREFIXES = (
    "vous avez vote",
    "vousavez vote",
    "vous vous etes abstenu",
    "vous n etiez pas inscrit",
    "n expriment pas",
    "ne savent pas",
    "nsp",
    "blancs",
    "abstentionnistes",
)

TURNOUT_INTENT_SCHEMA = pl.Schema(
    {
        "poll_id": pl.Utf8,
        "election": pl.Int32,
        "pollster": pl.Utf8,
        "fieldwork_end": pl.Date,
        "round": pl.Int8,
        "certain_share": pl.Float64,  # part de certains d'aller voter parmi les inscrits, 0-100
        "kind": pl.Utf8,  # certain, certain_expressed (certains ET exprimant une intention)
        "weighted": pl.Boolean,  # part redressée (sinon effectifs bruts)
        "n_registered": pl.Int32,
        "n_certain": pl.Int32,
        "detail": pl.Utf8,
    }
)


# ------------------------------------------------------------------------- texte

# Mots-témoins de la ligature « ti » perdue : le glyphe de substitution change d'une police à
# l'autre (« élec,on », « élec>on », « élec=on », « élec@on », « élecAon », « élecBon »).
_LIGATURE_ANCHOR = re.compile(
    r"(?:[ée]lec|inten|ques|popula|aggloméra|interroga|repr[ée]senta|No)([^\sti])(?:on|ons|f|ve|ce)\b"
    r"|présiden([^\sti])elle|échan([^\sti])llon"
)


def lost_ligature_glyphs(text: str) -> set[str]:
    """Glyphes qui remplacent « ti » dans la notice (vide pour une notice saine) : ceux vus
    au moins trois fois dans les mots-témoins."""
    counts = Counter(g for m in _LIGATURE_ANCHOR.finditer(text) for g in m.groups() if g)
    return {g for g, k in counts.items() if k >= 3}


def fix_ligatures(text: str, glyphs: set[str]) -> str:
    """« élec,on » -> « élection » : chaque glyphe de `glyphs` entre deux lettres minuscules."""
    for g in glyphs:
        text = re.sub(rf"(?<=[a-zéèàâêîôûç]){re.escape(g)}(?=[a-zéèàâêîôûç])", "ti", text)
    return text


_EVOLUTION = re.compile(r"^(?:[+\-−–]\s?\d{1,2}(?:[,.]\d+)?(?:pts?|points?)?|=)$")


def strip_evolutions(page: str) -> str:
    """Retire les évolutions signées en fin de ligne (« Emmanuel Macron 27,5% +/-2,9 points -1 »,
    Ipsos 2022) ou seules sur leur ligne (Kantar 2017) : une valeur signée ou « = » n'est
    jamais une intention de vote, et le parseur de lignes s'arrête dessus."""
    out = []
    for line in page.splitlines():
        tokens = line.split()
        if tokens and all(_EVOLUTION.match(t) for t in tokens):
            continue  # évolution seule sur sa ligne (Kantar 2017 : « 12 » / « +2 »)
        stripped = False
        while len(tokens) >= 3 and _EVOLUTION.match(tokens[-1]):
            rest = nt._MARGIN.sub(" ", " ".join(tokens[:-1])).split()
            if not rest or not nt._VALUE_TOKEN.match(rest[-1]):
                break
            tokens.pop()
            stripped = True
        out.append(" ".join(tokens) if stripped else line)
    return "\n".join(out)


def read_text(path: Path) -> nt.NoticeText:
    text = nt.extract_text(path)
    glyphs = lost_ligature_glyphs("\n".join(text.pages))
    if glyphs:
        text = nt.NoticeText(
            [fix_ligatures(p, glyphs) for p in text.pages],
            [fix_ligatures(c, glyphs) for c in text.columns],
        )
    return nt.NoticeText([strip_evolutions(p) for p in text.pages], text.columns)


def registry_labels() -> dict[str, list[str]]:
    """{id: libellés} : nom complet et prénom + nom de chaque candidat du registre."""
    from vvs.registry import default_registry

    reg = default_registry()
    out = {cid: [c.name, f"{c.first_name} {c.last_name}"] for cid, c in reg.candidates.items()}
    for cid, labels in ARCHIVE_CANDIDATE_LABELS.items():
        out.setdefault(cid, []).extend(labels)
    return out


def entry_with_pollster(entry: dict) -> dict:
    """Préfixe le nom complet de l'institut quand le titre n'en a qu'une abréviation."""
    from vvs.ingest.commission.aliases import find_pollster_in_title

    if find_pollster_in_title(entry["title"]):
        return entry
    for tok in norm_label(entry["title"]).split():
        if tok in ARCHIVE_POLLSTERS:
            return entry | {"title": f"{ARCHIVE_POLLSTERS[tok]} {entry['title']}"}
    return entry


# Titre de tableau sans la question (Kantar 2017 : « Intentions de vote pour le premier tour
# de l'élection présidentielle / Hypothèse 1 : François Fillon »).
_IV_TITLE = re.compile(
    r"intentions? de vote (pour le|au) (premier|1er|second|2nd) tour de l election presidentielle"
    r"\W+(?:\w+ )?(hypothese|en|base)\b"
)


_LOOSE_SPAN = re.compile(r"\b(?:du|les|le) ((?:1 ?er|\d{1,2})\b[a-z0-9 ]{0,60}?)\b(20\d\d)\b")
_GLUED_PREP = re.compile(r"(?<=[a-z])(du|les)(?= ?\d)")


def find_fieldwork_loose(
    text: str, window: tuple[date, date] | None, deposit_year: int | None = None
) -> tuple[date, date, bool] | None:
    """Dates de terrain que `notice.find_fieldwork` ne lit pas, après un indice de terrain :

    - liste de jours : « les 5, 6 et 7 février 2017 », « les 31 mars, 1er avril et 2 avril
      2017 » (PrésiTrack 2017) ;
    - préposition collée : « interrogésdu 30janvier au 2février 2022 », « réaliséesdu02au03 »
      (OpinionWay 2022).

    Premier jour et dernier jour de la liste, chacun avec le mois qui le suit. Si la fin tombe
    hors de la fenêtre de dépôt mais y entre avec l'année du dépôt (« du 02 au 03 janvier
    2021 » dans une notice de janvier 2022), on retient l'année du dépôt : coquille d'année,
    règle explicite, comptée dans le résumé (`ArchiveParse.fieldwork_year_fixed`).

    Rend (début, fin, année corrigée).
    """
    t = _GLUED_PREP.sub(r" \1", nt._n_dates(text))
    for m in _LOOSE_SPAN.finditer(t):
        if not nt._FIELDWORK_CUE.search(t[max(0, m.start() - 80) : m.start()]):
            continue
        tokens = m.group(1).replace("1 er", "1er").split()
        days: list[tuple[int, int | None]] = []
        for tok in tokens:
            if tok == "1er" or tok.isdigit():
                days.append((1 if tok == "1er" else int(tok), None))
            elif tok in nt.MONTHS_FR:
                # le mois s'applique aux jours qui le précèdent et n'en ont pas encore
                days = [(d, mo if mo is not None else nt.MONTHS_FR[tok]) for d, mo in days]
            elif tok not in ("et", "au"):
                break
        if not days or days[-1][1] is None or days[0][1] is None:
            continue
        year = int(m.group(2))
        for y in (year, deposit_year):
            if y is None:
                continue
            try:
                end = date(y, days[-1][1], days[-1][0])
                start = date(y if days[0][1] <= days[-1][1] else y - 1, days[0][1], days[0][0])
            except ValueError:
                break
            if 0 <= (end - start).days <= 31 and (window is None or window[0] <= end <= window[1]):
                return start, end, y != year
    return None


def is_vote_intention(text: str) -> bool:
    return nt.is_vote_intention(text) or bool(_IV_TITLE.search(nt._n(text)))


# ------------------------------------------------------------------- classement


def _recall_re(election: int) -> re.Pattern:
    """Rappel d'une présidentielle antérieure (la précédente surtout : reconstitution du vote ;
    les plus anciennes dans les rétrospectives Ifop de 2022)."""
    prev = "|".join(str(y) for y in range(2002, election, 5))
    return re.compile(
        rf"\b(presidentielles? (de )?({prev})|({prev}) (1er|premier|2nd|second) tour"
        rf"|(1er|premier|2nd|second|2d) tour( de)?( l election)?( la)? presidentielle( de)? ({prev})"
        rf"|vote (en|a|au) .{{0,80}}({prev}))\b"
    )


# Contexte d'une autre question que l'intention de vote, repris du parseur 2027 sans la
# mention de 2022 (remplacée par `_recall_re`).
_SKIP_CONTEXT = re.compile(
    r"reconstitution|evolutions?\b|report(s)? de voix|transfert|pour memoire|souvenir de vote"
    r"|chances d etre elu|meilleur president|potentiel electoral|pourriez voter"
    r"|pourrait voter|pourriez vous voter|bon candidat|mieux place|souhait\w* (vous )?(que|qu|voir)"
    r"|recapitulatif|plus dangereux|voir gagner|prefer|plus proche|incarne"
    r"|ferait le meilleur|redressement politique|primaire (de la |de |a )?(droite|gauche|du|des|ouverte|citoyenne)"
    r"|sur(e|s|es)? de (leur|votre|son) choix|choix definitif"
    r"|qui va gagner|va gagner l election|qui l emportera|pronostic"
    r"|avez vous vote|vote anterieur"
)
# Seconds tours Kantar (2017, 2022) en camembert : les deux valeurs précèdent ou suivent les
# noms sans ordre fixe (« Emmanuel Macron / 39 / 61 / … / Marine Le Pen ») ; on ne les attribue
# pas. Écart constaté avec Wikipédia : 30 points (parts inversées).
UNRELIABLE_ROUND2 = frozenset({"kantar"})
_DUAL_HEADER = re.compile(r"certain")
_DUAL_OTHER = re.compile(r"ensemble|inscrit")


def dual_columns(block: nt.Block) -> tuple[int, int, int, int] | None:
    """Tableau OpinionWay à deux bases -> (certains, inscrits, brut certains, brut inscrits).

    1 + 2k valeurs par ligne (k = 4 ou 5), en-tête citant les certains d'aller voter et
    l'ensemble des inscrits ou des répondants."""
    counts = sorted(len(r.values) for _, r in block.candidate_rows)
    n = counts[len(counts) // 2]
    if n not in (9, 11):
        return None
    head = nt._n(" ".join(block.context[-14:]))
    if not (_DUAL_HEADER.search(head) and _DUAL_OTHER.search(head)):
        return None
    k = (n - 1) // 2
    rows = [r.values for _, r in block.candidate_rows if len(r.values) == n]

    def gap(j: int) -> float:
        return sum(abs(v[j] - v[0]) for v in rows if v[j] is not None and v[0] is not None)

    j = min(range(k + 2, 2 * k + 1), key=gap)  # un redressement des certains, pas le brut
    return j, j - k, k + 1, 1


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
    # Toluna Harris Interactive 2027 : base en pied des tableaux détaillés, alors que le
    # récapitulatif dit « Aux inscrits sur les listes électorales » pour les mêmes chiffres.
    r"|base inscrits sur les listes electorales certains d aller voter et ayant exprime"
)


def notice_population(pages: list[str], head_population: str) -> str:
    """Population par défaut des tableaux à une base : `certain_voters` si la notice dit que
    les résultats publiés portent sur les certains d'aller voter, sinon celle de l'en-tête."""
    if _REFERENCE_CERTAIN.search(nt._n("\n".join(pages))):
        return "certain_voters"
    return head_population


def block_population(block: nt.Block, default_population: str) -> str:
    """Ligne « Base : … » du tableau, sauf « inscrits » quand la notice déclare publier les
    certains d'aller voter (la ligne décrit alors l'échantillon, pas la colonne publiée)."""
    pop = nt._block_population(block)
    if pop is None or (pop == "registered" and default_population == "certain_voters"):
        return default_population
    return pop


@dataclass
class DualResults:
    hypotheses: list[nt.Hypothesis] = field(default_factory=list)
    outcomes: list[nt.BlockOutcome] = field(default_factory=list)
    publish_gap: list[float] = field(default_factory=list)  # écart publié / colonne retenue


def _results(cands, share_col: int, raw_col: int | None) -> list[tuple[str, float, float | None]]:
    out = []
    for cid, row in cands:
        share = row.values[share_col] if share_col < len(row.values) else None
        raw = row.values[raw_col] if raw_col is not None and raw_col < len(row.values) else None
        if share is not None:
            out.append((cid, share, raw))
    return out


def classify_blocks(
    blocks: list[nt.Block], pollster: str, default_population: str, election: int
) -> DualResults:
    """Règles de `notice.classify_blocks`, adaptées aux archives et aux deux populations."""
    out = DualResults()
    seen: dict[tuple, int] = {}
    recall = _recall_re(election)
    for block in blocks:
        cands = block.candidate_rows
        if len(cands) < 2:
            out.outcomes.append(nt.BlockOutcome(block.page, "not_candidates"))
            continue
        near = nt._n(" ".join(block.context[-10:]))
        if _SKIP_CONTEXT.search(near) or recall.search(near):
            out.outcomes.append(nt.BlockOutcome(block.page, "recall_or_other_question"))
            continue
        unknown = [
            r.label
            for r in block.rows
            if not candidate_id(r.label)
            and not nt.is_non_candidate(r.label)
            and not norm_label(r.label).startswith(ARCHIVE_NON_CANDIDATE_PREFIXES)
        ]
        if not nt._IV_CONTEXT.search(nt._n(" ".join(block.context[-14:]))):
            out.outcomes.append(nt.BlockOutcome(block.page, "other_question", str(unknown[:3])))
            continue
        if unknown:
            # Archives : le bloc est écarté et journalisé, pas la notice entière (graphiques
            # de pronostic ou de souhait de victoire mis en page comme des tableaux, candidat
            # non nommé « [Candidat de la droite] ») ; on ne devine jamais le libellé.
            out.outcomes.append(nt.BlockOutcome(block.page, "unknown_label", str(unknown[:4])))
            continue
        dual = dual_columns(block) if pollster == "opinionway" else None
        if dual:
            cert, reg, raw_cert, raw_reg = dual
            versions = [
                ("certain_voters", _results(cands, cert, raw_cert)),
                ("registered", _results(cands, reg, raw_reg)),
            ]
            published = _results(cands, 0, None)
            gaps = {c: s for c, s, _ in versions[0][1]}
            out.publish_gap.append(
                max((abs(s - gaps[c]) for c, s, _ in published if c in gaps), default=0.0)
            )
        else:
            spec = nt._column_spec(block, pollster)
            if spec is None:
                out.outcomes.append(
                    nt.BlockOutcome(
                        block.page, "unknown_format", f"{len(cands[0][1].values)} valeurs"
                    )
                )
                continue
            population = block_population(block, default_population)
            versions = [(population, _results(cands, *spec))]
        union = schemas.hypothesis_id([c for c, _ in cands])
        for population, results in versions:
            if len({c for c, _, _ in results}) != len(results):
                out.outcomes.append(nt.BlockOutcome(block.page, "duplicate_candidate"))
                continue
            total = sum(s for _, s, _ in results)
            if not 94 <= total <= 106:
                out.outcomes.append(nt.BlockOutcome(block.page, "bad_sum", f"somme {total:g}"))
                continue
            rnd = 2 if len(results) == 2 else 1
            if rnd == 2 and pollster in UNRELIABLE_ROUND2:
                out.outcomes.append(nt.BlockOutcome(block.page, "unreliable_layout"))
                continue
            if rnd == 1 and len(results) < nt.MIN_FIRST_ROUND_CANDIDATES:
                out.outcomes.append(
                    nt.BlockOutcome(block.page, "too_few_candidates", str(len(results)))
                )
                continue
            ctx6 = nt._n(" ".join(block.context[-6:]))
            if (
                rnd == 1
                and nt._ROUND2.search(ctx6)
                and not re.search(r"\b(1er|premier) tour\b", ctx6)
            ):
                out.outcomes.append(nt.BlockOutcome(block.page, "round_mismatch"))
                continue
            key = (rnd, tuple(sorted(c for c, _, _ in results)), population)
            # `label` : configuration testée (tous les candidats du tableau), commune aux deux
            # bases même quand un petit candidat n'a de valeur que dans l'une.
            hyp = nt.Hypothesis(rnd, population, results, block.page, label=union)
            if key in seen:
                prev = out.hypotheses[seen[key]]
                has_raw = any(r is not None for _, _, r in results)
                if has_raw and not any(r is not None for _, _, r in prev.results):
                    out.hypotheses[seen[key]] = hyp
                out.outcomes.append(nt.BlockOutcome(block.page, "duplicate"))
                continue
            seen[key] = len(out.hypotheses)
            out.hypotheses.append(hyp)
            out.outcomes.append(nt.BlockOutcome(block.page, "accepted"))
    return out


# ------------------------------------------------------- part de certains d'aller voter

# OpinionWay : en-tête des tableaux à deux bases, texte normalisé (parenthèses retirées) :
# « ensemble des repondants 1487 tout a fait certains d aller voter personnes 928 personnes »,
# 2022 : « ensemble des personnes inscrites sur les listes electorales certains d aller voter
# 1562 personnes notes 9 et 10 1177 personnes ».
_OW_HEAD = re.compile(r"ensemble ?des (?:repondants|personnes inscrites|inscrits)")
# Harris : « Tout à fait certain(e) d'aller voter 69 70 » (bruts, redressés) ou « 73 ».
_HARRIS_SHARE = re.compile(r"^tout a fait certain e d aller voter (\d{1,3})(?: (\d{1,3}))?$")
# Ipsos : « 1651 personnes inscrites sur les listes électorales (dont 870 certaines d'aller
# voter et exprimant une intention de vote au premier tour… ».
_IPSOS_COUNTS = re.compile(
    r"(\d{3,5}) personnes inscrites sur les listes electorales dont (\d{3,5}) certaine?s? d aller voter et exprimant"
)
# Kantar : « 33% des répondants ne sont pas certains d'aller voter ou n'ont pas exprimé de choix ».
_KANTAR_SHARE = re.compile(
    r"(\d{1,2}) des (?:repondants|personnes interrogees) ne sont pas (?:tout a fait )?certain\w* d aller voter"
    r" ou n ont pas exprime"
)


@dataclass
class TurnoutIntent:
    round: int
    certain_share: float
    kind: str
    weighted: bool
    n_registered: int | None = None
    n_certain: int | None = None
    detail: str = ""


def _row_values(line: str) -> list[float | None] | None:
    row = nt.parse_row(line, 0)
    return row.values if row else None


def find_turnout_intent(pages: list[str], pollster: str) -> list[TurnoutIntent]:
    """Part de certains d'aller voter parmi les inscrits interrogés, si la notice la donne.

    Formats lus (un par institut) :

    - OpinionWay : effectifs bruts des deux bases en tête des tableaux d'intentions (« Ensemble
      des répondants (1487 personnes) / Tout à fait certains d'aller voter (928 personnes) ») ;
    - Toluna Harris Interactive : « Tout à fait certain(e) d'aller voter » redressé, au premier
      puis au second tour ;
    - Ifop : « envisagez-vous d'aller voter ? Oui » × « Et diriez-vous que… Tout à fait
      certain », redressés (le produit est la part de tout à fait certains parmi les inscrits) ;
    - Ipsos : « N personnes inscrites (dont M certaines d'aller voter et exprimant une
      intention de vote) » ; Kantar : « X % des répondants ne sont pas certains d'aller voter
      ou n'ont pas exprimé de choix ». Ces deux-là mêlent participation et expression d'un
      choix (`kind` = « certain_expressed »).

    Les définitions diffèrent d'un institut à l'autre (note 10 sur 10, « tout à fait certain »,
    notes 9-10…) : la part se compare dans le temps pour un même institut, pas entre instituts.
    """
    text = nt._n("\n".join(pages))
    out: list[TurnoutIntent] = []
    if pollster == "opinionway":
        for m in _OW_HEAD.finditer(text):
            chunk = text[m.end() : m.end() + 160]
            if "certain" not in chunk:
                continue
            # l'ordre des deux effectifs varie avec la mise en page : les certains sont le plus petit
            nums = [int(x) for x in re.findall(r"\b(\d{3,5})\b", chunk)][:2]
            if len(nums) == 2 and min(nums) > 0 and nums[0] != nums[1]:
                n_reg, n_cert = max(nums), min(nums)
                out.append(
                    TurnoutIntent(
                        1, 100 * n_cert / n_reg, "certain", False, n_reg, n_cert, chunk[:120]
                    )
                )
                break
    elif pollster == "harris":
        rnd = 1
        for page in pages:
            for line in page.splitlines():
                t = nt._n(line)
                if "certitude" in t and re.search(r"second ?tour|2nd tour", t):
                    rnd = 2
                elif "certitude" in t and re.search(r"premier ?tour|1er tour", t):
                    rnd = 1
                m = _HARRIS_SHARE.match(t)
                if m and not any(o.round == rnd for o in out):
                    share = float(m.group(2) or m.group(1))
                    out.append(TurnoutIntent(rnd, share, "certain", True, detail=line[:120]))
    elif pollster == "ifop":
        yes = certain = None
        for page in pages:
            lines = page.splitlines()
            for k, line in enumerate(lines):
                label = nt._n(line)
                ctx = nt._n(" ".join(lines[max(0, k - 8) : k]))
                vals = _row_values(line)
                if not vals or vals[-1] is None:
                    continue
                if (
                    yes is None
                    and label.startswith("oui ")
                    and "envisagez vous d aller voter" in ctx
                ):
                    yes = vals[-1]
                elif (
                    certain is None
                    and re.match(r"(c est )?tout a fait certain ", label)
                    and yes is not None
                ):
                    certain = vals[-1]
        if yes is not None and certain is not None:
            out.append(
                TurnoutIntent(
                    1,
                    yes * certain / 100,
                    "certain",
                    True,
                    detail=f"oui {yes:g} x tout a fait certain {certain:g}",
                )
            )
    elif pollster == "ipsos":
        m = _IPSOS_COUNTS.search(text)
        if m:
            n_reg, n_cert = int(m.group(1)), int(m.group(2))
            if 0 < n_cert < n_reg:
                out.append(
                    TurnoutIntent(
                        1,
                        100 * n_cert / n_reg,
                        "certain_expressed",
                        False,
                        n_reg,
                        n_cert,
                        m.group(0)[:120],
                    )
                )
    elif pollster == "kantar":
        m = _KANTAR_SHARE.search(text)
        if m:
            out.append(
                TurnoutIntent(
                    1, 100 - float(m.group(1)), "certain_expressed", True, detail=m.group(0)[:120]
                )
            )
    return out


# ----------------------------------------------------------------------- notices


@dataclass
class ArchiveParse(NoticeParse):
    election: int = 0
    turnout: list[TurnoutIntent] = field(default_factory=list)
    publish_gap: list[float] = field(default_factory=list)
    fieldwork_year_fixed: bool = False


def parse_notice(
    path: Path, entry: dict, election: int, labels: dict | None = None
) -> ArchiveParse:
    """Parse une notice (archive ou 2027) ; erreurs capturées dans le résultat."""
    entry = entry_with_pollster(entry)
    res = ArchiveParse(file_id=entry["file_id"], entry=entry, election=election)
    try:
        text = read_text(path)
    except (OSError, ValueError, PSException) as exc:  # PDF illisible : signalé, pas avalé
        res.error = f"pdf : {exc}"
        return res
    full = "\n".join(text.pages)
    res.is_iv = is_vote_intention(full)
    if not res.is_iv:
        return res
    try:
        res.meta = notice_metadata(text, entry)
    except KeyError as exc:
        res.error = str(exc)
        return res
    if res.meta.fieldwork_end is None:
        from vvs.ingest.commission.parse import _month_window

        window = _month_window(entry["month"])
        found = find_fieldwork_loose(
            "\n".join(text.pages[:6]), window, window[1].year if window else None
        )
        if found:
            res.meta.fieldwork_start, res.meta.fieldwork_end, res.fieldwork_year_fixed = found
    with archive_candidates(labels or {}):
        try:
            blocks = nt.find_blocks(text.pages) + nt.find_transposed_blocks(text.pages)
            for page_no, page in enumerate(text.pages, start=1):
                halves = nt.split_two_column_page(page)
                if halves:
                    for b in nt.find_blocks(list(halves)):
                        b.page = page_no
                        blocks.append(b)
            default_pop = notice_population(text.pages, res.meta.population)
            found = classify_blocks(blocks, res.meta.pollster, default_pop, election)
        except nt.UnknownLabelError as exc:
            res.error, res.unknown_label = str(exc), True
            return res
    res.hypotheses, res.outcomes, res.publish_gap = (
        found.hypotheses,
        found.outcomes,
        found.publish_gap,
    )
    res.turnout = find_turnout_intent(text.pages, res.meta.pollster)
    return res


def _parse_one(args) -> ArchiveParse:
    path, entry, election, labels = args
    return parse_notice(Path(path), entry, election, labels)


def raw_dir_for(election: int, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir if election == 2027 else archive_dir(election, raw_dir)


def round1_date(election: int) -> date:
    from vvs.registry import default_registry

    e = default_registry().elections[election]
    return next(r.date for r in e.rounds if r.round == 1)


def round2_date(election: int) -> date:
    from vvs.registry import default_registry

    e = default_registry().elections[election]
    return next(r.date for r in e.rounds if r.round == 2)


def _excluded(n: ArchiveParse) -> str | None:
    if (n.election, n.file_id) in ARCHIVE_EXCLUDED:
        return ARCHIVE_EXCLUDED[(n.election, n.file_id)]
    if n.election == 2027 and n.file_id in PARSE_EXCLUDED:
        return PARSE_EXCLUDED[n.file_id]
    return None


def _same_poll_key(m: NoticeMeta) -> tuple:
    sponsor = norm_label(m.sponsor) if m.sponsor else None
    return (m.pollster, m.fieldwork_start, m.fieldwork_end, sponsor)


def in_window(n: ArchiveParse) -> bool:
    if n.meta is None or n.meta.fieldwork_end is None:
        return False
    if n.election == 2027:
        return True
    end = round2_date(n.election)
    return end - dt.timedelta(days=WINDOW_DAYS) <= n.meta.fieldwork_end < end


def build_tables(
    parsed: list[ArchiveParse], election: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(polls, poll_results, turnout_intent) d'une élection, schémas curated."""
    kept = sorted(
        (
            n
            for n in parsed
            if n.is_iv and n.meta and not n.error and not _excluded(n) and in_window(n)
        ),
        key=lambda n: n.file_id,
    )
    polls_rows, result_rows, turnout_rows = [], [], []
    seq: dict[tuple, int] = {}
    poll_of: dict[tuple, str] = {}
    seen_hyp: set[tuple] = set()
    for n in kept:
        m = n.meta
        if not n.hypotheses and not n.turnout:
            continue
        same = _same_poll_key(m)
        poll_id = poll_of.get(same)
        if poll_id is None:
            key = (m.pollster, m.fieldwork_end)
            seq[key] = seq.get(key, 0) + 1
            poll_id = f"{election}-{m.pollster}-{m.fieldwork_end:%Y%m%d}-c{seq[key]}"
            poll_of[same] = poll_id
            polls_rows.append(
                {
                    "poll_id": poll_id,
                    "election": election,
                    "pollster": m.pollster,
                    "sponsor": m.sponsor,
                    "fieldwork_start": m.fieldwork_start,
                    "fieldwork_end": m.fieldwork_end,
                    "publish_date": m.publish_date,
                    "sample_size": m.sample_size,
                    "mode": m.mode,
                    "source": "commission",
                    "source_url": n.entry["url"],
                    "ingested_at": datetime.fromisoformat(n.entry["ingested_at"]),
                    "sources": ["commission"],
                    "is_rolling": bool(
                        re.search(r"rolling|presitrack", norm_label(n.entry["title"]))
                    )
                    or m.pollster == "opinionway"
                    and election == 2022,
                    "file_id": n.file_id,
                    "matched_poll_id": None,
                }
            )
            for t in n.turnout:
                turnout_rows.append(
                    {
                        "poll_id": poll_id,
                        "election": election,
                        "pollster": m.pollster,
                        "fieldwork_end": m.fieldwork_end,
                        "round": t.round,
                        "certain_share": t.certain_share,
                        "kind": t.kind,
                        "weighted": t.weighted,
                        "n_registered": t.n_registered,
                        "n_certain": t.n_certain,
                        "detail": t.detail,
                    }
                )
        for h in n.hypotheses:
            hyp = h.label or schemas.hypothesis_id([c for c, _, _ in h.results])
            if (poll_id, h.round, hyp, h.population) in seen_hyp:
                continue
            seen_hyp.add((poll_id, h.round, hyp, h.population))
            total = sum(s for _, s, _ in h.results)
            for cid, share, raw in h.results:
                result_rows.append(
                    {
                        "poll_id": poll_id,
                        "round": h.round,
                        "hypothesis_id": hyp,
                        "population": h.population,
                        "candidate_id": cid,
                        "share": share,
                        "share_base": "expressed",
                        "raw_share": raw,
                        "source": "commission",
                        "suspect": not 97 <= total <= 103,
                        "suspect_reason": None if 97 <= total <= 103 else "sum_out_of_range",
                    }
                )
    polls_schema = pl.Schema(
        {**schemas.CURATED_POLLS_SCHEMA, "file_id": pl.Int32, "matched_poll_id": pl.Utf8}
    )
    polls = pl.DataFrame(polls_rows, schema=polls_schema)
    schemas.validate_polls(polls)
    results = pl.DataFrame(result_rows, schema=schemas.CURATED_POLL_RESULTS_SCHEMA)
    schemas.validate_poll_results(results.select(list(schemas.POLL_RESULTS_SCHEMA)))
    turnout = pl.DataFrame(turnout_rows, schema=TURNOUT_INTENT_SCHEMA)
    return polls, results, turnout


@dataclass
class ArchiveReport:
    election: int
    notices: list[ArchiveParse]
    polls: pl.DataFrame
    results: pl.DataFrame
    turnout: pl.DataFrame
    comparison: pl.DataFrame = field(default_factory=pl.DataFrame)

    def by_pollster(self) -> pl.DataFrame:
        """Taux de réussite par institut : notices d'intentions de vote dans la fenêtre,
        notices avec au moins un tableau lu, avec les deux populations."""
        rows = []
        for n in self.notices:
            if not n.is_iv or (n.meta and not in_window(n)) or _excluded(n):
                continue
            pops = {h.population for h in n.hypotheses}
            rows.append(
                {
                    "pollster": n.meta.pollster if n.meta else "?",
                    "parsed": bool(n.hypotheses),
                    "dual": {"registered", "certain_voters"} <= pops,
                    "population": "+".join(sorted(pops)) or "-",
                    "error": n.error is not None,
                    "turnout": bool(n.turnout),
                }
            )
        df = pl.DataFrame(rows)
        return (
            df.group_by("pollster")
            .agg(
                pl.len().alias("iv_notices"),
                pl.col("parsed").sum().alias("parsed"),
                pl.col("dual").sum().alias("dual"),
                pl.col("error").sum().alias("errors"),
                pl.col("turnout").sum().alias("turnout_intent"),
                pl.col("population").mode().first().alias("main_population"),
            )
            .with_columns((pl.col("parsed") / pl.col("iv_notices")).round(2).alias("parse_rate"))
            .sort("pollster")
        )

    def summary(self, verbose: bool = False) -> str:
        iv = [n for n in self.notices if n.is_iv]
        lines = [
            f"{self.election} : {len(self.notices)} notices lues, {len(iv)} avec intentions de vote",
            (
                f"{self.polls.height} sondages, {self.results.height} lignes de résultats,"
                f" {self.turnout.height} parts de certains d'aller voter"
            ),
            str(self.by_pollster()),
        ]
        cmp = self.comparison
        if cmp.height:
            lines.append(
                f"comparaison avec poll_results.parquet : {self.polls['matched_poll_id'].is_not_null().sum()}"
                f" sondages appariés sur {self.polls.height}, {cmp.height} hypothèses comparées"
            )
            lines.append(
                str(
                    cmp.group_by("pollster")
                    .agg(
                        pl.len().alias("hypotheses"),
                        (pl.col("max_abs_diff") <= 0.5).sum().alias("le_0_5"),
                        (pl.col("max_abs_diff") > 1.5).sum().alias("gt_1_5"),
                    )
                    .sort("pollster")
                )
            )
            for r in (
                cmp.filter(pl.col("max_abs_diff") > 1.5)
                .sort("max_abs_diff", descending=True)
                .head(40 if verbose else 8)
                .iter_rows(named=True)
            ):
                lines.append(
                    f"  écart {r['max_abs_diff']:g} {r['poll_id']} / {r['matched_poll_id']} tour {r['round']}"
                    f" {r['worst_candidate']} : commission {r['commission_share']:g},"
                    f" curated {r['curated_share']:g}"
                )
        fixed = [n.file_id for n in iv if n.fieldwork_year_fixed]
        if fixed:
            lines.append(
                f"année de terrain remplacée par celle du dépôt (coquille) : notices {fixed}"
            )
        gaps = [g for n in iv for g in n.publish_gap]
        if gaps:
            lines.append(
                f"tableaux à deux bases : {len(gaps)}, écart max publié / colonne des certains"
                f" {max(gaps):g} (médiane {sorted(gaps)[len(gaps) // 2]:g})"
            )
        if verbose:
            for n in iv:
                if n.error:
                    lines.append(f"  ÉCHEC {n.file_id} {n.entry['title']} : {n.error}")
                elif not n.hypotheses and n.meta and in_window(n):
                    rej = Counter(o.status for o in n.outcomes if o.status != "not_candidates")
                    lines.append(f"  sans tableau {n.file_id} {n.entry['title']} {dict(rej)}")
        return "\n".join(lines)

    @property
    def has_unknown_labels(self) -> bool:
        return any(n.unknown_label for n in self.notices)


def match_curated(polls: pl.DataFrame, curated: pl.DataFrame) -> pl.DataFrame:
    """Colonne `matched_poll_id` : sondage de `polls.parquet` décrivant la même collecte (même
    institut, fin de terrain à un jour près, début à trois jours près), appariement un à un des
    écarts les plus faibles aux plus forts, comme `ingest.certainty.align_all` (les vagues
    quotidiennes d'un rolling se chevauchent)."""
    cands = []
    for p in polls.iter_rows(named=True):
        sub = curated.filter(
            (pl.col("election") == p["election"]) & (pl.col("pollster") == p["pollster"])
        )
        for r in sub.iter_rows(named=True):
            d_end = abs((r["fieldwork_end"] - p["fieldwork_end"]).days)
            d_start = (
                abs((r["fieldwork_start"] - p["fieldwork_start"]).days)
                if r["fieldwork_start"] and p["fieldwork_start"]
                else 0
            )
            if d_end <= 1 and d_start <= 3:
                cands.append((d_end, d_start, p["poll_id"], r["poll_id"]))
    out: dict[str, str] = {}
    taken: set[str] = set()
    for _, _, pid, cid in sorted(cands):
        if pid not in out and cid not in taken:
            out[pid] = cid
            taken.add(cid)
    return polls.with_columns(
        pl.col("poll_id")
        .replace_strict(out, default=None, return_dtype=pl.Utf8)
        .alias("matched_poll_id")
    )


def compare_with_curated(
    polls: pl.DataFrame, results: pl.DataFrame, curated_results: pl.DataFrame
) -> pl.DataFrame:
    """Écarts avec `poll_results.parquet` (Wikipédia pour 2017 et 2022) sur les mêmes sondages
    et les mêmes configurations de candidats : une ligne par hypothèse comparée, avec l'écart
    maximal en points. La base comparée est la publiée (certains d'aller voter pour les
    tableaux OpinionWay à deux bases)."""
    if "matched_poll_id" not in polls.columns:
        return pl.DataFrame()
    pub = results.join(polls.select("poll_id", "matched_poll_id", "pollster"), on="poll_id").filter(
        pl.col("matched_poll_id").is_not_null()
    )
    # une seule base par sondage : les certains quand les deux existent
    pops = pub.group_by("poll_id").agg(pl.col("population").unique())
    pub = pub.join(pops, on="poll_id", suffix="_all").filter(
        (pl.col("population") == "certain_voters") | (pl.col("population_all").list.len() == 1)
    )
    rows = []
    cur = curated_results.filter(
        ~pl.col("candidate_id").is_in(list(schemas.RESERVED_CANDIDATE_IDS))
    )
    cur_by = {
        (k[0], k[1], k[2]): dict(zip(g["candidate_id"], g["share"], strict=True))
        for k, g in cur.group_by(["poll_id", "round", "hypothesis_id"])
    }
    for (pid, mid, pollster, rnd, hyp), g in pub.group_by(
        ["poll_id", "matched_poll_id", "pollster", "round", "hypothesis_id"]
    ):
        mine = dict(zip(g["candidate_id"], g["share"], strict=True))
        theirs = cur_by.get((mid, rnd, schemas.hypothesis_id(list(mine))))
        if theirs is None:
            continue
        diffs = {c: mine[c] - theirs[c] for c in mine if c in theirs}
        worst = max(diffs, key=lambda c: abs(diffs[c]))
        rows.append(
            {
                "poll_id": pid,
                "matched_poll_id": mid,
                "pollster": pollster,
                "round": rnd,
                "hypothesis_id": hyp,
                "max_abs_diff": abs(diffs[worst]),
                "worst_candidate": worst,
                "commission_share": mine[worst],
                "curated_share": theirs[worst],
            }
        )
    return pl.DataFrame(rows)


def _replace_election(path: Path, df: pl.DataFrame, election: int) -> None:
    """Réécrit `path` en remplaçant les lignes de `election` (les autres élections restent)."""
    if path.exists():
        old = pl.read_parquet(path)
        if "election" in old.columns:
            old = old.filter(pl.col("election") != election)
        else:  # poll_results : l'élection est dans le poll_id
            old = old.filter(~pl.col("poll_id").str.starts_with(f"{election}-"))
        df = pl.concat(
            [old.select([c for c in df.columns if c in old.columns]), df], how="diagonal_relaxed"
        )
    df.write_parquet(path)


def parse_archive(
    election: int, raw_dir: Path = RAW_DIR, workers: int = 8, write: bool = True
) -> ArchiveReport:
    """Notices d'une présidentielle -> tables « archive » (2017, 2022) ou « dual » (2027)."""
    if election not in (*ARCHIVE_ELECTIONS, 2027):
        raise ValueError(f"élection non gérée : {election}")
    d = raw_dir_for(election, raw_dir)
    labels = registry_labels() if election != 2027 else {}
    jobs = []
    for _, entry in sorted(load_index(d / "index.jsonl").items()):
        path = d / entry["file"]
        if path.suffix != ".pdf":
            continue
        jobs.append((str(path), entry, election, labels))
    with cf.ProcessPoolExecutor(workers) as ex:
        parsed = list(ex.map(_parse_one, jobs, chunksize=4))
    polls, results, turnout = build_tables(parsed, election)
    curated_path = paths.CURATED / "polls.parquet"
    if curated_path.exists():
        polls = match_curated(polls.drop("matched_poll_id"), pl.read_parquet(curated_path))
    comparison = pl.DataFrame()
    curated_results = paths.CURATED / "poll_results.parquet"
    if election != 2027 and curated_results.exists():
        comparison = compare_with_curated(polls, results, pl.read_parquet(curated_results))
    report = ArchiveReport(election, parsed, polls, results, turnout, comparison)
    if write:
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        coverage = report.by_pollster().with_columns(pl.lit(election, pl.Int32).alias("election"))
        _replace_election(COVERAGE_PATH, coverage, election)
        if election == 2027:
            polls.write_parquet(POLLS_DUAL_PATH)
            results.write_parquet(RESULTS_DUAL_PATH)
        else:
            _replace_election(POLLS_ARCHIVE_PATH, polls, election)
            _replace_election(RESULTS_ARCHIVE_PATH, results, election)
        _replace_election(TURNOUT_INTENT_PATH, turnout, election)
    return report
