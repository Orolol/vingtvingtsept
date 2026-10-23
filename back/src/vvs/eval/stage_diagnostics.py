"""Diagnostics par étape de la chaîne de prévision (docs/12 § 3), sur tirages existants.

Aucune simulation n'est relancée : tout se calcule sur les tirages sauvegardés d'un cas
(`draws_<méthode>.npz` : candidats, parts, qualifiés, vainqueur, configuration) et sur les
configurations de son `case.json`. Les définitions ci-dessous ont été fixées avant toute
lecture des chiffres ; elles suivent les conventions de `consolidation.score_chain`.

Notations : N tirages ; W le vainqueur réel ; (A, B) la paire réelle de qualifiés ; un
candidat est **présent** dans un tirage s'il appartient à la configuration tirée (sa part
y est alors strictement positive ; un absent a une part nulle, ce qui est vérifié).

Univers
-------
- Officiels : candidats du premier tour réel (liste officielle), avec leurs parts.
- Support d'un modèle : candidats présents dans au moins un tirage.
- Univers de présence d'une comparaison : officiels ∪ supports de **tous** les modèles
  comparés. Il est fixé par comparaison, avant les scores ; un modèle évalué seul utilise
  officiels ∪ son support.
- Ensemble conditionnel d'une comparaison : officiels présents dans au moins un tirage de
  **chaque** modèle comparé. Les moyennes conditionnelles (Brier de qualification
  conditionnel, CRPS et couverture conditionnels) portent sur cet ensemble ; les officiels
  jamais présents sont comptés à part (effectif et voix finales concernées), jamais
  remplacés par une valeur arbitraire.

Étapes et scores
----------------
(a) Présence. P(présent) par candidat de l'univers ; cible 1 pour un officiel, 0 sinon.
    Brier de présence = somme sur l'univers de (P − cible)², et moyenne sur le même univers.
    La somme ne change pas quand on ajoute un nom de probabilité nulle ; la moyenne n'est
    comparable qu'à univers commun. Événements de la chaîne : W présent ; A et B présents
    ensemble.
(b) Qualification conditionnelle à la présence. Pour chaque officiel présent dans au moins
    un tirage : P(qualifié | présent). Brier conditionnel = moyenne sur l'ensemble
    conditionnel de (P(qualifié | présent) − 1{finaliste})². Sur les tirages où A et B sont
    présents (sous-ensemble S) : log-score de la paire réelle, P((A, B) qualifiés | S), et
    Brier de qualification sur l'univers, calculé dans S (même forme que le score global).
    Événement de la chaîne du vainqueur : W qualifié | W présent.
(c) Victoire conditionnelle à la qualification. P(W gagne | W qualifié) pour la chaîne du
    vainqueur ; P(W gagne | (A, B) qualifiés) isole le second tour sur le bon duel
    (reports, participation, erreur du second tour).
(d) Soutien conditionnel. Pour chaque officiel, CRPS et couverture à 50 % et 90 % de sa part
    de premier tour sur les seuls tirages où il est présent ; moyenne sur l'ensemble
    conditionnel. Le score global, lui, compte zéro pour un absent : il mêle présence et
    soutien. Les deux sont publiés.

Composition (descripteurs sans score, ajoutés après une première lecture de 2017 à
180 jours) : l'étape (b) mêle l'erreur de soutien et la concurrence de candidats
finalement absents (« fantômes »). On publie, dans les tirages où W est présent, la part
moyenne des fantômes et la probabilité qu'un fantôme se qualifie, puis, quand W est
qualifié, la probabilité que son adversaire soit un fantôme.

Règles de score : log-score (propre, local) pour les événements de la chaîne, Brier
(propre) pour les vecteurs de probabilités, CRPS (propre) pour les parts ; Gneiting et
Raftery (2007). Chaque événement conditionnel est binaire ; son log-score utilise le
lissage de Jeffreys à deux issues, log((k + ½) / (n + 1)), où n est l'effectif de la
condition et k celui de l'événement. Ce lissage, comme celui du score global, est séparé
des probabilités publiées (fréquences brutes).

Décomposition
-------------
Sur les fréquences brutes, l'identité est exacte dès que chaque effectif est positif :

    log P(W gagne) = log P(W présent) + log P(W qualifié | présent)
                     + log P(W gagne | W qualifié)
    log P(paire)   = log P(A, B présents) + log P(paire | A, B présents)

et log P(paire, W gagne) = log P(paire) + log P(W gagne | paire). Les log-scores publiés
utilisent un lissage de Jeffreys sur l'univers de score (K catégories pour le vainqueur,
K(K − 1)/2 pour la paire), alors que les étapes sont lissées chacune à deux issues :
l'écart entre la somme lissée des étapes et le log-score global publié est rapporté
(`smoothing_gap_*`). Il reste petit quand les effectifs sont grands et grossit quand une
étape a un effectif nul.

Condition vide : si aucun tirage ne réalise la condition (par exemple W jamais présent,
ou paire réelle jamais présente), la probabilité conditionnelle brute est indéfinie et
publiée nulle (`None`), `evaluable` vaut faux, et le log-score lissé vaut log ½ :
c'est le score de la loi prédictive de Jeffreys sans donnée, explicite et fini. Il ne
dit rien du modèle ; l'erreur de la chaîne est alors portée par l'étape amont, ce que
la décomposition montre (log-score de présence au plancher).

Erreur Monte-Carlo : `mc_se` ≈ √((1 − p) / (n p)) pour chaque log-score d'étape (loi
binomiale, p lissée) ; c'est un ordre de grandeur, les tirages étant stratifiés par
configuration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from vvs.eval import metrics as mt

METHODS = ("reference", "mean14d")
# Étapes de la synthèse : (colonne de log-score, libellé). L'ordre suit la chaîne.
WINNER_CHAIN = (
    ("log_presence_winner", "presence"),
    ("log_qualify_given_presence_winner", "qualification|presence"),
    ("log_win_given_qualify_winner", "victory|qualification"),
)
PAIR_CHAIN = (
    ("log_presence_pair", "presence"),
    ("log_pair_given_presence", "qualification|presence"),
)


@dataclass(frozen=True)
class Outcome:
    """Résultat réel : parts officielles du premier tour, paire triée, vainqueur."""

    shares: dict[str, float]
    finalists: tuple[str, str]
    winner: str

    @property
    def official(self) -> list[str]:
        return sorted(self.shares)


def election_outcome(election: int, registry) -> Outcome:
    e = registry.elections[election]
    second = e.round_(2).results
    return Outcome(
        shares={r.candidate_id: float(r.share) for r in e.round_(1).results},
        finalists=tuple(sorted(r.candidate_id for r in second)),
        winner=max(second, key=lambda r: r.share).candidate_id,
    )


def jeffreys_log(k: int, n: int) -> float:
    """Log-score lissé d'un événement binaire observé k fois sur n ; log ½ si n = 0."""
    if not 0 <= k <= n:
        raise ValueError(f"effectifs incohérents : k={k}, n={n}")
    return math.log((k + 0.5) / (n + 1))


def mc_se(k: int, n: int) -> float:
    p = (k + 0.5) / (n + 1)
    return math.sqrt((1 - p) / ((n + 1) * p))


def presence_matrix(sim: dict, configurations: list[dict] | None = None) -> np.ndarray:
    """Présence (N, K) : depuis la configuration tirée si connue, sinon parts positives."""
    candidates = [str(c) for c in sim["candidates"]]
    shares = np.asarray(sim["shares"])
    if configurations is None:
        present = shares > 0
    else:
        index = {c: j for j, c in enumerate(candidates)}
        member = np.zeros((len(configurations), len(candidates)), dtype=bool)
        for k, cfg in enumerate(configurations):
            member[k, [index[c] for c in cfg["candidates"]]] = True
        present = member[np.asarray(sim["config"])]
        if (shares[~present] != 0).any() or (shares[present] <= 0).any():
            raise ValueError("parts incompatibles avec les configurations tirées")
    return present


def support(sim: dict, configurations: list[dict] | None = None) -> set[str]:
    present = presence_matrix(sim, configurations)
    return {str(c) for c, any_ in zip(sim["candidates"], present.any(axis=0)) if any_}


def _event(k: int, n: int) -> dict:
    return {
        "k": int(k),
        "n": int(n),
        "p": int(k) / int(n) if n else None,
        "log": jeffreys_log(int(k), int(n)),
        "mc_se": mc_se(int(k), int(n)),
        "evaluable": bool(n),
    }


def diagnose(
    sim: dict,
    outcome: Outcome,
    configurations: list[dict] | None = None,
    universe: set[str] | None = None,
    conditional: set[str] | None = None,
) -> tuple[dict, pl.DataFrame, list[dict]]:
    """Diagnostics d'un cas et d'un modèle : (synthèse, table par candidat, événements).

    `universe` et `conditional` fixent les univers d'une comparaison ; par défaut,
    officiels ∪ support et officiels présents au moins une fois.
    """
    candidates = [str(c) for c in sim["candidates"]]
    index = {c: j for j, c in enumerate(candidates)}
    shares = np.asarray(sim["shares"], dtype=float)
    n = shares.shape[0]
    present = presence_matrix(sim, configurations)
    names = np.array(candidates)
    qual = np.sort(names[np.asarray(sim["qualified"])], axis=1)
    wins = names[np.asarray(sim["winner"])]
    own_support = {c for c in candidates if present[:, index[c]].any()}
    official = set(outcome.shares)
    universe = set(universe) if universe is not None else official | own_support
    conditional = set(conditional) if conditional is not None else official & own_support
    if not own_support <= universe or not official <= universe:
        raise ValueError("l'univers commun doit contenir officiels et support")
    if not conditional <= official & own_support:
        raise ValueError("ensemble conditionnel hors des officiels présents")

    def col(c: str, matrix: np.ndarray) -> np.ndarray:
        return matrix[:, index[c]] if c in index else np.zeros(n, dtype=matrix.dtype)

    is_qual = {c: (qual == c).any(axis=1) for c in universe}
    a, b = outcome.finalists
    w = outcome.winner
    pair_present = col(a, present) & col(b, present)
    pair_qual = (qual[:, 0] == a) & (qual[:, 1] == b)
    w_present = col(w, present)
    w_qual = is_qual[w]
    w_win = wins == w

    events = {
        "presence_winner": _event(w_present.sum(), n),
        "qualify_given_presence_winner": _event(w_qual.sum(), w_present.sum()),
        "win_given_qualify_winner": _event(w_win.sum(), w_qual.sum()),
        "presence_pair": _event(pair_present.sum(), n),
        "pair_given_presence": _event(pair_qual.sum(), pair_present.sum()),
        "win_given_pair": _event((pair_qual & w_win).sum(), pair_qual.sum()),
    }
    if (w_qual & ~w_present).any() or (pair_qual & ~pair_present).any():
        raise ValueError("qualifié absent de sa configuration")
    # Composition : part des candidats finalement absents (« fantômes ») dans les tirages
    # où W est présent, et duels perdus contre eux. Descriptif, pas un score.
    phantoms = [index[c] for c in candidates if c not in outcome.shares]
    phantom_share = shares[:, phantoms].sum(axis=1) if phantoms else np.zeros(n)
    phantom_qual = ~np.isin(qual, list(outcome.shares)).all(axis=1)
    composition = {
        "phantom_share_given_presence_winner": (
            float(phantom_share[w_present].mean()) if w_present.any() else None
        ),
        "phantom_qualify_given_presence_winner": (
            float(phantom_qual[w_present].mean()) if w_present.any() else None
        ),
        "phantom_opponent_given_qualify_winner": (
            float(phantom_qual[w_qual].mean()) if w_qual.any() else None
        ),
    }

    rows = []
    for c in sorted(universe):
        p_c = col(c, present)
        k_c = int(p_c.sum())
        row = {
            "candidate_id": c,
            "official": c in official,
            "in_support": c in own_support,
            "actual_share": outcome.shares.get(c),
            "actual_qualified": c in outcome.finalists,
            "actual_winner": c == w,
            "n_present": k_c,
            "p_present": k_c / n,
            "p_qualify": float(is_qual[c].mean()),
            "p_win": float((wins == c).mean()),
            "n_qualified": int(is_qual[c].sum()),
            "p_qualify_given_present": float(is_qual[c].sum() / k_c) if k_c else None,
            "p_qualify_given_pair_present": (
                float(is_qual[c][pair_present].mean()) if pair_present.any() else None
            ),
            "crps_conditional": None,
            "covered50_conditional": None,
            "covered90_conditional": None,
            "crps_global": None,
            "in_conditional_set": c in conditional,
        }
        if c in official:
            y = outcome.shares[c]
            x_all = col(c, shares)
            row["crps_global"] = float(mt.crps(x_all, y))
            if k_c:
                x = x_all[p_c]
                row["crps_conditional"] = float(mt.crps(x, y))
                row["covered50_conditional"] = bool(mt.covered(x, y, 0.5))
                row["covered90_conditional"] = bool(mt.covered(x, y, 0.9))
        rows.append(row)
    table = pl.DataFrame(rows, infer_schema_length=None)

    target = table["official"].cast(float).to_numpy()
    p_pres = table["p_present"].to_numpy()
    cond = table.filter(pl.col("in_conditional_set"))
    cond_q = cond["p_qualify_given_present"].cast(float).to_numpy()
    cond_y = cond["actual_qualified"].cast(float).to_numpy()
    if pair_present.any():
        q_in_s = table["p_qualify_given_pair_present"].cast(float).to_numpy()
        brier_pair_s = float(
            np.mean((q_in_s - table["actual_qualified"].cast(float).to_numpy()) ** 2)
        )
    else:
        brier_pair_s = None

    # Score global, mêmes conventions que consolidation.score_chain (univers propre).
    k_own = len(set(candidates) | official)
    k_win, k_pair = int(w_win.sum()), int(pair_qual.sum())
    global_win = math.log((k_win + 0.5) / (n + 0.5 * k_own))
    global_pair = math.log((k_pair + 0.5) / (n + 0.5 * k_own * (k_own - 1) / 2))
    x_off = np.column_stack([col(c, shares) for c in outcome.shares])
    y_off = np.array(list(outcome.shares.values()))

    e = events
    stage_sum_win = sum(
        e[k]["log"]
        for k in ("presence_winner", "qualify_given_presence_winner", "win_given_qualify_winner")
    )
    stage_sum_pair = e["presence_pair"]["log"] + e["pair_given_presence"]["log"]
    raw_win = [
        e[k]["k"] / e[k]["n"] if e[k]["n"] else 0.0
        for k in ("presence_winner", "qualify_given_presence_winner", "win_given_qualify_winner")
    ]
    raw_pair = [
        e[k]["k"] / e[k]["n"] if e[k]["n"] else 0.0
        for k in ("presence_pair", "pair_given_presence")
    ]

    def raw_log_sum(parts):
        return sum(math.log(p) for p in parts) if all(p > 0 for p in parts) else None

    summary = {
        "n_draws": n,
        "n_universe": len(universe),
        "n_support": len(own_support),
        "n_official": len(official),
        "n_official_absent": len(official - own_support),
        "n_conditional": len(conditional),
        "unforecast_share": float(
            sum(s for c, s in outcome.shares.items() if c not in own_support)
        ),
        "conditional_excluded_share": float(
            sum(s for c, s in outcome.shares.items() if c not in conditional)
        ),
        # (a) présence
        "brier_presence_sum": float(np.sum((p_pres - target) ** 2)),
        "brier_presence_mean": float(np.mean((p_pres - target) ** 2)),
        # (b) qualification | présence
        "brier_qualify_given_presence": (
            float(np.mean((cond_q - cond_y) ** 2)) if cond.height else None
        ),
        "brier_qualify_given_pair_presence": brier_pair_s,
        # (d) soutien | présence
        "crps_conditional": (
            float(cond["crps_conditional"].cast(float).mean()) if cond.height else None
        ),
        "coverage50_conditional": (
            float(cond["covered50_conditional"].cast(float).mean()) if cond.height else None
        ),
        "coverage90_conditional": (
            float(cond["covered90_conditional"].cast(float).mean()) if cond.height else None
        ),
        "crps_global_on_conditional": (
            float(cond["crps_global"].cast(float).mean()) if cond.height else None
        ),
        # Score global (reproduit le gel)
        "p_winner": k_win / n,
        "p_pair": k_pair / n,
        "log_score_winner": global_win,
        "log_score_pair": global_pair,
        "crps_global": float(mt.crps(x_off, y_off).mean()),
        "coverage90_global": float(mt.covered(x_off, y_off, 0.9).mean()),
        # Décomposition
        "stage_sum_winner": stage_sum_win,
        "smoothing_gap_winner": global_win - stage_sum_win,
        "identity_gap_winner_raw": (
            raw_log_sum(raw_win) - math.log(k_win / n)
            if k_win and raw_log_sum(raw_win) is not None
            else None
        ),
        "stage_sum_pair": stage_sum_pair,
        "smoothing_gap_pair": global_pair - stage_sum_pair,
        "identity_gap_pair_raw": (
            raw_log_sum(raw_pair) - math.log(k_pair / n)
            if k_pair and raw_log_sum(raw_pair) is not None
            else None
        ),
    }
    summary |= composition
    for name, ev in events.items():
        summary[f"p_{name}"] = ev["p"]
        summary[f"log_{name}"] = ev["log"]
        summary[f"n_{name}"] = ev["n"]
        summary[f"evaluable_{name}"] = ev["evaluable"]
    summary["error_stage_winner"] = min(WINNER_CHAIN, key=lambda s: summary[s[0]])[1]
    event_rows = [{"event": name} | ev for name, ev in events.items()]
    return summary, table, event_rows


# --------------------------------------------------------------------------- dossiers


def load_case_draws(folder: Path, method: str) -> tuple[dict, list[dict]]:
    """Tirages et configurations d'un dossier de cas (gel ou variante, même format)."""
    case = json.loads((folder / "case.json").read_text())
    with np.load(folder / f"draws_{method}.npz") as draws:
        sim = {k: draws[k] for k in draws.files}
    return sim, case["configurations"]


def case_names(spec: dict) -> list[tuple[int, int, str]]:
    return [(e, h, f"{e}-{h:03d}") for e in spec["elections"] for h in spec["horizons"]]


def diagnose_folder(folder: Path, outcome: Outcome, methods=METHODS, common=True):
    """Diagnostics des méthodes d'un même cas, à univers commun entre ces méthodes."""
    loaded = {m: load_case_draws(folder, m) for m in methods}
    supports = {m: support(*loaded[m]) for m in methods}
    universe, conditional = None, None
    if common:
        universe = set(outcome.shares).union(*supports.values())
        conditional = set(outcome.shares).intersection(*supports.values())
    return {
        m: diagnose(loaded[m][0], outcome, loaded[m][1], universe, conditional) for m in methods
    }


DELTA_COLUMNS = (
    # (colonne, sens : +1 si plus haut = mieux)
    ("log_score_winner", 1),
    ("log_score_pair", 1),
    ("log_presence_winner", 1),
    ("log_presence_pair", 1),
    ("brier_presence_sum", -1),
    ("log_qualify_given_presence_winner", 1),
    ("log_pair_given_presence", 1),
    ("brier_qualify_given_presence", -1),
    ("log_win_given_qualify_winner", 1),
    ("log_win_given_pair", 1),
    ("crps_conditional", -1),
    ("coverage90_conditional", 1),
    ("crps_global", -1),
    ("coverage90_global", 1),
)


def compare_study(study: Path, reference: Path, registry, methods=METHODS):
    """Diagnostics d'une étude de variantes et écarts par étape contre la référence.

    `study` suit le format de runs/2026-10-22-historical-scenarios :
    cases/<élection>-<horizon>/<variante>/{case.json, draws_<méthode>.npz}. La référence
    suit celui du gel : cases/<élection>-<horizon>/{case.json, draws_<méthode>.npz}.
    Pour chaque cas, variante et méthode, l'univers de présence et l'ensemble
    conditionnel sont communs à la variante et à la référence de même méthode.
    Rend (synthèses longues, écarts variante − référence).
    """
    from vvs.eval.consolidation import complete_score, verify

    spec = verify(reference)["spec"]
    summaries, deltas = [], []
    for election, horizon, name in case_names(spec):
        original = reference / "cases" / name
        if not complete_score(original):
            raise ValueError(f"cas de référence incomplet : {name}")
        folder = study / "cases" / name
        variants = sorted(p.name for p in folder.iterdir() if (p / "case.json").is_file())
        if not variants:
            raise ValueError(f"aucune variante lisible : {folder}")
        _check_study_case(folder)
        outcome = election_outcome(election, registry)
        for method in methods:
            ref = load_case_draws(original, method)
            ref_support = support(*ref)
            for variant in variants:
                var = load_case_draws(folder / variant, method)
                var_support = support(*var)
                universe = set(outcome.shares) | ref_support | var_support
                conditional = set(outcome.shares) & ref_support & var_support
                key = {"election": election, "horizon": horizon, "method": method}
                s_ref = diagnose(ref[0], outcome, ref[1], universe, conditional)[0]
                s_var = diagnose(var[0], outcome, var[1], universe, conditional)[0]
                summaries.append(key | {"variant": "reference_frozen", "against": variant} | s_ref)
                summaries.append(key | {"variant": variant, "against": "reference_frozen"} | s_var)
                row = key | {
                    "variant": variant,
                    "n_universe": len(universe),
                    "n_conditional": len(conditional),
                }
                for column, sign in DELTA_COLUMNS:
                    a, b = s_var[column], s_ref[column]
                    delta = None if a is None or b is None else a - b
                    row[f"delta_{column}"] = delta
                    row[f"better_{column}"] = None if delta is None else bool(sign * delta > 0)
                deltas.append(row)
    return (
        pl.DataFrame(summaries, infer_schema_length=None),
        pl.DataFrame(deltas, infer_schema_length=None),
    )


def _check_study_case(folder: Path) -> None:
    """Contrôle les empreintes de complete.json quand l'étude en fournit."""
    import hashlib

    manifest = folder / "complete.json"
    if not manifest.exists():
        return
    expected = json.loads(manifest.read_text())
    for rel, sha in expected.items():
        file = folder / rel
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != sha:
            raise ValueError(f"sortie d'étude modifiée : {folder}/{rel}")
