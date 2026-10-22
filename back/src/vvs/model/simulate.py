"""Simulation de la présidentielle 2027 (docs/04-modele.md, section 5).

Assemble les briques déjà ajustées : tirages du postérieur du premier tour (`vvs.model.run`,
`save_trajectories`), modèle d'erreur η(τ) (`vvs.model.error`), scénarios de candidats
(`vvs.registry.scenarios`) et second tour par reports (`vvs.model.second_round`).

Pour chaque tirage :

1. une configuration de dimensions `k` (scenarios_2027.yaml, mode `marginal`), puis la
   présence de chaque petit candidat par une Bernoulli de probabilité `small_presence` : c'est
   l'énumération complète des petits candidats, faite par Monte-Carlo plutôt qu'en listant
   2^20 sous-ensembles ;
2. un tirage `s` du postérieur : θ à la semaine `asof` de chaque candidat présent (voir
   « Candidats peu ou pas testés ») ;
3. η(τ) (biais de bloc désactivé, comme au backtest), parts de premier tour par le softmax
   restreint aux présents (`ShareRule`, IIA assumée), puis la part d'exprimés parmi les
   inscrits, tirée autour des premiers tours 2002-2022 (logit-normale) : elle donne le poids
   des abstentionnistes, blancs et nuls au second tour ;
4. les deux premiers sont qualifiés ;
5. second tour : `SecondRoundModel.sample` sur les tirages de premier tour de la cellule
   (configuration, paire), voir « Second tour » ;
6. on garde parts, qualifiés, vainqueur, configuration, participation.

Allocation. Les tirages sont répartis entre configurations proportionnellement à leur
probabilité (plus forts restes) : chaque tirage pèse autant, le mélange est exactement la
somme pondérée des conditions, et les configurations de probabilité inférieure à 1/(2n) ne
reçoivent aucun tirage (masse rapportée dans le manifeste).

Candidats peu ou pas testés. Le postérieur n'informe que les candidats testés dans la grille
de l'ajustement (78 semaines). Règle, par candidat de l'univers des scénarios :

- au moins `MIN_HYPOTHESES` hypothèses de sondage dans la grille : θ du postérieur ;
- candidat d'une dimension jamais ou trop peu testé (Bompard, remplaçant LFI) et candidat
  non identifié (`placeholder_<bloc>`) : θ = β du bloc, la moyenne des logits des candidats
  du bloc dans le postérieur (convention d'identification du modèle : δ somme à 0 dans un
  bloc), soit un « candidat moyen du bloc » ;
- petit candidat trop peu testé : part tirée à la date de prévision dans une log-normale
  centrée sur sa part médiane dans les sondages de la grille, à défaut son résultat à la
  dernière présidentielle, à défaut `SPARSE_DEFAULT_SHARE`, d'écart-type `SPARSE_SD` en log ;
  les autres candidats sont réduits proportionnellement (IIA), puis η s'applique à tous.

Second tour. `SecondRoundModel.sample` rééchantillonne les tirages de premier tour qu'on lui
donne (échantillonnage d'importance pondéré par les sondages de duel) : ses tirages ne sont
pas alignés ligne à ligne sur les nôtres. On l'appelle donc par cellule (configuration,
paire) avec les tirages de premier tour de la cellule, et on affecte ses `n` tirages aux `n`
tirages de la cellule. Les probabilités de victoire par paire et par condition sont exactes au
bruit Monte-Carlo près ; seule la corrélation, à l'intérieur d'une cellule, entre la part de
premier tour d'un tirage et son second tour est perdue. Les configurations hors des plus
probables (`CONFIG_MASS` de la masse) sont regroupées en une cellule « autres » par paire.

Biais RN du second tour. Option `rn_bias_curve` : biais estimé par âge des sondages de duel
(`second_round.bias_curve`, leave-one-election-out au backtest, rétréci vers 0 et monotone),
appliqué sondage par sondage ; `rn_bias` le multiplie (1 par défaut). Sans courbe (option
désactivée, ou erreurs injectées sans courbe) : atténuation linéaire provisoire du biais
constant, plein effet à `RN_BIAS_FULL_DAYS` jours ou moins du premier tour, nul à
`RN_BIAS_ZERO_DAYS` jours ou plus (`rn_bias_scale`), ou part imposée par `rn_bias`. Le mode
retenu est écrit dans le manifeste (`second_round_model.bias_mode`).

Couplage premier tour / duel (option `couple_far_right`). La composante de bloc far_right de
η de chaque tirage (`ErrorModel.sample_eta_parts`, réduite, avant λ) est passée au second tour
(`SecondRoundModel.sample(shared_error=…)`), qui corrèle avec elle l'erreur commune des
sondages de duel (ρ estimé par `second_round.far_right_coupling` sur 2002, 2017, 2022). Le
second tour rééchantillonnant les tirages de la cellule, le couplage agit sur la loi de la
cellule, pas ligne à ligne (même limite que ci-dessus).

Participation. Le poids des abstentionnistes, blancs et nuls du premier tour vient de la part
d'exprimés du premier tour (point 3). Celle du second tour (exprimés / inscrits) est, sans
option, implicite : Σ w (t_i + t_j) / Σ w des reports tirés. Option `second_round_turnout` :
tirée d'une logit-normale ajustée sur les seconds tours 2002-2022 et imposée aux reports
(`second_round.match_turnout`). Colonne `turnout_second_round` des tirages dans les deux cas.

Point d'extension de la substitution intra-bloc (phase 4) : `ShareRule`. La règle par défaut
`iia_shares` est le softmax restreint ; une règle de substitution recevra les mêmes logits et
la même présence et redistribuera d'abord la masse d'un bloc dans le bloc.

Sorties (`out/`) : draws.parquet, summary.parquet, pairs.parquet, transfers.parquet,
trajectories.parquet, house_effects.parquet, conditions.json, manifest.json.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.registry import Registry, default_registry
from vvs.registry.scenarios import (
    Enumeration,
    ScenariosFile,
    enumerate_configurations,
    load_scenarios,
)

log = logging.getLogger(__name__)

DEFAULT_DRAWS = 40_000
MIN_HYPOTHESES = 3  # hypothèses dans la grille pour utiliser θ du postérieur
SPARSE_SD = 0.5  # écart-type (log) de la part d'un petit candidat trop peu testé
SPARSE_DEFAULT_SHARE = 1.0  # part médiane (points) d'un petit candidat jamais candidat
CONFIG_MASS = 0.95  # masse des configurations portées individuellement au second tour
RN_BIAS_FULL_DAYS = 30
RN_BIAS_ZERO_DAYS = 180
# options du second tour (runs/2026-10-20-second-round-v2) : valeurs par défaut de `simulate`
RN_BIAS_CURVE = False
COUPLE_FAR_RIGHT = False
SECOND_ROUND_TURNOUT = False
TURNOUT_ELECTIONS = (2002, 2007, 2012, 2017, 2022)
ERROR_ELECTIONS = (2002, 2007, 2012, 2017, 2022)
# η calibré aussi sur les erreurs finales des scrutins de liste (ErrorModel.fit(with_lists=…),
# vvs.model.error_lists ; backtest : runs/2026-10-20-backtest-lists)
ERROR_WITH_LISTS = True
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
QNAMES = ("q05", "q25", "q50", "q75", "q95")
OTHER_CELL = -1  # cellule de second tour des configurations hors des plus probables


def placeholder_id(bloc: str) -> str:
    return f"placeholder_{bloc}"


def rn_bias_scale(
    horizon: float, full_days: float = RN_BIAS_FULL_DAYS, zero_days: float = RN_BIAS_ZERO_DAYS
) -> float:
    """Part du biais RN du second tour appliquée à `horizon` jours du premier tour : 1 à
    `full_days` ou moins, 0 à `zero_days` ou plus, linéaire entre les deux (provisoire)."""
    if zero_days <= full_days:
        raise ValueError("zero_days doit dépasser full_days")
    return float(np.clip((zero_days - horizon) / (zero_days - full_days), 0.0, 1.0))


def quantile_dict(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(x, dtype=float), QUANTILES)
    return dict(zip(QNAMES, map(float, q)))


# --- règle de parts (point d'extension de la substitution)

ShareRule = Callable[[np.ndarray, np.ndarray, Sequence[str], "dict | None"], np.ndarray]


def iia_shares(
    logits: np.ndarray, present: np.ndarray, blocs: Sequence[str], ctx: dict | None = None
) -> np.ndarray:
    """Softmax restreint aux candidats présents (indépendance des alternatives non pertinentes).

    `logits` (…, U), `present` (…, U) booléen ou diffusable ; parts nulles pour les absents.
    `blocs` et `ctx` sont inutilisés ici : ce sont les arguments de `nested_shares`."""
    x = np.where(present, logits, -np.inf)
    m = np.max(x, axis=-1, keepdims=True)
    z = np.where(present, np.exp(x - m), 0.0)
    return z / z.sum(-1, keepdims=True)


def _lse(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return m + np.log(np.exp(x - m).sum(-1, keepdims=True))


def nested_shares(
    logits: np.ndarray, present: np.ndarray, blocs: Sequence[str], ctx: dict | None
) -> np.ndarray:
    """Logit emboîté par bloc (substitution intra-bloc, `Priors.substitution`), même forme que
    `first_round.config_log_shares` mais avec une présence variable par tirage.

    `logits` = θ = β + δ (…, U) ; `ctx` : `beta` (…, B) aligné sur les dimensions de tête de
    `logits`, `lam` (N, B), `cand_bloc` (U,) indices dans B. Un bloc sans candidat présent
    pèse zéro ; λ = 1 redonne `iia_shares`."""
    if ctx is None:
        raise ValueError("nested_shares : contexte (β, λ, cand_bloc) requis")
    beta, lam, cb = ctx["beta"], ctx["lam"], ctx["cand_bloc"]
    if lam.ndim == 2 and logits.ndim == 3:  # (N, B) -> (N, 1, B) pour les trajectoires
        lam = lam[:, None, :]
    delta = logits - beta[..., cb]
    d = np.where(present, delta, -np.inf)
    nb = beta.shape[-1]
    lse = np.full(beta.shape, -np.inf)
    for b in range(nb):
        idx = cb == b
        if idx.any():
            lse[..., b] = _lse(d[..., idx])[..., 0]
    util = np.where(np.isfinite(lse), beta + lam * lse, -np.inf)
    log_pb = util - _lse(util)
    log_share = np.where(present, log_pb[..., cb] + d - lse[..., cb], -np.inf)
    z = np.where(present, np.exp(log_share), 0.0)
    return z / z.sum(-1, keepdims=True)


def share_context(fit: FitDraws, uni: Universe, s_idx: np.ndarray, week: int | slice) -> dict | None:
    """Contexte de `nested_shares` pour les tirages `s_idx` (None sans substitution)."""
    if fit.lam is None:
        return None
    bloc_index = {b: i for i, b in enumerate(fit.bloc_ids)}
    cb = np.array([bloc_index[b] for b in uni.blocs])
    return {"beta": fit.beta[s_idx][:, week], "lam": fit.lam[s_idx], "cand_bloc": cb}


def perturbed_shares(ref: np.ndarray, eta: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Parts au jour du vote : η s'applique aux log-parts (c'est sur des différences de
    log-parts qu'il est calibré), puis renormalisation sur les présents. Identique à
    softmax(θ + η) sous l'IIA."""
    log_ref = np.log(np.where(present, ref, 1.0))
    return iia_shares(log_ref + eta, present, [])


# --- tirages du premier tour


@dataclass
class FitDraws:
    """Tirages sauvegardés par `run_fit(save_draws=True, save_trajectories=True)`."""

    fit_dir: Path
    candidates: list[str]
    blocs: list[str]
    theta: np.ndarray  # (S, T, C)
    alpha: np.ndarray  # (S, P, C)
    pollsters: list[str]
    week_dates: list[dt.date]
    asof_week: int
    observed: pl.DataFrame
    manifest: dict
    beta: np.ndarray | None = None  # (S, T, B), substitution seulement
    lam: np.ndarray | None = None  # (S, B)
    bloc_ids: list[str] | None = None

    @property
    def substitution(self) -> bool:
        return self.lam is not None

    @classmethod
    def load(cls, fit_dir: Path) -> FitDraws:
        fit_dir = Path(fit_dir)
        d = np.load(fit_dir / "draws.npz")
        if "theta" not in d.files:
            raise ValueError(f"{fit_dir}/draws.npz sans trajectoires : relancer avec save_trajectories")
        extra: dict = {}
        if "lam" in d.files:
            # ajustement avec substitution intra-bloc (logit emboîté) : les parts se calculent
            # par `nested_shares` à partir de β et λ par tirage
            if "beta" not in d.files:
                raise ValueError(f"{fit_dir}/draws.npz : substitution sans β hebdomadaire")
            extra = {
                "beta": d["beta"].astype(np.float64),
                "lam": d["lam"].astype(np.float64),
                "bloc_ids": [str(b) for b in d["bloc_ids"]],
            }
        return cls(
            **extra,
            fit_dir=fit_dir,
            candidates=[str(c) for c in d["candidates"]],
            blocs=[str(b) for b in d["blocs"]],
            theta=d["theta"].astype(np.float64),
            alpha=d["alpha"].astype(np.float64),
            pollsters=[str(p) for p in d["pollsters"]],
            week_dates=[dt.date.fromisoformat(str(x)) for x in d["week_dates"]],
            asof_week=int(d["asof_week"]),
            observed=pl.read_parquet(fit_dir / "observed.parquet"),
            manifest=json.loads((fit_dir / "manifest.json").read_text()),
        )

    @property
    def n_draws(self) -> int:
        return self.theta.shape[0]

    def n_hypotheses(self) -> dict[str, int]:
        c = self.observed.group_by("candidate_id").len()
        return dict(zip(c["candidate_id"].to_list(), c["len"].to_list()))

    def first_week(self) -> dict[str, int]:
        c = self.observed.group_by("candidate_id").agg(pl.col("week").min())
        return dict(zip(c["candidate_id"].to_list(), c["week"].to_list()))

    def median_share(self) -> dict[str, float]:
        """Part médiane observée (points, parts normalisées sur l'hypothèse) dans la grille."""
        c = self.observed.group_by("candidate_id").agg((100 * pl.col("share")).median().alias("share"))
        return dict(zip(c["candidate_id"].to_list(), c["share"].to_list()))

    def bloc_theta(self, bloc: str, week: int | slice) -> np.ndarray:
        """β du bloc : moyenne des logits des candidats du bloc, (S,) ou (S, T)."""
        idx = [i for i, b in enumerate(self.blocs) if b == bloc]
        if not idx:
            raise KeyError(f"aucun candidat du bloc {bloc} dans le postérieur")
        return self.theta[:, week][..., idx].mean(-1)


@dataclass
class Universe:
    """Tous les candidats possibles d'après les scénarios, et l'origine de leur θ."""

    ids: list[str]
    blocs: list[str]
    source: list[str]  # posterior, bloc ou prior
    fit_index: list[int | None]
    prior_share: list[float | None]  # part médiane a priori (points) des candidats `prior`
    prior_basis: list[str | None]

    def index(self, cid: str) -> int:
        return self.ids.index(cid)

    def table(self) -> list[dict]:
        return [
            {
                "candidate_id": c,
                "bloc": b,
                "source": s,
                "prior_share": p,
                "prior_basis": pb,
            }
            for c, b, s, p, pb in zip(
                self.ids, self.blocs, self.source, self.prior_share, self.prior_basis
            )
        ]


def last_presidential_share(registry: Registry, cid: str, before: int) -> tuple[float, int] | None:
    years = sorted(
        (y for y, e in registry.elections.items() if y < before and e.round_(1).results),
        reverse=True,
    )
    for y in years:
        for r in registry.elections[y].round_(1).results:
            if r.candidate_id == cid:
                return float(r.share), y
    return None


def build_universe(
    scenarios: ScenariosFile,
    fit: FitDraws,
    registry: Registry,
    min_hypotheses: int = MIN_HYPOTHESES,
) -> Universe:
    n_hyp = fit.n_hypotheses()
    med = fit.median_share()
    small = {s.candidate_id: s.bloc for s in scenarios.small_candidates}
    ids: list[str] = []
    for d in scenarios.dimensions:
        for s in d.states:
            ids += [c for c in s.candidates if c not in ids]
            ids += [placeholder_id(b) for b in s.placeholders if placeholder_id(b) not in ids]
    ids += [c for c in small if c not in ids]
    blocs, source, fidx, prior, basis = [], [], [], [], []
    for c in ids:
        if c.startswith("placeholder_"):
            b = c.removeprefix("placeholder_")
        else:
            b = small.get(c) or registry.bloc_of(c, scenarios.election)
        blocs.append(b)
        tested = n_hyp.get(c, 0) >= min_hypotheses and c in fit.candidates
        if tested:
            source.append("posterior")
            fidx.append(fit.candidates.index(c))
            prior.append(None)
            basis.append(None)
        elif c in small:
            source.append("prior")
            fidx.append(None)
            if c in med:
                prior.append(float(med[c]))
                basis.append(f"médiane de {n_hyp.get(c, 0)} hypothèse(s) de la grille")
            elif (last := last_presidential_share(registry, c, scenarios.election)) is not None:
                prior.append(last[0])
                basis.append(f"résultat {last[1]}")
            else:
                prior.append(SPARSE_DEFAULT_SHARE)
                basis.append("valeur par défaut")
        else:
            if b not in fit.blocs:
                raise ValueError(f"{c} : bloc {b} absent du postérieur, θ de bloc impossible")
            source.append("bloc")
            fidx.append(None)
            prior.append(None)
            basis.append(f"β du bloc {b}")
    return Universe(ids, blocs, source, fidx, prior, basis)


def universe_theta(fit: FitDraws, uni: Universe, s_idx: np.ndarray, week: int | slice) -> np.ndarray:
    """θ (N, U) ou (N, T, U) des candidats `posterior` et `bloc` ; NaN pour les `prior`."""
    th = fit.theta[s_idx][:, week]  # (N, C) ou (N, T, C)
    out = np.full((*th.shape[:-1], len(uni.ids)), np.nan)
    bloc_cache: dict[str, np.ndarray] = {}
    for u, (src, b) in enumerate(zip(uni.source, uni.blocs)):
        if src == "posterior":
            out[..., u] = th[..., uni.fit_index[u]]
        elif src == "bloc":
            if b not in bloc_cache:
                idx = [i for i, x in enumerate(fit.blocs) if x == b]
                bloc_cache[b] = th[..., idx].mean(-1)
            out[..., u] = bloc_cache[b]
    return out


def insert_prior_shares(
    theta: np.ndarray, present: np.ndarray, uni: Universe, rng: np.random.Generator
) -> np.ndarray:
    """Complète θ des candidats `prior` présents : part tirée (log-normale, constante dans le
    temps si θ a un axe semaine) et réduction proportionnelle des autres présents."""
    prior_idx = [u for u, s in enumerate(uni.source) if s == "prior"]
    if not prior_idx:
        return theta
    n = present.shape[0]
    med = np.array([uni.prior_share[u] for u in prior_idx]) / 100.0
    s = med * np.exp(SPARSE_SD * rng.standard_normal((n, len(prior_idx))))
    on = present[:, prior_idx]
    s = np.where(on, s, 0.0)
    total = s.sum(-1)
    if np.any(total >= 0.5):
        scale = np.where(total >= 0.5, 0.5 / np.maximum(total, 1e-12), 1.0)
        s, total = s * scale[:, None], np.minimum(total, 0.5)
    out = theta.copy()
    known = present.copy()
    known[:, prior_idx] = False
    x = np.where(known if out.ndim == 2 else known[:, None, :], out, -np.inf)
    m = np.max(x, axis=-1, keepdims=True)
    lse = m[..., 0] + np.log(np.exp(x - m).sum(-1))  # (N,) ou (N, T)
    for j, u in enumerate(prior_idx):
        with np.errstate(divide="ignore"):
            val = np.log(s[:, j]) - np.log1p(-total)
        if out.ndim == 3:
            val = val[:, None]
        out[..., u] = np.where(np.isfinite(val), lse + val, np.nan)
    return out


# --- configurations et conditions


def allocate(probabilities: np.ndarray, n: int) -> np.ndarray:
    """Effectifs entiers proportionnels aux probabilités, de somme n (plus forts restes)."""
    p = np.asarray(probabilities, dtype=float)
    p = p / p.sum()
    raw = p * n
    k = np.floor(raw).astype(int)
    rest = n - k.sum()
    if rest > 0:
        k[np.argsort(-(raw - k), kind="stable")[:rest]] += 1
    return k


def sample_presence(
    enum: Enumeration, cfg: np.ndarray, uni: Universe, rng: np.random.Generator
) -> np.ndarray:
    """Présence (N, U) : candidats nommés et placeholders de la configuration, petits candidats
    par Bernoulli de `small_presence`."""
    n = len(cfg)
    present = np.zeros((n, len(uni.ids)), dtype=bool)
    for k in np.unique(cfg):
        c = enum[int(k)]
        rows = np.flatnonzero(cfg == k)
        named = [uni.index(x) for x in c.candidates]
        named += [uni.index(placeholder_id(b)) for b in c.placeholders]
        present[np.ix_(rows, named)] = True
        for cid, q in c.small_presence.items():
            present[rows, uni.index(cid)] = rng.random(len(rows)) < q
    return present


def config_id(states: dict[str, str], dims: Sequence[str]) -> str:
    return "cfg." + "-".join(states[d] for d in dims)


@dataclass
class Condition:
    id: str
    kind: str  # mixture, dimension_state, configuration
    label: str
    states: dict[str, str]
    probability: float
    configs: np.ndarray  # index des configurations membres

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "states": dict(self.states),
            "probability": self.probability,
            "configs": [int(x) for x in self.configs],
        }


def build_conditions(f: ScenariosFile, enum: Enumeration, n_configs: int) -> list[Condition]:
    """`mix`, chaque état de dimension, puis les `n_configs` configurations les plus probables."""
    p = np.array([c.probability for c in enum])
    dims = [d.id for d in f.dimensions]
    out = [Condition("mix", "mixture", "Mélange des scénarios", {}, 1.0, np.arange(len(enum)))]
    for d in f.dimensions:
        for s in d.states:
            members = np.array([k for k, c in enumerate(enum) if c.states[d.id] == s.id], dtype=int)
            if len(members) == 0:
                continue
            out.append(
                Condition(
                    f"{d.id}.{s.id}",
                    "dimension_state",
                    d.state_label(s.id),
                    {d.id: s.id},
                    float(p[members].sum()),
                    members,
                )
            )
    for k, c in enumerate(list(enum)[:n_configs]):
        states = dict(c.states)
        label = " · ".join(f.dimension(d).state(states[d]).label for d in dims)
        out.append(
            Condition(config_id(states, dims), "configuration", label, states, c.probability,
                      np.array([k]))
        )
    return out


# --- participation


def turnout_model(registry: Registry, elections=TURNOUT_ELECTIONS) -> dict:
    """Part d'exprimés parmi les inscrits au premier tour : moyenne et écart-type sur l'échelle
    logit des premiers tours `elections` (« modèle simple » de docs/04 § 5, point 3)."""
    vals = {}
    for y in elections:
        r = registry.elections[y].round_(1)
        if r.registered and r.expressed:
            vals[y] = r.expressed / r.registered
    x = np.array(list(vals.values()))
    lx = np.log(x / (1 - x))
    return {
        "elections": {str(k): round(v, 4) for k, v in vals.items()},
        "logit_mean": float(lx.mean()),
        "logit_sd": float(lx.std(ddof=1)),
    }


def sample_turnout(model: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    z = model["logit_mean"] + model["logit_sd"] * rng.standard_normal(n)
    return 1.0 / (1.0 + np.exp(-z))


# --- second tour


def placeholder_registry(registry: Registry) -> Registry:
    """Registre dont `bloc_of` connaît les candidats non identifiés `placeholder_<bloc>`."""

    class _PlaceholderRegistry(Registry):
        def bloc_of(self, candidate_id: str, election: int) -> str:
            if candidate_id.startswith("placeholder_"):
                return candidate_id.removeprefix("placeholder_")
            return super().bloc_of(candidate_id, election)

    return _PlaceholderRegistry(**{f.name: getattr(registry, f.name) for f in fields(registry)})


def orient_pair(a: str, b: str, bloc: dict[str, str], strength: dict[str, float]) -> tuple[str, str]:
    """Ordre d'affichage d'un duel : le finaliste d'extrême droite d'abord s'il y en a un seul,
    sinon le plus fort au premier tour."""
    fa, fb = bloc[a] == "far_right", bloc[b] == "far_right"
    if fa != fb:
        return (a, b) if fa else (b, a)
    return (a, b) if strength.get(a, 0) >= strength.get(b, 0) else (b, a)


# --- simulation


@dataclass
class SimulationResult:
    draws: pl.DataFrame
    summary: pl.DataFrame
    pairs: pl.DataFrame
    transfers: pl.DataFrame
    trajectories: pl.DataFrame
    house_effects: pl.DataFrame
    conditions: list[Condition]
    manifest: dict


def first_round_draws(
    fit: FitDraws,
    uni: Universe,
    enum: Enumeration,
    error_model,
    horizon: float,
    n: int,
    rng: np.random.Generator,
    share_rule: ShareRule = iia_shares,
) -> dict:
    """Configurations, présence, parts de premier tour au jour du vote (fractions)."""
    p = np.array([c.probability for c in enum])
    counts = allocate(p, n)
    cfg = np.repeat(np.arange(len(enum)), counts)
    rng.shuffle(cfg)
    s_idx = rng.integers(0, fit.n_draws, size=n)
    present = sample_presence(enum, cfg, uni, rng)
    theta = universe_theta(fit, uni, s_idx, fit.asof_week)
    theta = insert_prior_shares(theta, present, uni, rng)
    theta_safe = np.where(present, theta, 0.0)
    ctx = share_context(fit, uni, s_idx, fit.asof_week)
    ref = share_rule(theta_safe, present, uni.blocs, ctx)
    p_ref = np.where(present, ref, 1.0)  # parts des absents sans effet (η retiré ensuite)
    far_right_z = None
    if hasattr(error_model, "sample_eta_parts"):
        # mêmes tirages que sample_eta, plus la composante de bloc réduite (couplage)
        parts = error_model.sample_eta_parts(horizon, list(uni.blocs), p_ref, rng)
        eta, far_right_z = parts.eta, parts.bloc("far_right")
    else:
        eta = error_model.sample_eta(horizon, list(uni.blocs), p_ref, rng)
    shares = perturbed_shares(ref, eta, present)
    return {
        "config": cfg,
        "posterior": s_idx,
        "present": present,
        "shares": shares,
        "config_counts": counts,
        "far_right_z": far_right_z,
    }


def simulate(
    fit_dir: Path,
    asof: dt.date,
    out: Path | None = None,
    election: int = 2027,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    config_mass: float = CONFIG_MASS,
    n_condition_configs: int | None = None,
    rn_bias: float | None = None,
    rn_bias_full_days: float = RN_BIAS_FULL_DAYS,
    rn_bias_zero_days: float = RN_BIAS_ZERO_DAYS,
    rn_bias_curve: bool = RN_BIAS_CURVE,
    couple_far_right: bool = COUPLE_FAR_RIGHT,
    second_round_turnout: bool = SECOND_ROUND_TURNOUT,
    error_with_lists: bool = ERROR_WITH_LISTS,
    min_hypotheses: int = MIN_HYPOTHESES,
    share_rule: ShareRule | None = None,
    scenarios: ScenariosFile | None = None,
    registry: Registry | None = None,
    error_model=None,
    second_round_errors=None,
    second_round_settings=None,
    trajectory_draws: int | None = None,
) -> SimulationResult:
    """Simulation complète ; écrit les sorties dans `out` si donné.

    `rn_bias` : part du biais RN du second tour appliquée (défaut : 1 avec la courbe par
    horizon, `rn_bias_scale` à l'horizon sinon). `rn_bias_curve`, `couple_far_right`,
    `second_round_turnout` : options du second tour (docstring du module).
    `error_with_lists` : η calibré aussi sur les scrutins de liste (`ErrorModel.fit`).
    `n_condition_configs` : configurations résumées individuellement (défaut : celles qui
    couvrent `config_mass`). `error_model`, `second_round_errors`, `scenarios`, `registry` :
    injectables pour les tests (défaut : calibrés sur 2002-2022, fichiers du dépôt).
    `second_round_settings` : réglages du prior des reports (`SecondRoundSettings` : σ_u, σ_v,
    matrice de proximité ; défaut : constantes de second_round.py), écrits dans le manifeste."""
    from vvs.model.error import ErrorModel
    from vvs.model.second_round import SecondRoundModel, calibrate_errors

    started = time.time()
    reg = registry or default_registry()
    f = scenarios or load_scenarios(registry=reg)
    fit = FitDraws.load(fit_dir)
    if share_rule is None:
        share_rule = nested_shares if fit.substitution else iia_shares
    e = reg.elections[election]
    d1, d2 = e.round_(1).date, e.round_(2).date
    horizon = float((d1 - asof).days)
    rng = np.random.default_rng(seed)

    enum = enumerate_configurations(f, small="marginal")
    p_cfg = np.array([c.probability for c in enum])
    n_top = int(np.searchsorted(np.cumsum(p_cfg), config_mass - 1e-12) + 1)
    n_top = min(n_top, len(enum))
    n_cond_cfg = n_top if n_condition_configs is None else min(n_condition_configs, len(enum))
    conditions = build_conditions(f, enum, n_cond_cfg)
    uni = build_universe(f, fit, reg, min_hypotheses)
    bloc_of = dict(zip(uni.ids, uni.blocs))

    # 1-3. premier tour
    if error_model is None:
        error_model = ErrorModel.fit(elections=ERROR_ELECTIONS, registry=reg, use_bias=False,
                                     with_lists=error_with_lists)
    fr = first_round_draws(fit, uni, enum, error_model, horizon, n_draws, rng, share_rule)
    cfg, present, shares = fr["config"], fr["present"], fr["shares"]
    turnout_params = turnout_model(reg)
    turnout = sample_turnout(turnout_params, rng, n_draws)
    abstain = (1 - turnout) / turnout

    # 4. qualifiés, orientés pour l'affichage (a, b) ; tout est en index de l'univers
    order = np.argsort(-shares, axis=1)
    strength = {c: float(np.median(shares[present[:, u], u])) for u, c in enumerate(uni.ids)
                if present[:, u].any()}
    pair_list: list[tuple[str, str]] = []
    pair_code = np.empty(n_draws, dtype=int)
    code_of: dict[tuple[int, int], int] = {}
    for i, (x, y) in enumerate(zip(order[:, 0], order[:, 1])):
        k = (min(x, y), max(x, y))
        if k not in code_of:
            code_of[k] = len(pair_list)
            pair_list.append(orient_pair(uni.ids[x], uni.ids[y], bloc_of, strength))
        pair_code[i] = code_of[k]
    pa = np.array([uni.index(a) for a, _ in pair_list])
    pb = np.array([uni.index(b) for _, b in pair_list])
    a_idx, b_idx = pa[pair_code], pb[pair_code]

    # 5. second tour, par cellule (configuration ou « autres », paire)
    far_right_z = fr["far_right_z"] if couple_far_right else None
    if couple_far_right and far_right_z is None:
        raise ValueError("couple_far_right : le modèle d'erreur ne rend pas la composante de bloc")
    sr_reg = placeholder_registry(reg)
    base_errors = second_round_errors or calibrate_errors(
        exclude={election}, registry=reg, bias_by_horizon=rn_bias_curve
    )
    if rn_bias is not None:
        scale, bias_mode = float(rn_bias), "manual"
    elif rn_bias_curve and base_errors.bias_curve:
        scale, bias_mode = 1.0, "curve"
    else:
        scale, bias_mode = rn_bias_scale(horizon, rn_bias_full_days, rn_bias_zero_days), "linear"
    if bias_mode == "manual" and rn_bias_curve and base_errors.bias_curve:
        bias_mode = "manual_curve"
    errors = base_errors.scaled_bias(scale)
    if not (rn_bias_curve and base_errors.bias_curve):
        errors = replace(errors, bias_curve=())  # biais constant × part (comportement historique)
    model = SecondRoundModel(
        sr_reg,
        errors=errors,
        exclude={election},
        couple_far_right=couple_far_right,
        second_round_turnout=second_round_turnout,
        settings=second_round_settings,
    ).fit(election, asof, pair_list)
    cell = np.where(cfg < n_top, cfg, OTHER_CELL)
    share_a = np.full(n_draws, np.nan)
    turnout2 = np.full(n_draws, np.nan)
    cells: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(zip(cell.tolist(), pair_code.tolist())):
        cells.setdefault(key, []).append(i)
    transfer_acc: dict[tuple[str, str], dict[str, dict]] = {}
    pair_ess: dict[tuple[str, str], float] = {}
    for (_, code), idx in cells.items():
        rows = np.array(idx)
        pair = pair_list[code]
        cols = np.flatnonzero(present[rows].any(0))
        frd = {uni.ids[u]: shares[rows, u] for u in cols}
        frd["abstain"] = abstain[rows]
        shared = None if far_right_z is None else far_right_z[rows]
        d = model.sample(pair, frd, horizon, rng, n=len(rows), shared_error=shared)
        share_a[rows] = d.share
        turnout2[rows] = d.turnout
        pair_ess[pair] = min(pair_ess.get(pair, math.inf), d.ess / len(rows))
        acc = transfer_acc.setdefault(pair, {})
        for g, name in enumerate(d.groups):
            slot = acc.setdefault(name, {"t": [], "w": []})
            slot["t"].append(d.transfers[:, g, :])
            slot["w"].append(d.weights[:, g])
    win_idx = np.where(share_a > 50.0, a_idx, b_idx)

    # 6. sorties
    draws = pl.DataFrame(
        {
            "draw": np.arange(n_draws),
            "config": cfg,
            "posterior_draw": fr["posterior"],
            "turnout": turnout,
            "turnout_second_round": turnout2,
            "qualified_a": np.array(uni.ids)[a_idx],
            "qualified_b": np.array(uni.ids)[b_idx],
            "share_a_second_round": share_a,
            "winner": np.array(uni.ids)[win_idx],
            **{
                c: pl.Series(c, np.where(present[:, u], 100 * shares[:, u], np.nan)).fill_nan(None)
                for u, c in enumerate(uni.ids)
            },
        }
    )
    summary = summarize_conditions(conditions, cfg, present, shares, a_idx, b_idx, win_idx, uni)
    pairs_df = summarize_pairs(conditions, cfg, pair_code, pair_list, share_a, shares, uni, model,
                               horizon, pair_ess, scale)
    transfers = summarize_transfers(transfer_acc, pairs_df)
    traj = trajectories(fit, uni, enum, conditions, summary, rng, n=trajectory_draws)
    house = house_effects_bloc(fit)

    commit = _git_commit()
    manifest = {
        "kind": "simulate",
        "election": election,
        "asof": asof.isoformat(),
        "first_round": d1.isoformat(),
        "second_round": d2.isoformat(),
        "horizon_days": horizon,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "versions": _versions(),
        "n_draws": n_draws,
        "seed": seed,
        "fit": {
            "dir": str(Path(fit_dir)),
            "n_posterior_draws": fit.n_draws,
            "n_polls": fit.manifest["data"]["n_polls"],
            "n_hypotheses": fit.manifest["data"]["n_hypotheses"],
            "first_poll_date": str(fit.observed["date"].min()),
            "diagnostics": {
                k: fit.manifest["diagnostics"].get(k)
                for k in ("r_hat_max", "ess_min", "divergences", "derived")
            },
            "posterior": fit.manifest.get("posterior"),
            "commit": fit.manifest.get("commit"),
        },
        "scenarios": {
            "as_of": f.as_of.isoformat(),
            "sha256": hashlib.sha256(
                (paths.REGISTRY / "scenarios_2027.yaml").read_bytes()
            ).hexdigest()[:16] if scenarios is None else None,
            "mode": "marginal",
            "n_configurations": len(enum),
            "impossible_mass": enum.impossible_mass,
            "n_top_configurations": n_top,
            "top_mass": float(p_cfg[:n_top].sum()),
            "mass_without_draws": float(p_cfg[fr["config_counts"] == 0].sum()),
        },
        "universe": uni.table(),
        "error_model": error_model.to_dict()["params"] if hasattr(error_model, "to_dict") else None,
        "error_model_nu_scores": getattr(error_model, "info", {}).get("nu_scores"),
        "error_model_extra_elections": getattr(error_model, "info", {}).get("extra_elections"),
        "second_round_model": {
            "sigma0": errors.sigma0,
            "sigma1": errors.sigma1,
            "sigma_house": errors.sigma_house,
            "bias_far_right_calibrated": base_errors.bias_far_right,
            "rn_bias_scale": scale,
            "bias_far_right_applied": errors.bias_far_right,
            "rn_bias_full_days": rn_bias_full_days,
            "rn_bias_zero_days": rn_bias_zero_days,
            "bias_mode": bias_mode,
            "bias_curve_calibrated": [list(x) for x in base_errors.bias_curve],
            "bias_curve_applied": [list(x) for x in errors.bias_curve],
            "couple_far_right": couple_far_right,
            "coupling": None if model.coupling is None else {
                k: v for k, v in model.coupling.items() if k != "pairs"
            },
            "second_round_turnout": second_round_turnout,
            "turnout2": model.turnout2,
            "turnout2_draws": quantile_dict(turnout2),
            "calibrated_on": list(errors.calibrated_on),
            "settings": model.settings.to_dict(),
            "n_pairs": len(pair_list),
            "n_cells": len(cells),
        },
        "turnout": turnout_params,
        "settings": {
            "min_hypotheses": min_hypotheses,
            "sparse_sd": SPARSE_SD,
            "sparse_default_share": SPARSE_DEFAULT_SHARE,
            "config_mass": config_mass,
            "share_rule": getattr(share_rule, "__name__", str(share_rule)),
        },
        "duration_s": round(time.time() - started, 1),
    }
    res = SimulationResult(draws, summary, pairs_df, transfers, traj, house, conditions, manifest)
    if out is not None:
        write_result(res, Path(out))
    return res


# --- résumés


def summarize_conditions(
    conditions: list[Condition],
    cfg: np.ndarray,
    present: np.ndarray,
    shares: np.ndarray,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    win_idx: np.ndarray,
    uni: Universe,
) -> pl.DataFrame:
    """P(présence), P(qualification), P(victoire), quantiles de la part (points, conditionnels
    à la présence) par condition et candidat. Qualifiés et vainqueur en index de l'univers."""
    U = len(uni.ids)
    rows = []
    for cond in conditions:
        sel = np.isin(cfg, cond.configs)
        n = int(sel.sum())
        if n == 0:
            log.warning("condition %s sans tirage", cond.id)
            continue
        p_q = (np.bincount(a_idx[sel], minlength=U) + np.bincount(b_idx[sel], minlength=U)) / n
        p_w = np.bincount(win_idx[sel], minlength=U) / n
        for u, c in enumerate(uni.ids):
            pres = present[sel, u]
            if not pres.any():
                continue
            x = 100 * shares[sel, u][pres]
            rows.append(
                {
                    "condition_id": cond.id,
                    "candidate_id": c,
                    "bloc": uni.blocs[u],
                    "n_draws": n,
                    "p_present": float(pres.mean()),
                    "p_qualify": float(p_q[u]),
                    "p_win": float(p_w[u]),
                    **quantile_dict(x),
                    "mean": float(x.mean()),
                }
            )
    return pl.DataFrame(rows).sort(
        ["condition_id", "p_win", "p_qualify", "q50"], descending=[False, True, True, True]
    )


def summarize_pairs(
    conditions, cfg, pair_code, pair_list, share_a, shares, uni, model, horizon, pair_ess, scale
) -> pl.DataFrame:
    """Par condition et paire : P(paire), P(a gagne | paire), quantiles du score de a."""
    rows = []
    for cond in conditions:
        sel = np.isin(cfg, cond.configs)
        n = int(sel.sum())
        if n == 0:
            continue
        codes, sa, sh = pair_code[sel], share_a[sel], shares[sel]
        for code in np.unique(codes):
            m = codes == code
            a, b = pair_list[code]
            row = {
                "condition_id": cond.id,
                "candidate_a": a,
                "candidate_b": b,
                "n_draws": int(m.sum()),
                "p_pair": float(m.mean()),
                "p_a_wins": float(np.mean(sa[m] > 50.0)),
                **quantile_dict(sa[m]),
                "mean": float(sa[m].mean()),
                "first_round_a": float(100 * sh[m, uni.index(a)].mean()),
                "first_round_b": float(100 * sh[m, uni.index(b)].mean()),
            }
            if cond.id == "mix":
                pf = model.pair_fit((a, b))
                ps = pf.poll_summary(horizon)
                row |= {
                    "n_polls": pf.polls.n,
                    "poll_mean": None if ps is None else ps[0],
                    "poll_sd": None if ps is None else math.sqrt(ps[1]),
                    "bias_applied": None if ps is None else ps[2],
                    "rn_bias_scale": scale,
                    "inedit": pf.inedit,
                    "ess_ratio_min": pair_ess.get((a, b)),
                    "notes": "; ".join(pf.notes),
                }
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["condition_id", "p_pair"], descending=[False, True]
    )


def summarize_transfers(acc: dict, pairs: pl.DataFrame) -> pl.DataFrame:
    """Reports par paire et électorat (mélange) : quantiles en fraction de l'électorat."""
    rows = []
    for (a, b), groups in acc.items():
        for g, slot in groups.items():
            t = np.concatenate(slot["t"])
            w = np.concatenate(slot["w"])
            row = {"candidate_a": a, "candidate_b": b, "group": g, "n": len(t),
                   "weight": float(w.mean())}
            for k, name in enumerate(("to_a", "to_b", "none")):
                q = quantile_dict(t[:, k])
                row |= {f"{name}_{qn}": v for qn, v in q.items()}
            rows.append(row)
    return pl.DataFrame(rows).sort(["candidate_a", "candidate_b", "weight"],
                                   descending=[False, False, True])


def trajectories(
    fit: FitDraws,
    uni: Universe,
    enum: Enumeration,
    conditions: list[Condition],
    summary: pl.DataFrame,
    rng: np.random.Generator,
    n: int | None = None,
    share_rule: ShareRule = iia_shares,
) -> pl.DataFrame:
    """Parts hebdomadaires (points) du soutien latent jusqu'à `asof`, par condition : chaque
    tirage du postérieur reçoit une configuration de la condition et une présence des petits
    candidats ; quantiles conditionnels à la présence, nuls avant la première semaine où le
    candidat est testé (et toujours nuls pour les candidats `bloc` et `prior`)."""
    n = fit.n_draws if n is None else n
    weeks = slice(0, fit.asof_week + 1)
    dates = fit.week_dates[: fit.asof_week + 1]
    mondays = [d - dt.timedelta(days=d.weekday()) for d in dates]
    first = fit.first_week()
    p = np.array([c.probability for c in enum])
    frames = []
    for cond in conditions:
        if cond.id not in set(summary["condition_id"].to_list()):
            continue
        pc = p[cond.configs] / p[cond.configs].sum()
        cfg = rng.choice(cond.configs, size=n, p=pc)
        s_idx = rng.permutation(fit.n_draws)[:n] if n <= fit.n_draws else rng.integers(
            0, fit.n_draws, n)
        present = sample_presence(enum, cfg, uni, rng)
        theta = universe_theta(fit, uni, s_idx, weeks)  # (n, W, U)
        theta = insert_prior_shares(theta, present, uni, rng)
        pr = present[:, None, :]
        ctx = share_context(fit, uni, s_idx, weeks)
        sh = 100 * share_rule(np.where(pr, theta, 0.0), pr, uni.blocs, ctx)
        for u, c in enumerate(uni.ids):
            if not present[:, u].any():
                continue
            if uni.source[u] == "posterior":
                q = np.quantile(sh[present[:, u], :, u], QUANTILES, axis=0)  # (5, W)
                w0 = first.get(c, len(dates))
                q[:, : min(w0, len(dates))] = np.nan
            else:
                q = np.full((len(QUANTILES), len(dates)), np.nan)
            frames.append(
                pl.DataFrame(
                    {
                        "condition_id": cond.id,
                        "candidate_id": c,
                        "week": np.arange(len(dates)),
                        "date": mondays,
                        **{name: q[k] for k, name in enumerate(QNAMES)},
                    }
                ).with_columns(pl.col(list(QNAMES)).fill_nan(None))
            )
    return pl.concat(frames)


def house_effects_bloc(fit: FitDraws) -> pl.DataFrame:
    """Effet maison par institut et bloc à la semaine `asof`, en points : décalage des parts du
    bloc sur la configuration de référence de l'ajustement (positif = surestime le bloc)."""
    ref = [fit.candidates.index(c) for c in fit.manifest["data"]["reference"]]
    blocs = [fit.blocs[i] for i in ref]
    th = fit.theta[:, fit.asof_week][:, ref]
    base = iia_shares(th, np.ones_like(th, dtype=bool), blocs)
    n_polls = (
        fit.observed.group_by("pollster").agg(pl.col("poll_id").n_unique()).iter_rows()
    )
    n_polls = dict(n_polls)
    rows = []
    for j, pol in enumerate(fit.pollsters):
        with_house = iia_shares(th + fit.alpha[:, j][:, ref], np.ones_like(th, dtype=bool), blocs)
        eff = 100 * (with_house - base)
        for b in sorted(set(blocs)):
            m = np.array([x == b for x in blocs])
            rows.append(
                {"pollster": pol, "bloc": b, "n_polls": int(n_polls.get(pol, 0)),
                 **quantile_dict(eff[:, m].sum(-1))}
            )
    return pl.DataFrame(rows)


# --- écriture


def write_result(res: SimulationResult, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    res.draws.write_parquet(out / "draws.parquet")
    res.summary.write_parquet(out / "summary.parquet")
    res.pairs.write_parquet(out / "pairs.parquet")
    res.transfers.write_parquet(out / "transfers.parquet")
    res.trajectories.write_parquet(out / "trajectories.parquet")
    res.house_effects.write_parquet(out / "house_effects.parquet")
    (out / "conditions.json").write_text(
        json.dumps([c.to_dict() for c in res.conditions], indent=1, ensure_ascii=False)
    )
    (out / "manifest.json").write_text(
        json.dumps(res.manifest, indent=2, ensure_ascii=False, default=str)
    )
    return out


def _git_commit() -> str | None:
    from vvs.model.run import _git_commit as commit

    return commit()


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("vvs", "numpy", "polars", "jax", "numpyro", "scipy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def fit_for_simulation(election: int, asof: dt.date, out: Path, **kw) -> dict:
    """Ajustement du premier tour avec les tirages nécessaires à la simulation (variante de
    référence du backtest : plancher d'erreur, `Priors(n_max=2500)`)."""
    from vvs.eval.backtest import DEFAULT_VARIANT, VARIANTS
    from vvs.model.run import run_fit

    kw.setdefault("priors", VARIANTS[DEFAULT_VARIANT])
    return run_fit(election, asof, out, save_draws=True, save_trajectories=True, **kw)

