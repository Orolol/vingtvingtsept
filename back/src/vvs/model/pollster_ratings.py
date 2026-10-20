"""Palmarès des instituts de sondage, à la manière des notes de sondeurs de FiveThirtyEight.

Deux quantités par institut, apprises sur les présidentielles passées et toujours calibrées
sans l'élection évaluée (`PollsterRatings.fit(..., exclude=...)`, argument obligatoire) :

(a) **Prior des effets maison institut × bloc.** Pour chaque élection passée, l'effet maison de
    campagne α_{p,b,e} est estimé par moindres carrés pondérés sur les sondages des
    `CAMPAIGN_DAYS` derniers jours :

        log s_{i,c} = a_i + θ_{c,k(i)} + α_{p(i),b(c)} + ε_{i,c}

    (a_i : normalisation de l'hypothèse i, θ par candidat et tranche de `HOUSE_BIN_DAYS` jours,
    poids s_{i,c}/H_i, soit l'inverse de la variance d'une log-part à un facteur près). α est
    ensuite doublement centré (somme nulle sur les instituts présents et sur les blocs), comme
    dans le modèle (`first_round`) : c'est un écart à l'institut moyen, sur l'échelle logit.
    Modèle à effets aléatoires sur les élections, par (lignée, bloc) :

        α_{p,b,e} = m_{p,b} + u_{p,b,e} + bruit,  m ~ N(0, s_m²),  u ~ N(0, s_e²),
        bruit ~ N(0, se²_{p,b,e})  (se² ∝ 1 / nombre de sondages)

    s_m et s_e sont estimés par maximum de vraisemblance marginale, bloc par bloc (repli sur
    les valeurs communes à tous les blocs sous `MIN_GROUPS` lignées).
    Le prior d'une nouvelle élection est N(μ, V + s_e²), où (μ, V) est le postérieur de m : la
    moyenne passée rétrécie vers 0 selon le nombre d'élections et de sondages. Institut
    inconnu : N(0, s_m² + s_e²), l'institut moyen.

(b) **Facteur de précision** q sur l'effectif effectif n_eff (voir `rated_concentration` :
    q multiplie la précision de l'erreur qui excède l'échantillonnage pur). Sur les sondages des
    `FINAL_WINDOW_DAYS` derniers jours (configuration officielle exacte), erreur
    standardisée z_{i,c} = (part − résultat) / écart-type d'un sondage de 1 000 personnes ;
    on retire l'erreur de l'institut moyen de l'élection (moyenne sur les instituts de leur
    erreur moyenne par candidat : l'erreur commune relève de η, pas de l'institut), puis
    x_{p,e} = log de l'écart quadratique moyen de l'institut, centré sur les instituts de
    l'élection. Même modèle à effets aléatoires (bruit de mesure ≈ 2 / nombre d'erreurs) ;
    facteur q_p = exp(−μ_p) borné à `PRECISION_BOUNDS`. Comme dans les notes de
    FiveThirtyEight, la précision est jugée contre le résultat, biais maison relatif compris.
    Institut inconnu : q = 1.

Successions. Un changement de nom (Sofres → Kantar → Verian, chaîne `successor` de
`pollsters.yaml`) prolonge la même maison : l'historique est mis en commun sous la lignée
terminale (`verian`). Un rachat (BVA par Ipsos en 2024, `ABSORBED`) ne transmet pas
l'historique : les méthodes de l'acquéreur priment, BVA garde son propre historique et les
sondages « Ipsos bva » sont notés comme Ipsos.

Intégration au modèle de premier tour (`first_round.Priors.pollster_ratings`, désactivé par
défaut) : `ratings_inputs` fournit à `ModelInputs` les tableaux (μ, sd) (P, B) et q (P,),
`rated_house_bloc` remplace σ_house · z par μ + sd · z (puis double centrage), et
`rated_concentration` applique q à la part de 1/n_eff qui excède l'erreur
d'échantillonnage pure : 1/n_eff = 1/n + (1/(f·n) + 1/n_max − 1/n) / q.
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from vvs.model.data import election_date, load_polls
from vvs.registry import BLOCS, Registry, default_registry

log = logging.getLogger(__name__)

ELECTIONS = (2002, 2007, 2012, 2017, 2022)
FINAL_WINDOW_DAYS = 21
CAMPAIGN_DAYS = 180
HOUSE_BIN_DAYS = 14
MIN_SHARE = 0.005  # parts plus petites écartées de la régression des effets maison (arrondis)
REFERENCE_N = 1000  # z est l'erreur en écarts-types d'un sondage aléatoire de 1 000 personnes
PRECISION_BOUNDS = (0.5, 1.5)
MIN_GROUPS = 5  # lignées minimum pour estimer les composantes de variance d'un bloc
# Rachats : le successeur ne reprend pas l'historique de l'institut absorbé.
ABSORBED = frozenset({"bva"})


# --- successions


def lineage(pollster: str, registry: Registry | None = None) -> str:
    """Lignée d'un institut : terminus de la chaîne des changements de nom (`successor`),
    sans traverser un rachat (`ABSORBED`)."""
    reg = registry or default_registry()
    seen = {pollster}
    p = pollster
    while p not in ABSORBED and p in reg.pollsters and reg.pollsters[p].successor:
        p = reg.pollsters[p].successor
        if p in seen:  # garde-fou : cycle dans le registre
            raise ValueError(f"cycle de successeurs à partir de {pollster}")
        seen.add(p)
    return p


# --- données par élection


def _official(reg: Registry, election: int) -> dict[str, float]:
    e = reg.elections[election]
    return {r.candidate_id: r.share for r in e.round_(1).results}


def final_poll_errors(
    election: int, registry: Registry | None = None, window_days: int = FINAL_WINDOW_DAYS
) -> pl.DataFrame:
    """Erreurs des sondages des `window_days` derniers jours, une ligne par sondage × candidat.

    Sondages de `load_polls` (suspects exclus, doublons fusionnés, vagues de rolling
    éclaircies) terminés avant le vote ; une seule hypothèse par sondage, celle qui contient
    exactement les candidats officiels (les sondages sans elle sont écartés). Colonnes :
    election, pollster, poll_id, fieldwork_end, candidate_id, bloc, share, result (points),
    error (points), z (erreur standardisée, voir le docstring du module)."""
    reg = registry or default_registry()
    vote = election_date(reg, election)
    off = _official(reg, election)
    df = load_polls(election, vote - dt.timedelta(days=1))
    df = df.filter(pl.col("fieldwork_end") >= vote - dt.timedelta(days=window_days))
    rows = []
    seen: set[str] = set()
    for (poll_id, _hyp), grp in df.group_by(["poll_id", "hypothesis_id"], maintain_order=True):
        if poll_id in seen or set(grp["candidate_id"].to_list()) != set(off):
            continue
        seen.add(poll_id)
        for r in grp.iter_rows(named=True):
            res = off[r["candidate_id"]]
            p = res / 100
            sd = 100 * math.sqrt(p * (1 - p) / REFERENCE_N)
            rows.append(
                {
                    "election": election,
                    "pollster": r["pollster"],
                    "poll_id": poll_id,
                    "fieldwork_end": r["fieldwork_end"],
                    "candidate_id": r["candidate_id"],
                    "bloc": reg.bloc_of(r["candidate_id"], election),
                    "share": float(r["share"]),
                    "result": float(res),
                    "error": float(r["share"] - res),
                    "z": float((r["share"] - res) / sd),
                }
            )
    return pl.DataFrame(rows)


def _lsqr_house(
    hyp: np.ndarray, cell: np.ndarray, house: np.ndarray, y: np.ndarray, w: np.ndarray,
    n_hyp: int, n_cell: int, n_house: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Moindres carrés pondérés y = a[hyp] + θ[cell] + α[house] ; rend (α, résidus)."""
    from scipy.sparse import csr_matrix, hstack
    from scipy.sparse.linalg import lsqr

    n = len(y)
    rows = np.arange(n)
    ones = np.ones(n)
    X = hstack(
        [
            csr_matrix((ones, (rows, hyp)), shape=(n, n_hyp)),
            csr_matrix((ones, (rows, cell)), shape=(n, n_cell)),
            csr_matrix((ones, (rows, house)), shape=(n, n_house)),
        ]
    ).tocsr()
    sw = np.sqrt(w)
    Xw = X.multiply(sw[:, None]).tocsr()
    sol = lsqr(Xw, y * sw, atol=1e-10, btol=1e-10, iter_lim=20_000)[0]
    return sol[n_hyp + n_cell :], y - X @ sol


def campaign_house_effects(
    election: int,
    registry: Registry | None = None,
    days: int = CAMPAIGN_DAYS,
    bin_days: int = HOUSE_BIN_DAYS,
) -> pl.DataFrame:
    """Effets maison de campagne institut × bloc d'une élection (échelle logit, doublement
    centrés), avec leur erreur type ; voir le docstring du module. Colonnes : election,
    pollster, bloc, alpha, se, n_polls."""
    reg = registry or default_registry()
    vote = election_date(reg, election)
    df = load_polls(election, vote - dt.timedelta(days=1))
    df = df.with_columns(
        (pl.col("fieldwork_start") + (pl.col("fieldwork_end") - pl.col("fieldwork_start")) / 2)
        .alias("mid")
    ).filter((pl.col("mid") >= vote - dt.timedelta(days=days)) & (pl.col("mid") < vote))
    blocs: dict[str, str | None] = {}
    for c in df["candidate_id"].unique().to_list():
        try:
            blocs[c] = reg.bloc_of(c, election)
        except KeyError:
            blocs[c] = None
    df = df.filter(
        pl.col("candidate_id").is_in([c for c, b in blocs.items() if b is not None])
        & (pl.col("share") / 100 >= MIN_SHARE)
    )
    df = df.with_columns(
        pl.col("candidate_id").replace_strict(blocs, return_dtype=pl.Utf8).alias("bloc"),
        ((vote - pl.col("mid")).dt.total_days() // bin_days).alias("bin"),
        (pl.col("poll_id") + "|" + pl.col("hypothesis_id")).alias("hyp"),
    )
    # hypothèses d'au moins deux candidats, poids 1/H par sondage
    df = df.join(df.group_by("hyp").len("n_c"), on="hyp").filter(pl.col("n_c") >= 2)
    df = df.join(
        df.select("poll_id", "hyp").unique().group_by("poll_id").len("H"), on="poll_id"
    )
    if df.is_empty():
        return pl.DataFrame()
    hyps = sorted(df["hyp"].unique().to_list())
    cells = sorted({(c, k) for c, k in df.select("candidate_id", "bin").iter_rows()})
    pollsters = sorted(df["pollster"].unique().to_list())
    present = [b for b in BLOCS if b in set(df["bloc"].to_list())]
    hi = {h: i for i, h in enumerate(hyps)}
    ci = {c: i for i, c in enumerate(cells)}
    P, B = len(pollsters), len(present)
    hyp_idx = np.array([hi[h] for h in df["hyp"]])
    cell_idx = np.array([ci[(c, k)] for c, k in df.select("candidate_id", "bin").iter_rows()])
    p_idx = np.array([pollsters.index(p) for p in df["pollster"]])
    b_idx = np.array([present.index(b) for b in df["bloc"]])
    share = df["share"].to_numpy() / 100
    y = np.log(share)
    w = share / df["H"].to_numpy()
    alpha, resid = _lsqr_house(
        hyp_idx, cell_idx, p_idx * B + b_idx, y, w, len(hyps), len(cells), P * B
    )
    alpha = alpha.reshape(P, B)
    # Instituts × blocs jamais observés : pas d'estimation (le moindre-norme les met à 0).
    seen = np.zeros((P, B), dtype=bool)
    seen[p_idx, b_idx] = True
    alpha = _double_center_observed(alpha, seen)
    # Erreur type : variance des résidus moyens de bloc par hypothèse, divisée par le nombre
    # de sondages de l'institut qui testent le bloc.
    d = pl.DataFrame(
        {"hyp": df["hyp"], "poll_id": df["poll_id"], "p": p_idx, "b": b_idx, "r": resid, "w": w}
    )
    hb = d.group_by("hyp", "poll_id", "p", "b").agg(
        ((pl.col("r") * pl.col("w")).sum() / pl.col("w").sum()).alias("rb")
    )
    var_b = hb.group_by("b").agg((pl.col("rb") ** 2).mean().alias("v"))
    var_b = dict(zip(var_b["b"].to_list(), var_b["v"].to_list()))
    npolls = hb.group_by("p", "b").agg(pl.col("poll_id").n_unique().alias("n"))
    npolls = {(p, b): n for p, b, n in npolls.iter_rows()}
    rows = []
    for p in range(P):
        for b in range(B):
            if not seen[p, b]:
                continue
            n = npolls.get((p, b), 0)
            rows.append(
                {
                    "election": election,
                    "pollster": pollsters[p],
                    "bloc": present[b],
                    "alpha": float(alpha[p, b]),
                    "se": float(math.sqrt(var_b[b] / max(n, 1))),
                    "n_polls": int(n),
                }
            )
    return pl.DataFrame(rows)


def _double_center_observed(a: np.ndarray, seen: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """Double centrage (lignes puis colonnes, itéré) sur les seules cases observées."""
    a = np.where(seen, a, 0.0)
    for _ in range(n_iter):
        cnt = np.maximum(seen.sum(0), 1)
        a = np.where(seen, a - (a * seen).sum(0) / cnt, 0.0)
        cnt = np.maximum(seen.sum(1), 1)
        a = np.where(seen, a - ((a * seen).sum(1) / cnt)[:, None], 0.0)
    return a


@functools.lru_cache(maxsize=16)
def election_tables(election: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(erreurs finales, effets maison de campagne) d'une élection, registre par défaut."""
    return final_poll_errors(election), campaign_house_effects(election)


def relative_accuracy(errors: pl.DataFrame) -> pl.DataFrame:
    """Statistique de précision x_{p,e} par institut et élection (voir le docstring du module).

    Colonnes : election, pollster, n_polls, n_errors, rms_z (écart quadratique moyen de z
    autour de l'institut moyen), x (log rms² centré sur les instituts de l'élection)."""
    if errors.is_empty():
        return pl.DataFrame()
    avg = (
        errors.group_by("election", "pollster", "candidate_id")
        .agg(pl.col("z").mean().alias("zp"))
        .group_by("election", "candidate_id")
        .agg(pl.col("zp").mean().alias("zbar"))
    )
    dev = errors.join(avg, on=["election", "candidate_id"]).with_columns(
        (pl.col("z") - pl.col("zbar")).alias("dev")
    )
    out = dev.group_by("election", "pollster").agg(
        pl.col("poll_id").n_unique().alias("n_polls"),
        pl.len().alias("n_errors"),
        (pl.col("dev") ** 2).mean().alias("ms"),
    )
    out = out.with_columns(pl.col("ms").sqrt().alias("rms_z"), pl.col("ms").log().alias("lx"))
    out = out.with_columns((pl.col("lx") - pl.col("lx").mean().over("election")).alias("x"))
    return out.drop("ms", "lx").sort("election", "pollster")


# --- effets aléatoires


def fit_random_effects(
    groups: list[tuple[np.ndarray, np.ndarray]], floor: float = 1e-6
) -> tuple[float, float]:
    """(s_m², s_e²) par maximum de vraisemblance marginale de y_g ~ N(0, s_m² 11ᵀ + diag(s_e²
    + se²)), une paire (y, se²) par groupe. Rend (0, s_e²) si rien n'est persistant."""
    from scipy.optimize import minimize

    groups = [(np.asarray(y, float), np.asarray(v, float)) for y, v in groups if len(y)]
    if not groups:
        return floor, floor

    def nll(theta):
        sm2, se2 = np.exp(theta)
        tot = 0.0
        for y, v in groups:
            d = se2 + v
            # Σ = D + sm2 11ᵀ : inverse et déterminant par Sherman-Morrison
            s = (1 / d).sum()
            q = (y**2 / d).sum() - sm2 * (y / d).sum() ** 2 / (1 + sm2 * s)
            logdet = np.log(d).sum() + np.log1p(sm2 * s)
            tot += 0.5 * (q + logdet)
        return tot

    allv = np.concatenate([y for y, _ in groups])
    v0 = max(float(np.var(allv)), 1e-4)
    best = None
    for start in ((v0 / 2, v0 / 2), (v0 / 10, v0), (v0, v0 / 10)):
        r = minimize(nll, np.log(start), method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    sm2, se2 = np.exp(best.x)
    return float(max(sm2, floor)), float(max(se2, floor))


def shrunk_mean(y: np.ndarray, v: np.ndarray, sm2: float, se2: float) -> tuple[float, float]:
    """Postérieur (moyenne, variance) de m pour un groupe : rétrécissement vers 0."""
    prec = 1 / sm2 + (1 / (se2 + np.asarray(v))).sum()
    return float((np.asarray(y) / (se2 + np.asarray(v))).sum() / prec), float(1 / prec)


# --- palmarès


@dataclass
class PollsterRatings:
    """Notes apprises sur `elections` (hors `excluded`). `house` : une ligne par (lignée, bloc)
    observée (mu, sd du prior de l'effet maison d'une nouvelle élection) ; `precision` : une
    ligne par lignée (x_mean, factor)."""

    elections: tuple[int, ...]
    excluded: tuple[int, ...]
    house: pl.DataFrame
    precision: pl.DataFrame
    house_sm: float  # écart-type de l'effet maison persistant
    house_se: float  # écart-type de la variation d'une élection à l'autre
    prec_sm: float
    prec_se: float
    bounds: tuple[float, float] = PRECISION_BOUNDS
    lineages: dict[str, str] = field(default_factory=dict)
    house_bloc_var: dict[str, tuple[float, float]] = field(default_factory=dict)  # (s_m, s_e)

    @classmethod
    def fit(
        cls,
        elections=ELECTIONS,
        *,
        exclude,
        registry: Registry | None = None,
        bounds: tuple[float, float] = PRECISION_BOUNDS,
        tables: dict[int, tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    ) -> PollsterRatings:
        """Notes calibrées sur `elections` privées de `exclude` (obligatoire : l'élection
        évaluée, ou () pour une élection à venir). `tables` : {élection: (erreurs finales,
        effets maison)} à la place des données curated (tests)."""
        reg = registry or default_registry()
        excluded = tuple(sorted({int(e) for e in exclude}))
        train = tuple(e for e in elections if e not in excluded)
        if tables is None:
            tables = {e: election_tables(e) for e in train}
        errs = [tables[e][0] for e in train if not tables[e][0].is_empty()]
        houses = [tables[e][1] for e in train if not tables[e][1].is_empty()]
        lin = functools.partial(lineage, registry=reg)

        # (a) effets maison
        house_rows = []
        sm2 = se2 = 1e-6
        bloc_var: dict[str, tuple[float, float]] = {}
        if houses:
            h = pl.concat(houses, how="diagonal_relaxed")
            h = h.with_columns(
                pl.col("pollster").map_elements(lin, return_dtype=pl.Utf8).alias("lineage")
            )
            groups = {
                k: (g["alpha"].to_numpy(), g["se"].to_numpy() ** 2, int(g["n_polls"].sum()),
                    g["election"].n_unique())
                for k, g in sorted(h.group_by("lineage", "bloc"), key=lambda kg: kg[0])
            }
            sm2, se2 = fit_random_effects([(y, v) for y, v, _, _ in groups.values()])
            # Composantes de variance par bloc : la persistance diffère beaucoup d'un bloc à
            # l'autre (nulle à droite et à l'extrême droite, nette à la gauche radicale).
            for b in sorted({b for _, b in groups}):
                gb = [(y, v) for (_, bb), (y, v, _, _) in groups.items() if bb == b]
                bloc_var[b] = fit_random_effects(gb) if len(gb) >= MIN_GROUPS else (sm2, se2)
            for (lname, bloc), (y, v, n, ne) in sorted(groups.items()):
                bm2, be2 = bloc_var[bloc]
                mu, var = shrunk_mean(y, v, bm2, be2)
                house_rows.append(
                    {"lineage": lname, "bloc": bloc, "mu": mu, "sd": math.sqrt(var + be2),
                     "raw_mean": float(np.mean(y)), "n_elections": ne, "n_polls": n}
                )
        # (b) précision
        prec_rows = []
        pm2 = pe2 = 1e-6
        if errs:
            acc = relative_accuracy(pl.concat(errs, how="diagonal_relaxed"))
            acc = acc.with_columns(
                pl.col("pollster").map_elements(lin, return_dtype=pl.Utf8).alias("lineage")
            )
            groups = {
                k[0]: (g["x"].to_numpy(), 2.0 / g["n_errors"].to_numpy(),
                       int(g["n_polls"].sum()), g["election"].n_unique())
                for k, g in sorted(acc.group_by("lineage"), key=lambda kg: kg[0])
            }
            pm2, pe2 = fit_random_effects([(y, v) for y, v, _, _ in groups.values()])
            for lname, (y, v, n, ne) in sorted(groups.items()):
                mu, var = shrunk_mean(y, v, pm2, pe2)
                prec_rows.append(
                    {"lineage": lname, "x_mean": mu, "x_sd": math.sqrt(var),
                     "raw_mean": float(np.mean(y)),
                     "factor": float(np.clip(math.exp(-mu), *bounds)),
                     "n_elections": ne, "n_polls": n}
                )
        lineages = {p: lin(p) for p in reg.pollsters}
        return cls(
            elections=train,
            excluded=excluded,
            house=pl.DataFrame(house_rows),
            precision=pl.DataFrame(prec_rows),
            house_sm=math.sqrt(sm2),
            house_se=math.sqrt(se2),
            house_bloc_var={b: (math.sqrt(m), math.sqrt(e)) for b, (m, e) in bloc_var.items()},
            prec_sm=math.sqrt(pm2),
            prec_se=math.sqrt(pe2),
            bounds=bounds,
            lineages=lineages,
        )

    def _lineage(self, pollster: str) -> str:
        return self.lineages.get(pollster, pollster)

    def house_prior(self, pollsters: list[str], blocs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(μ, sd) (P, B) du prior des effets maison ; institut ou bloc jamais vu : (0, sd de
        l'institut moyen). μ n'est pas recentré : le modèle double-centre."""
        pooled = (self.house_sm, self.house_se)
        sd0 = [math.hypot(*self.house_bloc_var.get(b, pooled)) for b in blocs]
        mu = np.zeros((len(pollsters), len(blocs)))
        sd = np.tile(np.array(sd0, dtype=float), (len(pollsters), 1))
        if self.house.is_empty():
            return mu, sd
        tab = {(r["lineage"], r["bloc"]): r for r in self.house.iter_rows(named=True)}
        for i, p in enumerate(pollsters):
            for j, b in enumerate(blocs):
                r = tab.get((self._lineage(p), b))
                if r is not None:
                    mu[i, j], sd[i, j] = r["mu"], r["sd"]
        return mu, sd

    def precision_factor(self, pollsters: list[str]) -> np.ndarray:
        """q (P,) ; institut inconnu : 1."""
        q = np.ones(len(pollsters))
        if self.precision.is_empty():
            return q
        tab = dict(zip(self.precision["lineage"].to_list(), self.precision["factor"].to_list()))
        for i, p in enumerate(pollsters):
            q[i] = tab.get(self._lineage(p), 1.0)
        return q

    def arrays(self, pollsters: list[str], blocs: list[str]) -> dict:
        mu, sd = self.house_prior(pollsters, blocs)
        return {
            "mu": mu,
            "sd": sd,
            "q": self.precision_factor(pollsters),
            "pollsters": list(pollsters),
            "blocs": list(blocs),
            "excluded": list(self.excluded),
        }

    def to_dict(self) -> dict:
        return {
            "elections": list(self.elections),
            "excluded": list(self.excluded),
            "house_sm": self.house_sm,
            "house_se": self.house_se,
            "house_bloc_var": {b: list(v) for b, v in self.house_bloc_var.items()},
            "prec_sm": self.prec_sm,
            "prec_se": self.prec_se,
            "bounds": list(self.bounds),
            "house": self.house.to_dicts(),
            "precision": self.precision.to_dicts(),
        }


@functools.lru_cache(maxsize=8)
def fitted_ratings(exclude: tuple[int, ...]) -> PollsterRatings:
    """Notes calibrées sur `ELECTIONS` hors `exclude` (registre et données par défaut)."""
    return PollsterRatings.fit(ELECTIONS, exclude=exclude)


def attach_ratings(data, out: Path | None = None) -> dict:
    """Calcule les notes sans l'élection de `data` et les attache (`data.pollster_ratings`) ;
    les écrit dans `out/pollster_ratings.json` si `out` est donné."""
    r = fitted_ratings((int(data.election),))
    arr = r.arrays(data.pollsters, data.blocs)
    data.pollster_ratings = arr
    if out is not None:
        payload = r.to_dict() | {
            "applied": {
                "pollsters": arr["pollsters"],
                "blocs": arr["blocs"],
                "mu": arr["mu"].tolist(),
                "sd": arr["sd"].tolist(),
                "q": arr["q"].tolist(),
            }
        }
        (Path(out) / "pollster_ratings.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        )
    return arr


# --- branchement dans le modèle (jax)


def ratings_inputs(data, priors) -> dict:
    """Champs de `ModelInputs` pour l'option `Priors.pollster_ratings` ({} sans l'option).

    Utilise `data.pollster_ratings` (attaché par `run_fit`) ou, à défaut, calcule les notes
    sans l'élection de `data`."""
    if priors is None or not getattr(priors, "pollster_ratings", False):
        return {}
    import jax.numpy as jnp

    arr = getattr(data, "pollster_ratings", None)
    if arr is None:
        arr = fitted_ratings((int(data.election),)).arrays(data.pollsters, data.blocs)
    if list(arr["pollsters"]) != list(data.pollsters) or list(arr["blocs"]) != list(data.blocs):
        raise ValueError("notes d'instituts non alignées sur les instituts ou blocs des données")
    return {
        "rating_mu": jnp.asarray(arr["mu"]),
        "rating_sd": jnp.asarray(arr["sd"]),
        "rating_q": jnp.asarray(arr["q"]),
    }


def _plain_double_center(x):
    x = x - x.mean(axis=0, keepdims=True)
    return x - x.mean(axis=1, keepdims=True)


def _scaled_center(x, axis):
    import jax.numpy as jnp

    n = x.shape[axis]
    if n < 2:
        return jnp.zeros_like(x)
    return (x - x.mean(axis=axis, keepdims=True)) * jnp.sqrt(n / (n - 1))


def rated_house_bloc(z, inp):
    """Effets maison de bloc (P, B) sous le prior appris : μ doublement centré + sd · z
    doublement centré et rééchelonné (même convention que `first_round.latent`)."""
    mu = _plain_double_center(inp.rating_mu)
    return mu + _scaled_center(_scaled_center(inp.rating_sd * z, 0), 1)


def rated_concentration(f, n, n_max, q):
    """κ de la Dirichlet-multinomiale quand l'institut a le facteur de précision q.

    q multiplie la précision de l'erreur excédentaire, celle qui s'ajoute à la multinomiale
    d'effectif n (redressement, cadre, plancher n_max) : 1/n_eff = 1/n + (1/n_eff° − 1/n)/q,
    n_eff° étant l'effectif effectif sans note (`first_round.effective_size`). En r = n/n_eff,
    r = 1 + (r° − 1)/q : r reste au-dessus de 1 pour tout q > 0 (une Dirichlet-multinomiale
    ne peut pas être moins dispersée qu'une multinomiale) et la fonction reste lisse. Un
    institut ne peut pas battre l'erreur d'échantillonnage pure. q = 1 redonne
    `first_round.concentration`."""
    import jax.numpy as jnp

    r0 = 1.0 / f if n_max is None else 1.0 / f + n / n_max
    r = 1.0 + (r0 - 1.0) / q
    return jnp.maximum(n - r, 1e-3) / (r - 1.0)
