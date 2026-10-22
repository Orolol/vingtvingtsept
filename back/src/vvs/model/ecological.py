"""Inférence écologique des reports de voix entre les deux tours, sur les communes.

Pour une commune `c`, `X_c` est la répartition de ses inscrits au premier tour entre les
électorats (candidats ou groupes de candidats, blancs et nuls, abstention) et `Y_c` celle de
ses inscrits au second tour entre les deux finalistes, les blancs et nuls et l'abstention.
Le modèle de Goodman suppose une matrice de reports `B` (une ligne par électorat, sommant à 1)
commune à toutes les communes :

    Y_c = X_c B + e_c

et l'estime par moindres carrés pondérés par les inscrits, sous les contraintes
`0 ≤ B ≤ 1` et lignes sommant à 1 (`goodman`). Le problème ne dépend des données qu'à travers
G = Xᵀ W X et H = Xᵀ W Y : le bootstrap de communes (`bootstrap`) ne refait que ces produits.

Version stratifiée (`stratified`) : une matrice par strate de communes (par exemple quintile
de revenu × type urbain), agrégée au national en pondérant chaque strate par la taille de
l'électorat dans la strate. Elle réduit le biais d'agrégation quand les reports varient avec
la sociologie de la commune, sans le supprimer.

Limite à garder en tête : le vote d'un électorat n'est jamais observé, seule la corrélation
géographique entre la part de cet électorat au premier tour et les parts du second tour l'est.
Si les électeurs d'un candidat votent différemment selon la commune où ils vivent (biais
d'agrégation, « ecological fallacy »), et que cette différence est corrélée à la composition
de la commune, les estimations sont biaisées, et le bootstrap de communes n'en dit rien : il
ne mesure que la variabilité d'échantillonnage des communes.

Appariement avec les indicateurs socio-économiques (`match_communes`) : le code INSEE de la
commune à la date de l'élection (results_<année>.parquet) est apparié tel quel à la
géographie de référence de communes_socio.parquet (communes de métropole, géographie récente,
voir vvs.ingest.conflit_politique). Règle : une commune électorale dont le code n'existe plus
dans la géographie de référence (fusionnée depuis dans une commune nouvelle) est exclue, faute
de table de passage ; le chef-lieu d'une commune nouvelle, qui a gardé son code, reçoit les
indicateurs de la commune nouvelle entière (approximation : ce sont des parts et des ratios).
L'outre-mer et l'étranger (codes 97x, 98x, Z*) n'ont pas d'indicateurs et sont exclus.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.optimize import minimize

from vvs import paths
from vvs.registry import Registry, default_registry

BLANK_NULL = "blank_null"
ABSTAIN = "abstain"
SPECIAL = (BLANK_NULL, ABSTAIN)
# Seuil (% des exprimés au premier tour) au-dessus duquel un candidat forme son propre électorat
MIN_SHARE = 4.0

_PLM_ARR = re.compile(r"^(751(0[1-9]|1[0-9]|20)|132(0[1-9]|1[0-6])|6938[1-9])$")
_OVERSEAS = re.compile(r"^(9[78]|Z)")


# --- électorats


@dataclass(frozen=True)
class Groups:
    """Électorats de premier tour d'une élection : id -> candidats membres.

    Les ids sont des candidate_id quand le groupe est un candidat seul, `blank_null` et
    `abstain` pour les électorats spéciaux (membres vides), un id de groupe sinon.
    """

    election: int
    finalists: tuple[str, str]
    members: dict[str, tuple[str, ...]]

    @property
    def ids(self) -> list[str]:
        return list(self.members)

    @property
    def outcomes(self) -> list[str]:
        return [*self.finalists, BLANK_NULL, ABSTAIN]


def default_groups(
    election: int, registry: Registry | None = None, min_share: float = MIN_SHARE
) -> Groups:
    """Finalistes et candidats d'au moins `min_share` % seuls, les autres regroupés par bloc
    (`small_<bloc>`), puis blancs et nuls et abstention."""
    reg = registry or default_registry()
    e = reg.elections[election]
    shares = {r.candidate_id: r.share for r in e.round_(1).results}
    fin = tuple(r.candidate_id for r in e.round_(2).results)
    members: dict[str, tuple[str, ...]] = {}
    small: dict[str, list[str]] = {}
    for c in sorted(e.candidates, key=lambda c: -shares.get(c.candidate_id, 0.0)):
        cid = c.candidate_id
        if cid in fin or shares.get(cid, 0.0) >= min_share:
            members[cid] = (cid,)
        else:
            small.setdefault(c.bloc, []).append(cid)
    for bloc, cids in small.items():
        members[f"small_{bloc}"] = tuple(cids)
    members[BLANK_NULL] = ()
    members[ABSTAIN] = ()
    return Groups(election, fin, members)  # type: ignore[arg-type]


def custom_groups(election: int, spec: Mapping[str, Sequence[str]], registry: Registry | None = None) -> Groups:
    """Groupes donnés explicitement ; chaque candidat de l'élection doit être dans un groupe."""
    reg = registry or default_registry()
    e = reg.elections[election]
    fin = tuple(r.candidate_id for r in e.round_(2).results)
    members = {k: tuple(v) for k, v in spec.items() if k not in SPECIAL}
    seen = [c for v in members.values() for c in v]
    cands = {c.candidate_id for c in e.candidates}
    if sorted(seen) != sorted(cands):
        raise ValueError(f"{election} : groupes incomplets ou en double ({set(cands) ^ set(seen)})")
    members[BLANK_NULL] = ()
    members[ABSTAIN] = ()
    return Groups(election, fin, members)  # type: ignore[arg-type]


# --- données communales


def load_results(election: int) -> pl.DataFrame:
    path = paths.CURATED / f"results_{election}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : `vvs parse interieur --election {election}`")
    return pl.read_parquet(path).filter(pl.col("geo_level") == "commune")


def commune_counts(results: pl.DataFrame, groups: Groups) -> pl.DataFrame:
    """Une ligne par commune : inscrits et voix par électorat aux deux tours.

    Colonnes : geo_code, registered1, x_<groupe>… (voix au premier tour, blancs et nuls =
    votants − exprimés, abstention = inscrits − votants), registered2, y_<issue>… Les communes
    sans inscrits à l'un des tours sont écartées.
    """
    to_group = {c: g for g, cs in groups.members.items() for c in cs}
    r1 = results.filter(pl.col("round") == 1)
    unknown = set(r1["candidate_id"].unique()) - set(to_group)
    if unknown:
        raise ValueError(f"{groups.election} : candidats sans groupe {sorted(unknown)}")
    tot1 = r1.group_by("geo_code").agg(
        pl.col("registered").first().alias("registered1"),
        pl.col("voters").first().alias("_v1"),
        pl.col("expressed").first().alias("_e1"),
    )
    x = (
        r1.with_columns(pl.col("candidate_id").replace_strict(to_group).alias("g"))
        .group_by("geo_code", "g")
        .agg(pl.col("votes").sum())
        .pivot(on="g", index="geo_code", values="votes")
    )
    cand_groups = [g for g in groups.ids if g not in SPECIAL]
    for g in cand_groups:
        if g not in x.columns:
            x = x.with_columns(pl.lit(0).alias(g))
    x = x.rename({g: f"x_{g}" for g in cand_groups})
    r2 = results.filter(pl.col("round") == 2)
    tot2 = r2.group_by("geo_code").agg(
        pl.col("registered").first().alias("registered2"),
        pl.col("voters").first().alias("_v2"),
        pl.col("expressed").first().alias("_e2"),
    )
    y = r2.pivot(on="candidate_id", index="geo_code", values="votes", aggregate_function="sum")
    y = y.rename({f: f"y_{f}" for f in groups.finalists})
    df = (
        tot1.join(x, on="geo_code")
        .join(tot2, on="geo_code")
        .join(y, on="geo_code")
        .with_columns(
            (pl.col("_v1") - pl.col("_e1")).alias(f"x_{BLANK_NULL}"),
            (pl.col("registered1") - pl.col("_v1")).alias(f"x_{ABSTAIN}"),
            (pl.col("_v2") - pl.col("_e2")).alias(f"y_{BLANK_NULL}"),
            (pl.col("registered2") - pl.col("_v2")).alias(f"y_{ABSTAIN}"),
        )
        .drop("_v1", "_e1", "_v2", "_e2")
        .fill_null(0)
        .filter((pl.col("registered1") > 0) & (pl.col("registered2") > 0))
    )
    return df.sort("geo_code")


def match_status(code: str, socio_codes: set[str]) -> str:
    """Statut d'appariement d'un code commune électoral : matched, overseas_or_abroad,
    plm_arrondissement, merged_or_unknown (voir la règle en tête de module)."""
    if _OVERSEAS.match(code):
        return "overseas_or_abroad"
    if _PLM_ARR.match(code):
        return "plm_arrondissement"
    return "matched" if code in socio_codes else "merged_or_unknown"


def match_communes(frame: pl.DataFrame, socio_codes: set[str]) -> pl.DataFrame:
    """Ajoute la colonne `match` (statut d'appariement) à une table indexée par geo_code."""
    return frame.with_columns(
        pl.col("geo_code")
        .map_elements(lambda c: match_status(c, socio_codes), return_dtype=pl.Utf8)
        .alias("match")
    )


def coverage(frame: pl.DataFrame) -> pl.DataFrame:
    """Communes et inscrits (premier tour) par statut d'appariement."""
    total = frame["registered1"].sum()
    return (
        frame.group_by("match")
        .agg(pl.len().alias("communes"), pl.col("registered1").sum().alias("registered"))
        .with_columns((pl.col("registered") / total).alias("registered_share"))
        .sort("match")
    )


def matrices(frame: pl.DataFrame, groups: Groups) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, Y, w) : parts des inscrits de chaque tour par électorat et par issue, poids =
    inscrits du premier tour."""
    xc = [f"x_{g}" for g in groups.ids]
    yc = [f"y_{o}" for o in groups.outcomes]
    r1 = frame["registered1"].to_numpy().astype(float)
    r2 = frame["registered2"].to_numpy().astype(float)
    X = frame.select(xc).to_numpy().astype(float) / r1[:, None]
    Y = frame.select(yc).to_numpy().astype(float) / r2[:, None]
    return X, Y, r1


# --- estimation


def sufficient_stats(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """G = Xᵀ W X, H = Xᵀ W Y, poids normalisés à une somme de 1."""
    w = w / w.sum()
    Xw = X * w[:, None]
    return X.T @ Xw, Xw.T @ Y


def project_simplex_rows(B: np.ndarray) -> np.ndarray:
    """Projection euclidienne de chaque ligne sur le simplexe (Duchi et al., 2008)."""
    out = np.empty_like(B)
    for k, v in enumerate(B):
        u = np.sort(v)[::-1]
        css = np.cumsum(u) - 1.0
        idx = np.arange(1, len(v) + 1)
        rho = np.nonzero(u - css / idx > 0)[0][-1]
        theta = css[rho] / (rho + 1.0)
        out[k] = np.maximum(v - theta, 0.0)
    return out


def solve_constrained(G: np.ndarray, H: np.ndarray, start: np.ndarray | None = None,
                      tol: float = 1e-12, max_iter: int = 2000) -> np.ndarray:
    """min tr(Bᵀ G B) − 2 tr(Bᵀ H) sous B ≥ 0 et B 1 = 1 (SLSQP, gradient exact).

    Point de départ : la solution sans contrainte projetée sur le simplexe."""
    K, J = H.shape
    if start is None:
        try:
            start = np.linalg.lstsq(G, H, rcond=None)[0]
        except np.linalg.LinAlgError:
            start = np.full((K, J), 1.0 / J)
        start = project_simplex_rows(start)

    def f(b: np.ndarray) -> tuple[float, np.ndarray]:
        B = b.reshape(K, J)
        GB = G @ B
        val = float(np.sum(B * GB) - 2.0 * np.sum(B * H))
        return val, (2.0 * (GB - H)).ravel()

    A = np.kron(np.eye(K), np.ones((1, J)))
    res = minimize(
        f, start.ravel(), jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * (K * J),
        constraints=[{"type": "eq", "fun": lambda b: A @ b - 1.0, "jac": lambda b: A}],
        options={"ftol": tol, "maxiter": max_iter},
    )  # fmt: skip
    B = project_simplex_rows(res.x.reshape(K, J))  # nettoie les écarts numériques de 1e-10
    return B


def goodman(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Régression de Goodman contrainte, pondérée par `w` : matrice (K, J) de reports."""
    G, H = sufficient_stats(X, Y, w)
    return solve_constrained(G, H)


def bootstrap(X: np.ndarray, Y: np.ndarray, w: np.ndarray, n_boot: int,
              rng: np.random.Generator, start: np.ndarray | None = None) -> np.ndarray:
    """Tirages (n_boot, K, J) par rééchantillonnage des communes avec remise."""
    n = len(w)
    out = np.empty((n_boot, X.shape[1], Y.shape[1]))
    for b in range(n_boot):
        m = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        G, H = sufficient_stats(X, Y, w * m)
        out[b] = solve_constrained(G, H, start)
    return out


def split_none(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(part du premier finaliste parmi les exprimés de l'électorat, part ni l'un ni l'autre)
    par ligne ; colonnes de B dans l'ordre de `Groups.outcomes`. NaN si l'électorat ne vote
    pour aucun finaliste."""
    to_fin = B[..., 0] + B[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        split = np.where(to_fin > 0, B[..., 0] / to_fin, np.nan)
    return split, B[..., 2] + B[..., 3]


# --- strates


def weighted_quantile_bins(values: np.ndarray, weights: np.ndarray, q: int = 5) -> np.ndarray:
    """Classe 1..q de chaque valeur, bornes aux quantiles pondérés (chaque classe porte
    environ 1/q du poids total). NaN -> 0."""
    ok = ~np.isnan(values)
    order = np.argsort(values[ok])
    cw = np.cumsum(weights[ok][order])
    cw = cw / cw[-1]
    cuts = [values[ok][order][np.searchsorted(cw, k / q)] for k in range(1, q)]
    out = np.zeros(len(values), dtype=int)
    out[ok] = 1 + np.searchsorted(np.asarray(cuts), values[ok], side="left")
    return out


@dataclass
class StratifiedFit:
    """Matrices par strate, tailles des électorats par strate et agrégat national."""

    strata: list[str]
    B: dict[str, np.ndarray]  # strate -> (K, J)
    sizes: dict[str, np.ndarray]  # strate -> (K,) inscrits de chaque électorat dans la strate
    n_communes: dict[str, int]
    boot: dict[str, np.ndarray] = field(default_factory=dict)  # strate -> (n_boot, K, J)

    def national(self, sizes: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """Agrégat Σ_s N_sk B_sk / Σ_s N_sk ; `sizes` permet d'appliquer d'autres tailles
        d'électorats (composition d'une autre élection)."""
        sizes = self.sizes if sizes is None else sizes
        num = sum(sizes[s][:, None] * self.B[s] for s in self.strata if s in sizes)
        den = sum(sizes[s] for s in self.strata if s in sizes)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den[:, None]

    def national_boot(self, sizes: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        sizes = self.sizes if sizes is None else sizes
        num = sum(sizes[s][None, :, None] * self.boot[s] for s in self.strata if s in sizes)
        den = sum(sizes[s] for s in self.strata if s in sizes)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den[None, :, None]


def electorate_sizes(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (X * w[:, None]).sum(0)


def stratified(X: np.ndarray, Y: np.ndarray, w: np.ndarray, labels: Sequence[str],
               n_boot: int = 0, rng: np.random.Generator | None = None,
               min_communes: int = 50) -> StratifiedFit:
    """Goodman contraint strate par strate ; bootstrap des communes dans chaque strate.
    Une strate de moins de `min_communes` communes fait échouer (strates à revoir)."""
    labels = np.asarray(labels)
    strata = sorted(set(labels.tolist()))
    fit = StratifiedFit(strata, {}, {}, {})
    for s in strata:
        m = labels == s
        if m.sum() < min_communes:
            raise ValueError(f"strate {s} : {m.sum()} communes (< {min_communes})")
        fit.B[s] = goodman(X[m], Y[m], w[m])
        fit.sizes[s] = electorate_sizes(X[m], w[m])
        fit.n_communes[s] = int(m.sum())
        if n_boot:
            assert rng is not None
            fit.boot[s] = bootstrap(X[m], Y[m], w[m], n_boot, rng, start=fit.B[s])
    return fit


def predict_shares(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Parts prédites des issues du second tour (en inscrits) : X B."""
    return X @ B
