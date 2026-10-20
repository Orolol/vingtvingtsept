"""Extraction des quantités d'intérêt du postérieur : trajectoires, effets maison, estimation
au jour du vote.

Les parts dépendent de la configuration de candidats : softmax restreint, ou logit emboîté par
bloc si l'ajustement a la substitution (`Priors.substitution`, voir `first_round`). `shares`,
`log_shares` et `config_shares` les calculent pour une configuration arbitraire. Sauf mention
contraire, elles sont calculées sur la configuration de référence `data.reference`
(candidats officiels pour une élection passée). L'« estimation au jour du vote » est le
soutien latent prolongé jusqu'à la semaine du vote par la seule marche aléatoire : c'est
la projection naïve de la phase 2, que le modèle d'erreur de la phase 3 remplacera
(docs/04-modele.md, section 2).
"""

from __future__ import annotations

import jax
import numpy as np
import polars as pl
from numpyro.diagnostics import summary

from vvs.model.data import PollData
from vvs.model.first_round import FitResult, ModelInputs, Priors, config_log_shares, latent

QUANTILES = {"q05": 0.05, "q25": 0.25, "median": 0.5, "q75": 0.75, "q95": 0.95}


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def _quantiles(x: np.ndarray, axis: int = 0) -> dict[str, np.ndarray]:
    q = np.quantile(x, list(QUANTILES.values()), axis=axis)
    return dict(zip(QUANTILES, q))


class Posterior:
    def __init__(self, data: PollData, fit: FitResult):
        self.data = data
        self.fit = fit
        priors = Priors(**fit.settings.get("priors", {}))
        self.inp = ModelInputs.from_data(data, priors, fit.settings.get("centered", False))
        sites = {k: v for k, v in fit.samples.items()}
        lat = jax.jit(jax.vmap(lambda s: latent(s, self.inp)))(sites)
        self.theta = np.asarray(lat["theta"])  # (S, T, C)
        self.alpha = np.asarray(lat["alpha"])  # (S, P, C)
        self.n_draws = self.theta.shape[0]
        self.ref_idx = [data.candidates.index(c) for c in data.reference]
        self.substitution = priors.substitution
        if self.substitution:
            self.beta = np.asarray(lat["beta"])  # (S, T, B)
            self.delta = np.asarray(lat["delta"])  # (S, T, C)
            self.lam = np.asarray(lat["lam"])  # (S, B)
            self.house_bloc = np.asarray(lat["house_bloc"])  # (S, P, B)
            self.house_cand = np.asarray(lat["house_cand"])  # (S, P, C)

    # --- parts

    def log_shares(
        self, idx: list[int], week: int | slice = slice(None), pollster: int | None = None
    ) -> np.ndarray:
        """log-parts (S, [T,] K) sur la configuration d'index candidats `idx`.

        Sans substitution : softmax de θ restreint à `idx` ; avec : logit emboîté par bloc
        (`first_round.config_log_shares`). `pollster` : ajoute l'effet maison de cet institut."""
        if not self.substitution:
            th = self.theta[:, week][..., idx]
            if pollster is not None:
                a = self.alpha[:, pollster][:, idx]
                th = th + (a if th.ndim == 2 else a[:, None])
            m = th.max(-1, keepdims=True)
            return th - m - np.log(np.exp(th - m).sum(-1, keepdims=True))
        beta = self.beta[:, week]
        delta = self.delta[:, week][..., idx]
        if pollster is not None:
            hb, hc = self.house_bloc[:, pollster], self.house_cand[:, pollster][:, idx]
            beta = beta + (hb if beta.ndim == 2 else hb[:, None])
            delta = delta + (hc if delta.ndim == 2 else hc[:, None])
        return config_log_shares(beta, delta, self.data.cand_bloc[idx], self.lam)

    def shares(self, idx: list[int], week: int | slice = slice(None)) -> np.ndarray:
        """Parts (S, [T,] K) sur la configuration d'index candidats `idx`."""
        if not self.substitution:
            return _softmax(self.theta[:, week][..., idx])
        return np.exp(self.log_shares(idx, week))

    def config_shares(self, candidates: list[str], week: int | slice = slice(None)) -> np.ndarray:
        """Parts (S, [T,] K) d'une configuration donnée par ses ids de candidats (tous connus
        de l'ajustement), avec ou sans substitution selon l'ajustement."""
        missing = [c for c in candidates if c not in self.data.candidates]
        if missing:
            raise KeyError(f"candidats absents de l'ajustement : {missing}")
        return self.shares([self.data.candidates.index(c) for c in candidates], week)

    # --- sorties tabulaires

    def trajectories(self) -> pl.DataFrame:
        """Trajectoires hebdomadaires des parts (en %) : candidats et blocs de la référence,
        puis chaque autre candidat ajouté seul à la référence (`config` = `reference+<id>`)."""
        d = self.data
        dates = d.week_date(np.arange(d.n_weeks))
        wb = d.weeks_before(np.arange(d.n_weeks))
        frames = []

        def frame(level: str, config: str, ids: list[str], blocs: list[str], sh: np.ndarray):
            q = _quantiles(100 * sh)  # (T, K) chacun
            for k, cid in enumerate(ids):
                frames.append(
                    pl.DataFrame(
                        {
                            "level": level,
                            "config": config,
                            "id": cid,
                            "bloc": blocs[k],
                            "week": np.arange(d.n_weeks),
                            "steps_before": wb,
                            "date": dates,
                            "observed": np.arange(d.n_weeks) <= d.asof_week,
                            **{name: v[:, k] for name, v in q.items()},
                        }
                    )
                )

        ref_bloc = [d.blocs[d.cand_bloc[i]] for i in self.ref_idx]
        sh = self.shares(self.ref_idx)
        frame("candidate", "reference", d.reference, ref_bloc, sh)
        bloc_ids = sorted(set(ref_bloc), key=d.blocs.index)
        bsh = np.stack([sh[..., [b == x for x in ref_bloc]].sum(-1) for b in bloc_ids], axis=-1)
        frame("bloc", "reference", bloc_ids, bloc_ids, bsh)
        for i, cid in enumerate(d.candidates):
            if i in self.ref_idx:
                continue
            sh_x = self.shares(self.ref_idx + [i])[..., -1:]
            frame("candidate", f"reference+{cid}", [cid], [d.blocs[d.cand_bloc[i]]], sh_x)
        return pl.concat(frames)

    def house_effects(self) -> pl.DataFrame:
        """Effets maison : α en logit et décalage induit (points) sur la configuration de
        référence à la semaine `asof`, pour chaque institut × candidat de la référence."""
        d = self.data
        w = d.asof_week
        base = self.shares(self.ref_idx, w)  # (S, K)
        n_polls = {
            p: len({d.poll_id[i] for i in range(d.n_hyp) if d.pollster_idx[i] == j})
            for j, p in enumerate(d.pollsters)
        }
        rows = []
        for j, p in enumerate(d.pollsters):
            with_house = np.exp(self.log_shares(self.ref_idx, w, pollster=j))
            eff = 100 * (with_house - base)
            qe = _quantiles(eff)
            qa = _quantiles(self.alpha[:, j][:, self.ref_idx])
            for k, cid in enumerate(d.reference):
                rows.append(
                    {
                        "pollster": p,
                        "n_polls": n_polls[p],
                        "candidate_id": cid,
                        "bloc": d.blocs[d.cand_bloc[self.ref_idx[k]]],
                        "alpha_median": float(qa["median"][k]),
                        "alpha_q05": float(qa["q05"][k]),
                        "alpha_q95": float(qa["q95"][k]),
                        "effect_pts_median": float(qe["median"][k]),
                        "effect_pts_q05": float(qe["q05"][k]),
                        "effect_pts_q95": float(qe["q95"][k]),
                    }
                )
        return pl.DataFrame(rows)

    def final(self, official: dict[str, float] | None = None) -> pl.DataFrame:
        """Estimation par candidat de la référence, à la semaine `asof` et au jour du vote
        (en %), avec le résultat officiel en regard s'il est connu."""
        d = self.data
        at_asof = _quantiles(100 * self.shares(self.ref_idx, d.asof_week))
        at_vote = _quantiles(100 * self.shares(self.ref_idx, d.n_weeks - 1))
        rows = []
        for k, cid in enumerate(d.reference):
            row = {"candidate_id": cid, "bloc": d.blocs[d.cand_bloc[self.ref_idx[k]]]}
            row |= {f"asof_{q}": float(v[k]) for q, v in at_asof.items()}
            row |= {f"vote_{q}": float(v[k]) for q, v in at_vote.items()}
            off = None if official is None else official.get(cid)
            row["official"] = off
            if off is not None:
                row["error"] = row["vote_median"] - off
                row["in_50"] = row["vote_q25"] <= off <= row["vote_q75"]
                row["in_90"] = row["vote_q05"] <= off <= row["vote_q95"]
            rows.append(row)
        df = pl.DataFrame(rows, infer_schema_length=None)
        for col, dtype in (("error", pl.Float64), ("in_50", pl.Boolean), ("in_90", pl.Boolean)):
            if col not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
        return df.sort("vote_median", descending=True)

    def derived_diagnostics(self) -> dict:
        """r_hat et ESS des parts de référence à `asof`, au vote et sur toutes les semaines
        entre le premier sondage et `asof` (NUTS seulement).

        Les coordonnées brutes des marches sans données (un candidat avant son premier
        sondage ou après son dernier) n'ont que le prior et mélangent lentement : on juge la
        convergence sur ces parts plutôt que sur le maximum de r_hat des sites bruts."""
        if self.fit.method != "nuts":
            return {}
        d = self.data
        der = {
            "share_asof": self.fit.by_chain(self.shares(self.ref_idx, d.asof_week)),
            "share_vote": self.fit.by_chain(self.shares(self.ref_idx, d.n_weeks - 1)),
            "share_observed_weeks": self.fit.by_chain(
                self.shares(self.ref_idx, slice(int(d.week.min()), d.asof_week + 1))
            ),
        }
        out = {}
        for k, v in summary(der, group_by_chain=True).items():
            out[k] = {
                "r_hat_max": float(np.nanmax(v["r_hat"])),
                "ess_min": float(np.nanmin(v["n_eff"])),
            }
        return out
