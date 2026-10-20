"""Étude : substitution intra-bloc contre IIA sur les sondages à plusieurs hypothèses (phase 4).

    uv run python studies/substitution.py [--out runs/<date>-backtest-subst]

Un même sondage (mêmes répondants, même terrain, même institut) teste souvent plusieurs
hypothèses de candidats. Quand une hypothèse « réduite » est contenue dans une hypothèse plus
large du même sondage, on prédit les parts de la réduite à partir des parts observées de la
large, de deux façons :

- IIA (softmax restreint du modèle de la phase 3) : parts de la large renormalisées sur les
  candidats restants ;
- substitution (logit emboîté par bloc, `vvs.model.first_round`) : le poids d'un bloc de part
  P_b dans la large, dont il reste Q_b, devient P_b^{1−λ_b} · Q_b^{λ_b}, les blocs sont
  renormalisés entre eux, et chaque bloc se répartit au prorata de ses candidats restants.
  Un bloc entièrement retiré se reporte au prorata sur les autres blocs, comme à l'IIA.

λ_b est estimé par maximum de vraisemblance multinomiale des parts de la réduite (effectif
nominal, chaque sondage pesant 1 quelle que soit la quantité de paires qu'il fournit), en
laissant de côté l'élection évaluée (leave-one-election-out sur toutes les élections qui ont
des paires). λ_b n'est identifié que pour les blocs qui perdent un candidat sans disparaître ;
les autres restent à 1 (sans effet). Erreurs en points, absolues, moyennes par candidat (selon
son bloc) et par bloc (somme des parts du bloc dans la réduite).

Sortie : `<out>/substitution_validation.md` et `substitution_pairs.parquet`.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

from vvs import paths
from vvs.model.data import DEFAULT_SAMPLE_SIZE, load_polls
from vvs.registry import BLOCS, default_registry

ELECTIONS = (2002, 2007, 2012, 2017, 2022, 2027)
# Part plancher (fraction) d'un candidat noté 0 dans l'hypothèse large : la moitié du pas
# d'arrondi habituel des instituts (0,5 point). Sans elle, un bloc réduit à un candidat à 0
# a un poids nul.
MIN_SHARE = 0.0025
EVAL_ELECTIONS = (2017, 2022, 2027)


def nested_pairs(election: int, reg) -> list[dict]:
    """Paires (large, réduite) d'un même sondage, réduite strictement incluse dans la large."""
    df = load_polls(election)
    meta = {r["poll_id"]: r for r in df.select("poll_id", "sample_size").unique().iter_rows(named=True)}
    hyps: dict[str, dict[str, dict[str, float]]] = {}
    for pid, hid, cid, s in df.select("poll_id", "hypothesis_id", "candidate_id", "share").iter_rows():
        hyps.setdefault(pid, {}).setdefault(hid, {})[cid] = s
    bloc: dict[str, str | None] = {}
    for c in df["candidate_id"].unique().to_list():
        try:
            bloc[c] = reg.bloc_of(c, election)
        except KeyError:
            bloc[c] = None
    pairs = []
    for pid, hs in hyps.items():
        if len(hs) < 2:
            continue
        found = []
        for hf, full in hs.items():
            for hr, red in hs.items():
                if hr == hf or not set(red) < set(full):
                    continue
                cands = sorted(full)
                if any(bloc[c] is None for c in cands) or len(red) < 2:
                    continue
                pf = np.array([full[c] for c in cands], dtype=float)
                pf = np.maximum(pf / max(pf.sum(), 1e-12), MIN_SHARE)
                pr = np.array([red.get(c, 0.0) for c in cands], dtype=float)
                if pf.sum() <= 0 or pr.sum() <= 0:
                    continue
                found.append(
                    {
                        "election": election,
                        "poll_id": pid,
                        "full": hf,
                        "reduced": hr,
                        "candidates": cands,
                        "blocs": [bloc[c] for c in cands],
                        "present": np.array([c in red for c in cands]),
                        "p_full": pf / pf.sum(),
                        "y_red": pr / pr.sum(),
                        "n": float(meta[pid]["sample_size"] or DEFAULT_SAMPLE_SIZE),
                    }
                )
        for p in found:
            p["weight"] = 1.0 / len(found)
        pairs += found
    return pairs


def predict(pair: dict, lam: dict[str, float] | None) -> np.ndarray:
    """Parts prédites de la réduite (sur `candidates`, 0 pour les absents). `lam` None : IIA."""
    p, keep = pair["p_full"], pair["present"]
    out = np.where(keep, p, 0.0)
    if lam is None:
        return out / out.sum()
    blocs = np.array(pair["blocs"])
    weights = {}
    for b in set(blocs[keep]):
        m = blocs == b
        P, Q = p[m].sum(), p[m & keep].sum()
        weights[b] = P ** (1 - lam.get(b, 1.0)) * Q ** lam.get(b, 1.0)
    total = sum(weights.values())
    for b, w in weights.items():
        m = (blocs == b) & keep
        out[m] = p[m] / p[m].sum() * w / total
    return out


def identified_blocs(pairs: list[dict]) -> list[str]:
    """Blocs qui perdent un candidat sans disparaître dans au moins une paire."""
    out = set()
    for pr in pairs:
        blocs = np.array(pr["blocs"])
        for b in set(blocs):
            m = blocs == b
            if (m & ~pr["present"]).any() and (m & pr["present"]).any():
                out.add(b)
    return [b for b in BLOCS if b in out]


def fit_lambda(pairs: list[dict]) -> dict[str, float]:
    """λ_b par maximum de vraisemblance multinomiale (bornes [0, 1])."""
    blocs = identified_blocs(pairs)
    if not blocs:
        return {}

    def nll(x):
        lam = dict(zip(blocs, x))
        total = 0.0
        for pr in pairs:
            q = predict(pr, lam)
            k = pr["present"]
            total -= pr["weight"] * pr["n"] * (pr["y_red"][k] * np.log(np.maximum(q[k], 1e-12))).sum()
        return total

    res = minimize(nll, np.full(len(blocs), 0.5), method="L-BFGS-B", bounds=[(0.0, 1.0)] * len(blocs))
    return dict(zip(blocs, map(float, res.x)))


def error_rows(pairs: list[dict], lam: dict[str, float], fold: str) -> list[dict]:
    rows = []
    for pr in pairs:
        k = pr["present"]
        blocs = np.array(pr["blocs"])
        removed = {b for b in set(blocs[~k])}
        whole = {b for b in removed if not (k & (blocs == b)).any()}
        kind = "bloc_entier" if whole else "intra_bloc"
        preds = {"iia": predict(pr, None), "subst": predict(pr, lam)}
        for i in np.flatnonzero(k):
            rows.append(
                {
                    "election": pr["election"],
                    "fold": fold,
                    "poll_id": pr["poll_id"],
                    "pair": f"{pr['full']}>{pr['reduced']}",
                    "kind": kind,
                    "level": "candidate",
                    "unit": pr["candidates"][i],
                    "bloc": pr["blocs"][i],
                    "observed": 100 * pr["y_red"][i],
                    "full": 100 * pr["p_full"][i],
                    **{f"err_{m}": 100 * abs(q[i] - pr["y_red"][i]) for m, q in preds.items()},
                    "weight": pr["weight"],
                }
            )
        for b in sorted(set(blocs[k]), key=BLOCS.index):
            m = (blocs == b) & k
            rows.append(
                {
                    "election": pr["election"],
                    "fold": fold,
                    "poll_id": pr["poll_id"],
                    "pair": f"{pr['full']}>{pr['reduced']}",
                    "kind": kind,
                    "level": "bloc",
                    "unit": b,
                    "bloc": b,
                    "observed": 100 * pr["y_red"][m].sum(),
                    "full": 100 * pr["p_full"][blocs == b].sum(),
                    **{f"err_{m_}": 100 * abs(q[m].sum() - pr["y_red"][m].sum())
                       for m_, q in preds.items()},
                    "weight": pr["weight"],
                }
            )
    return rows


def _table(df: pl.DataFrame, by: list[str]) -> str:
    agg = (
        df.group_by(by)
        .agg(
            pl.len().alias("n"),
            pl.col("poll_id").n_unique().alias("sondages"),
            pl.col("err_iia").mean().alias("iia"),
            pl.col("err_subst").mean().alias("subst"),
        )
        .sort(by)
    )
    head = "| " + " | ".join(by) + " | n | sondages | EAM IIA | EAM substitution | gain |"
    lines = [head, "|" + "---|" * (len(by) + 5)]
    for r in agg.iter_rows(named=True):
        cells = [str(r[c]) for c in by]
        gain = r["iia"] - r["subst"]
        lines.append(
            "| " + " | ".join(cells) + f" | {r['n']} | {r['sondages']} | {r['iia']:.2f} | "
            f"{r['subst']:.2f} | {gain:+.2f} |"
        )
    return "\n".join(lines)


def main(out: Path) -> None:
    reg = default_registry()
    pairs = {e: nested_pairs(e, reg) for e in ELECTIONS}
    with_pairs = [e for e in ELECTIONS if pairs[e]]
    all_pairs = [p for e in with_pairs for p in pairs[e]]
    lam_all = fit_lambda(all_pairs)
    lam_by_e = {e: fit_lambda(pairs[e]) for e in with_pairs}
    rows, lam_loo = [], {}
    for e in EVAL_ELECTIONS:
        train = [p for f in with_pairs if f != e for p in pairs[f]]
        lam_loo[e] = fit_lambda(train)
        rows += error_rows(pairs[e], lam_loo[e], "loo")
    df = pl.DataFrame(rows)
    df.write_parquet(out / "substitution_pairs.parquet")

    md = [
        "# Substitution intra-bloc contre IIA : sondages à plusieurs hypothèses",
        "",
        (
            f"Généré le {dt.date.today().isoformat()} par `back/studies/substitution.py`. "
            "Erreur absolue moyenne (points) des parts prédites de l'hypothèse réduite à partir "
            "de l'hypothèse large du même sondage ; λ estimé sans l'élection évaluée."
        ),
        "",
        "## Paires disponibles",
        "",
        "| élection | sondages à plusieurs hypothèses avec paire emboîtée | paires |",
        "|---|---|---|",
    ]
    for e in ELECTIONS:
        md.append(f"| {e} | {len({p['poll_id'] for p in pairs[e]})} | {len(pairs[e])} |")
    md += ["", "## λ_b estimés (1 = IIA, 0 = le bloc garde toute sa part)", ""]
    cols = sorted({b for d in [lam_all, *lam_by_e.values(), *lam_loo.values()] for b in d},
                  key=BLOCS.index)
    md.append("| estimation | " + " | ".join(cols) + " |")
    md.append("|---|" + "---|" * len(cols))

    def line(name, d):
        return f"| {name} | " + " | ".join(f"{d[b]:.2f}" if b in d else "–" for b in cols) + " |"

    md.append(line("toutes élections", lam_all))
    for e in with_pairs:
        md.append(line(f"{e} seule", lam_by_e[e]))
    for e in EVAL_ELECTIONS:
        md.append(line(f"sans {e} (pour évaluer {e})", lam_loo[e]))
    cand = df.filter(pl.col("level") == "candidate")
    bloc = df.filter(pl.col("level") == "bloc")
    md += [
        "",
        "## Erreur par élection (candidats)",
        "",
        _table(cand, ["election"]),
        "",
        "## Erreur par élection et type de paire (candidats)",
        "",
        (
            "`intra_bloc` : chaque bloc touché garde au moins un candidat ; `bloc_entier` : un "
            "bloc disparaît (la substitution n'y change que la part des autres blocs touchés)."
        ),
        "",
        _table(cand, ["election", "kind"]),
        "",
        "## Erreur par bloc du candidat",
        "",
        _table(cand, ["election", "bloc"]),
        "",
        "## Erreur sur la part des blocs",
        "",
        _table(bloc, ["election", "bloc"]),
        "",
    ]
    (out / "substitution_validation.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.RUNS / f"{dt.date.today().isoformat()}-backtest-subst")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    main(a.out)
