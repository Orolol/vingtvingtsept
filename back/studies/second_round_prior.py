"""Étude : prior des reports du second tour, propositions de l'inférence écologique.

    uv run python studies/second_round_prior.py eval --seed 0 [--out runs/<date>-second-round-prior]
    uv run python studies/second_round_prior.py simulate [--out ...]
    uv run python studies/second_round_prior.py report [--out ...]

Deux propositions de runs/2026-10-22-ecological (journal de backtest, entrée du 2026-10-22),
évaluées au protocole du second tour (vvs.eval.second_round.run_structural : leave-one-
election-out, duel réellement joué orienté sur le vainqueur, comparaison appariée, mêmes tirages
de premier tour et même graine par cas pour toutes les variantes, graines 0, 1, 2, 8 000 tirages
par cas) :

- `sigma_wide` : décalages communs élargis, σ_u 0,30 → 0,5 (analogue) et 0,6 → 0,8 (inédit),
  σ_v 0,25 → 0,30 et 0,40 → 0,45 ;
- `sigma_mid` : version intermédiaire, σ_u 0,4 et 0,7, σ_v comme `sigma_wide` ;
- `prox_fr` : ligne `far_right` de la matrice de proximité, centre 2,5 → 6,5, droite 6 → 7,
  part « ni l'un ni l'autre » de l'électorat d'extrême droite (0,10 ; 0,55) au lieu du défaut
  (0,05 ; 0,55) ;
- `both`, `both_mid` : `prox_fr` avec `sigma_wide` ou `sigma_mid`.

Référence : la configuration publiée (biais RN par horizon, réglages du 2026-10-22, écrits en
toutes lettres ici pour que l'étude reste valable si les défauts changent). Deux familles de
premier tour : `dirichlet` (comme runs/2026-10-20-second-round-v2) et `eta` (modèle d'erreur η
de la simulation, sans les scrutins de liste). `simulate` rejoue la simulation 2027 publiée
(ajustement runs/2026-10-22-fit-2027-subst, substitution, courbe de biais RN, η avec listes,
40 000 tirages, graine 0) avec chaque jeu de réglages ; `ref` doit être identique tirage à
tirage au run publié runs/2026-10-22-simulate-subst.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.eval import second_round as ev
from vvs.model import simulate as sim
from vvs.model.plots import display_name
from vvs.model.second_round import ABSTAIN, SecondRoundSettings, proximity_mean
from vvs.registry import default_registry

SEEDS = (0, 1, 2)
REFERENCE_ELECTIONS = (2012, 2017, 2022)
FIT_DIR = paths.RUNS / "2026-10-22-fit-2027-subst"
PUBLISHED_SIM = paths.RUNS / "2026-10-22-simulate-subst"
ASOF = dt.date(2026, 10, 22)
HORIZON_LABEL = {"r2_eve": "veille", "r1_30": "30 j", "r1_90": "90 j", "r1_180": "180 j"}
HORIZON_ORDER = ("r2_eve", "r1_30", "r1_90", "r1_180")

# réglages publiés le 2026-10-22 (constantes de second_round.py à cette date)
OLD_SIGMA_U = {"analog": 0.30, "inedit": 0.60}
OLD_SIGMA_V = {"analog": 0.25, "inedit": 0.40}
OLD_FAR_RIGHT = {"far_left": 2, "left": 1.5, "centre": 2.5, "right": 6, "far_right": 10, "other": 4}
OLD_NONE = {"other": (0.10, 0.55), ABSTAIN: (0.55, 0.80)}
REF = SecondRoundSettings().with_(
    sigma_u=OLD_SIGMA_U, sigma_v=OLD_SIGMA_V, proximity_rows={"far_right": OLD_FAR_RIGHT},
    proximity_none=OLD_NONE,
)
WIDE = {"sigma_u": {"analog": 0.5, "inedit": 0.8}, "sigma_v": {"analog": 0.30, "inedit": 0.45}}
MID = {"sigma_u": {"analog": 0.4, "inedit": 0.7}, "sigma_v": {"analog": 0.30, "inedit": 0.45}}
PROX = {
    "proximity_rows": {"far_right": {"centre": 6.5, "right": 7}},
    "proximity_none": {**OLD_NONE, "far_right": (0.10, 0.55)},
}
SETTINGS: dict[str, SecondRoundSettings] = {
    "ref": REF,
    "sigma_wide": REF.with_(**WIDE),
    "sigma_mid": REF.with_(**MID),
    "prox_fr": REF.with_(**PROX),
    "both": REF.with_(**WIDE, **PROX),
    "both_mid": REF.with_(**MID, **PROX),
}
LABEL = {
    "ref": "référence publiée",
    "sigma_wide": "(a) σ élargis",
    "sigma_mid": "(d) σ intermédiaires",
    "prox_fr": "(b) ligne far_right",
    "both": "(c) (a) + (b)",
    "both_mid": "(d) + (b)",
}
FAMILIES = {"dirichlet": "", "eta": "eta_"}
SPECS: dict[str, tuple[str, dict]] = {
    f"{prefix}{name}": (family, {"bias_by_horizon": True, "settings": s})
    for family, prefix in FAMILIES.items()
    for name, s in SETTINGS.items()
}
DUELS = (
    ("le_pen_marine", "philippe_edouard"),
    ("le_pen_marine", "retailleau_bruno"),
    ("le_pen_marine", "glucksmann_raphael"),
    ("bardella_jordan", "philippe_edouard"),
    ("le_pen_marine", "melenchon_jean_luc"),
    ("philippe_edouard", "glucksmann_raphael"),
)


def _f(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",")


def _s(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.{nd}f}".replace(".", ",")


# --- évaluation


def run_eval(out: Path, seed: int, n: int = 8000) -> None:
    r = ev.run_structural(n=n, seed=seed, specs=SPECS)
    out.mkdir(parents=True, exist_ok=True)
    r["metrics"].with_columns(pl.lit(seed).alias("seed")).write_parquet(out / f"metrics_{seed}.parquet")
    if seed == SEEDS[0]:
        r["folds"].write_parquet(out / "folds.parquet")


def load_metrics(out: Path) -> pl.DataFrame:
    m = pl.concat([pl.read_parquet(p) for p in sorted(out.glob("metrics_*.parquet"))])
    return m.with_columns(
        (pl.col("q95") - pl.col("q05")).alias("width90"),
        (pl.col("q75") - pl.col("q25")).alias("width50"),
    )


def paired(m: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    """Différences appariées a − b par élection, horizon et graine."""
    keys = ["election", "horizon", "seed"]
    cols = ["crps", "log_score", "abs_error", "cover50", "cover90", "width90"]
    x = m.filter(pl.col("method") == a).select(keys + cols)
    y = m.filter(pl.col("method") == b).select(keys + cols)
    j = x.join(y, on=keys, suffix="_ref")
    return j.select(
        keys + [(pl.col(c).cast(pl.Float64) - pl.col(f"{c}_ref").cast(pl.Float64)).alias(f"d_{c}")
                for c in cols]
    )


# --- matrice : ligne far_right contre les reports observés


def far_right_row_check() -> pl.DataFrame:
    """Électorat d'extrême droite non finaliste, chaque second tour : répartition et part « ni
    l'un ni l'autre » de la matrice (ancienne et proposée) contre les enquêtes."""
    reg = default_registry()
    rows = []
    for e in ev.ELECTIONS:
        el = reg.elections[e]
        pair = tuple(r.candidate_id for r in el.round_(2).results)
        blocs = tuple(reg.bloc_of(c, e) for c in pair)
        split, none = ev.observed_transfers(e, "bloc:far_right", pair, reg)
        if split is None:
            continue
        row = {"election": e, "pair": " – ".join(display_name(c) for c in pair),
               "blocs": f"{blocs[0]} – {blocs[1]}", "obs_split": split, "obs_none": none}
        for name in ("ref", "prox_fr"):
            mi, mj, mn = proximity_mean("far_right", *blocs, SETTINGS[name])
            row[f"{name}_split"] = mi / (mi + mj)
            row[f"{name}_none"] = mn
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


# --- simulation 2027


def run_simulations(out: Path, final_dir: Path | None = None, final: str | None = None) -> None:
    from vvs.model.error import ErrorModel

    reg = default_registry()
    em = ErrorModel.fit(elections=sim.ERROR_ELECTIONS, registry=reg, use_bias=False,
                        with_lists=sim.ERROR_WITH_LISTS)
    mix_rows, pair_rows, man = [], [], {}
    for name, s in SETTINGS.items():
        target = final_dir if name == final else None
        res = sim.simulate(FIT_DIR, ASOF, out=target, n_draws=40_000, seed=0, rn_bias_curve=True,
                           second_round_settings=s, error_model=em, registry=reg)
        mix = res.summary.filter(pl.col("condition_id") == "mix")
        mix_rows.append(mix.select("candidate_id", "p_qualify", "p_win").with_columns(
            pl.lit(name).alias("variant")))
        pair_rows.append(res.pairs.filter(pl.col("condition_id") == "mix").with_columns(
            pl.lit(name).alias("variant")))
        man[name] = {"bias_mode": res.manifest["second_round_model"]["bias_mode"]}
        if name == "ref":
            old = pl.read_parquet(PUBLISHED_SIM / "draws.parquet")
            man[name]["identical_to_published"] = all(
                old[c].equals(res.draws[c]) if old[c].dtype == pl.Utf8 else bool(np.allclose(
                    old[c].fill_null(-1).to_numpy().astype(float),
                    res.draws[c].fill_null(-1).to_numpy().astype(float), atol=1e-9))
                for c in old.columns
            )
        logging.info("simulation %s faite", name)
    pl.concat(mix_rows).write_parquet(out / "sim_mix.parquet")
    pl.concat(pair_rows, how="diagonal_relaxed").write_parquet(out / "sim_pairs.parquet")
    (out / "sim_manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))


# --- rapport


def report(out: Path) -> str:
    m = load_metrics(out)
    seeds = sorted(m["seed"].unique().to_list())
    L = [
        "# Second tour : décalages communs élargis et ligne far_right de la matrice",
        "",
        (f"Étude du {dt.date.today().isoformat()} (`back/studies/second_round_prior.py`). "
         "Propositions de runs/2026-10-22-ecological évaluées au protocole du second tour : "
         "leave-one-election-out 2002-2022, duel réellement joué orienté sur le vainqueur, "
         f"8 000 tirages par cas, graines {', '.join(map(str, seeds))}, comparaison appariée "
         "(mêmes tirages de premier tour et même graine de second tour par cas pour toutes les "
         "variantes). Référence : configuration publiée (biais RN par horizon, réglages du "
         "2026-10-22). Familles de premier tour : `dirichlet` (bruit de `noisy_first_round`) et "
         "`eta` (modèle d'erreur η, sans scrutins de liste) ; à la veille du second tour, premier "
         "tour officiel dans les deux."),
        "",
        "Réglages :",
        "",
    ]
    for name, s in SETTINGS.items():
        fr = s.proximity["far_right"]
        L.append(
            f"- `{name}` ({LABEL[name]}) : σ_u {_f(s.sigma_u['analog'])} / {_f(s.sigma_u['inedit'])}, "
            f"σ_v {_f(s.sigma_v['analog'])} / {_f(s.sigma_v['inedit'])} (analogue / inédit) ; "
            f"far_right → centre {_f(fr['centre'], 1)}, droite {_f(fr['right'], 1)}, "
            f"« ni » {s.proximity_none.get('far_right', (0.05, 0.55))}"
        )
    L += [
        "",
        ("Où agit la ligne far_right au backtest : seulement en 2002 (duel droite – extrême "
         "droite inédit, électorat Mégret, 2,3 % des exprimés). En 2017 et 2022 l'électorat "
         "d'extrême droite non finaliste (Dupont-Aignan, Zemmour) a un analogue (l'autre "
         "élection) et la matrice n'est pas utilisée ; en 2007 et 2012, la ligne far_right sert "
         "l'électorat FN dans un duel droite – gauche, que les analogues couvrent aussi."),
        "",
    ]
    for family, prefix in FAMILIES.items():
        L += family_section(m, family, prefix)
    L += matrix_section()
    sm_path = out / "sim_manifest.json"
    if sm_path.exists():
        L += simulation_section(out, json.loads(sm_path.read_text()))
    text = "\n".join(L) + "\n" + CONCLUSIONS
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def family_section(m: pl.DataFrame, family: str, prefix: str) -> list[str]:
    names = [f"{prefix}{n}" for n in SETTINGS]
    mf = m.filter(pl.col("method").is_in(names))
    if mf.is_empty():
        return []
    L = [
        f"## Famille `{family}`",
        "",
        "### Synthèse par horizon (moyennes sur cinq élections et les graines)",
        "",
        ("CRPS et erreur absolue en points de la part du vainqueur ; log-score : log P(vainqueur "
         "gagne) ; largeur : intervalle à 90 % (points) ; couverture : fraction des 15 cas "
         "(5 élections × 3 graines) où le résultat tombe dans l'intervalle à 50 / 90 %."),
        "",
        "| horizon | variante | erreur abs. | CRPS | log-score | couv. 50 | couv. 90 | largeur 90 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    agg = mf.group_by("horizon", "method").agg(
        pl.col("abs_error").mean(), pl.col("crps").mean(), pl.col("log_score").mean(),
        pl.col("cover50").cast(pl.Float64).mean(), pl.col("cover90").cast(pl.Float64).mean(),
        pl.col("width90").mean(),
    )
    for h in HORIZON_ORDER:
        for name in names:
            r = agg.filter((pl.col("horizon") == h) & (pl.col("method") == name))
            if r.is_empty():
                continue
            r = r.row(0, named=True)
            L.append(
                f"| {HORIZON_LABEL[h]} | {LABEL[name.removeprefix(prefix)]} | {_f(r['abs_error'])} | "
                f"{_f(r['crps'])} | {_s(r['log_score'], 4)} | {_f(r['cover50'])} | "
                f"{_f(r['cover90'])} | {_f(r['width90'], 1)} |"
            )
    L += [
        "",
        "### Comparaisons appariées (variante − référence)",
        "",
        ("CRPS : négatif = mieux ; log-score : positif = mieux. « Réf. + / − » : élections de "
         "référence (2012, 2017, 2022) où le CRPS moyen sur les graines s'améliore / se dégrade "
         "(seuil 0,005 point). L'écart entre graines donne l'ordre du bruit Monte-Carlo."),
        "",
        "| variante | horizon | Δ CRPS | Δ CRPS par graine | Δ log-score | Δ couv. 50 | Δ couv. 90 | Δ largeur 90 | réf. + / − |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names[1:]:
        d = paired(mf, name, names[0])
        for h in HORIZON_ORDER:
            x = d.filter(pl.col("horizon") == h)
            per_seed = x.group_by("seed").agg(pl.col("d_crps").mean()).sort("seed")["d_crps"].to_list()
            ref_e = x.filter(pl.col("election").is_in(list(REFERENCE_ELECTIONS))).group_by(
                "election").agg(pl.col("d_crps").mean())
            gains = int((ref_e["d_crps"] < -0.005).sum())
            losses = int((ref_e["d_crps"] > 0.005).sum())
            L.append(
                f"| {LABEL[name.removeprefix(prefix)]} | {HORIZON_LABEL[h]} | {_s(x['d_crps'].mean())} | "
                f"{' ; '.join(_s(v) for v in per_seed)} | {_s(x['d_log_score'].mean(), 4)} | "
                f"{_s(x['d_cover50'].mean(), 2)} | {_s(x['d_cover90'].mean(), 2)} | "
                f"{_s(x['d_width90'].mean(), 2)} | {gains} / {losses} |"
            )
    L += [
        "",
        "### Par élection et horizon (moyennes sur les graines)",
        "",
        ("Cellule : CRPS / log-score / couverture 50 et 90 % (fraction des graines) ; "
         "première colonne : résultat et médiane de la référence [intervalle à 90 %]."),
        "",
        "| élection | horizon | résultat, réf. | " + " | ".join(LABEL[n.removeprefix(prefix)] for n in names) + " |",
        "|---|---|---|" + "---|" * len(names),
    ]
    per = mf.group_by("election", "horizon", "method").agg(
        pl.col("result").first(), pl.col("median").mean(), pl.col("q05").mean(),
        pl.col("q95").mean(), pl.col("crps").mean(), pl.col("log_score").mean(),
        pl.col("cover50").cast(pl.Float64).mean(), pl.col("cover90").cast(pl.Float64).mean(),
    )
    for e in sorted(per["election"].unique().to_list()):
        for h in HORIZON_ORDER:
            x = per.filter((pl.col("election") == e) & (pl.col("horizon") == h))
            if x.is_empty():
                continue
            ref = x.filter(pl.col("method") == names[0]).row(0, named=True)
            cells = []
            for name in names:
                r = x.filter(pl.col("method") == name)
                if r.is_empty():
                    cells.append("–")
                    continue
                r = r.row(0, named=True)
                cells.append(f"{_f(r['crps'])} / {_s(r['log_score'], 3)} / {_f(r['cover50'], 1)}·{_f(r['cover90'], 1)}")
            L.append(
                f"| {e} | {HORIZON_LABEL[h]} | {_f(ref['result'], 1)} ; {_f(ref['median'], 1)} "
                f"[{_f(ref['q05'], 1)}–{_f(ref['q95'], 1)}] | " + " | ".join(cells) + " |"
            )
    L.append("")
    return L


def matrix_section() -> list[str]:
    chk = far_right_row_check()
    L = [
        "## Ligne far_right contre les reports observés (enquêtes)",
        "",
        ("Électorat d'extrême droite non finaliste : part du premier finaliste parmi ses "
         "exprimés et part « ni l'un ni l'autre », enquêtes post-électorales de l'élection, "
         "contre la matrice actuelle et la proposition (en %). Recoupement, pas un test : la "
         "proposition a été tirée de ces cellules."),
        "",
        "| élection | duel (blocs) | répartition observée | actuelle | proposée | « ni » observée | actuelle | proposée |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in chk.iter_rows(named=True):
        L.append(
            f"| {r['election']} | {r['pair']} ({r['blocs']}) | {_f(100 * r['obs_split'], 1)} | "
            f"{_f(100 * r['ref_split'], 1)} | {_f(100 * r['prox_fr_split'], 1)} | "
            f"{_f(None if r['obs_none'] is None else 100 * r['obs_none'], 1)} | "
            f"{_f(100 * r['ref_none'], 1)} | {_f(100 * r['prox_fr_none'], 1)} |"
        )
    for name in ("ref", "prox_fr"):
        e_split = (chk[f"{name}_split"] - chk["obs_split"]).abs().mean()
        nn = chk.filter(pl.col("obs_none").is_not_null())
        e_none = (nn[f"{name}_none"] - nn["obs_none"]).abs().mean()
        L.append("")
        L.append(f"Écart absolu moyen, `{name}` : répartition {_f(100 * e_split, 1)} points, "
                 f"« ni » {_f(100 * e_none, 1)} points.")
    L.append("")
    return L


def simulation_section(out: Path, sm: dict) -> list[str]:
    mix = pl.read_parquet(out / "sim_mix.parquet")
    pairs = pl.read_parquet(out / "sim_pairs.parquet")
    variants = list(SETTINGS)
    L = [
        "## Simulation 2027 (au 2026-10-22)",
        "",
        (f"Ajustement {FIT_DIR.name}, substitution, biais RN par horizon, η avec scrutins de "
         "liste, 40 000 tirages, graine 0 ; premier tour identique dans toutes les variantes. "
         f"`ref` identique tirage à tirage au run publié {PUBLISHED_SIM.name} : "
         f"{'oui' if sm['ref'].get('identical_to_published') else 'NON'}."),
        "",
        "### Mélange des scénarios : P(victoire), %",
        "",
        "| candidat | P(qualif.) | " + " | ".join(f"`{v}`" for v in variants) + " |",
        "|---|---|" + "---|" * len(variants),
    ]
    ref = mix.filter(pl.col("variant") == "ref").filter(pl.col("p_win") >= 0.004).sort(
        "p_win", descending=True)
    for r in ref.iter_rows(named=True):
        c = r["candidate_id"]
        cells = []
        for v in variants:
            x = mix.filter((pl.col("variant") == v) & (pl.col("candidate_id") == c))
            cells.append(_f(100 * x["p_win"][0], 1) if x.height else "–")
        L.append(f"| {display_name(c)} | {_f(100 * r['p_qualify'], 1)} | " + " | ".join(cells) + " |")
    L += [
        "",
        "### Duels (mélange)",
        "",
        ("P(a gagne | duel), % ; entre parenthèses, médiane de la part de a [intervalle à 90 %]. "
         "P(duel) identique dans toutes les variantes."),
        "",
        "| duel | P(duel) | inédit | " + " | ".join(f"`{v}`" for v in variants) + " |",
        "|---|---|---|" + "---|" * len(variants),
    ]
    for a, b in DUELS:
        o = pairs.filter((pl.col("variant") == "ref") & (pl.col("candidate_a") == a) & (pl.col("candidate_b") == b))
        if o.is_empty():
            continue
        cells = []
        for v in variants:
            x = pairs.filter((pl.col("variant") == v) & (pl.col("candidate_a") == a) & (pl.col("candidate_b") == b))
            cells.append(
                f"{_f(100 * x['p_a_wins'][0], 1)} ({_f(x['q50'][0], 1)} [{_f(x['q05'][0], 1)}–{_f(x['q95'][0], 1)}])"
                if x.height else "–")
        L.append(
            f"| {display_name(a)} – {display_name(b)} | {_f(100 * o['p_pair'][0], 1)} | "
            f"{'oui' if o['inedit'][0] else 'non'} | " + " | ".join(cells) + " |"
        )
    L.append("")
    return L


CONCLUSIONS = """
## Conclusions

1. **La couverture ne bouge pas.** Aucune variante ne change la couverture d'un seul cas :
   90 % à 0,80 à la veille et à 180 jours, 1,00 à 30 et 90 jours, 50 % inchangée (sauf un cas
   de 2017 à 90 jours en famille η avec σ élargis). Les deux cas hors de l'intervalle à 90 %
   sont 2012 (veille et 180 jours : sondages de duel trop favorables à Hollande) ; élargir les
   décalages communs du prior n'y change rien, parce que la loi a posteriori est tenue par les
   sondages de duel (V ≥ σ0²) : l'intervalle à 90 % ne s'élargit que de 0,1 point à la veille
   et de 0,6 à 0,7 point avant le premier tour avec (a). La lecture « correction de
   structure » (couverture plus proche de la nominale sans perte de log-score) ne s'applique
   donc à aucune variante.
2. **(a) σ élargis : non admise.** CRPS meilleur à tous les horizons (−0,10 à la veille,
   −0,11 / −0,04 / −0,07 à 30 / 90 / 180 jours en famille dirichlet ; −0,07 / −0,02 / −0,04 en
   famille η), stable sur les trois graines, sur 2017 et 2022 (prior moins pesant, la loi
   suit davantage les sondages) contre 2012 (2 élections de référence sur 3). Mais le
   log-score du vainqueur baisse avant le premier tour dans les deux familles (−0,001 à
   −0,005 ; famille η : négatif pour 8 des 9 couples graine × horizon), surtout par 2002 (plus
   de tirages où Le Pen gagne). Règle de docs/07 : log-score non amélioré, refus.
3. **(d) σ intermédiaires : non admise.** Mêmes gains de CRPS, réduits de moitié environ
   (−0,004 à −0,06) ; log-score nul au bruit près (+0,0004 à −0,0023, signe changeant d'une
   graine à l'autre). Pas de gain de log-score ni de couverture : refus.
4. **(b) ligne far_right : non admise, non évaluable au backtest.** Elle n'agit qu'en 2002
   (électorat Mégret, 2,3 % des exprimés) : gain mécanique de CRPS (−0,02 à −0,07) et de
   log-score (+0,001) parce que le modèle sous-estime Chirac de 10 points et que la ligne
   déplace un peu de voix vers lui. Effet nul sur les trois élections de référence (critère 2
   de la règle : 0 sur 3). Le recoupement avec les enquêtes est mitigé : meilleur en 2002,
   2017 et 2022 (dont la proposition est tirée), moins bon en 2007 et 2012 (électorat FN entre
   droite et gauche). Surtout, la ligne s'applique en 2027 à **tout l'électorat RN** dès que
   le RN n'est pas au second tour, ce que les cellules d'origine (électorats Dupont-Aignan et
   Zemmour face à un finaliste RN) ne renseignent pas : Philippe – Glucksmann passe de 48 à
   66 % pour Philippe (électorat RN : 62 → 92 % de Philippe parmi ses exprimés, « ni » 43 →
   26 %). C'est une extrapolation, pas une correction mesurée.
5. **(c) et (d) + (b) : non admises.** Les gains de CRPS s'additionnent, mais (d) + (b) ne
   passe la règle stricte qu'en famille dirichlet (log-score +0,0009 et +0,0004 à 180 et
   90 jours, gain apporté par 2002 seul) et la manque en famille η, celle de la simulation
   publiée (90 jours : −0,0014). Une combinaison choisie après lecture parmi cinq variantes,
   dont le log-score change de signe d'une graine à l'autre, n'est pas un gain établi.

Défauts inchangés (`SIGMA_U`, `SIGMA_V`, `PROXIMITY`, `PROXIMITY_NONE` de second_round.py).
Effet qu'aurait eu chaque variante sur 2027 : tableau ci-dessus ; le mélange bouge de moins
de 0,5 point (P(Le Pen) 70,0 → 70,4 % au plus).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["eval", "simulate", "report"])
    ap.add_argument("--out", default=str(paths.RUNS / f"{dt.date.today().isoformat()}-second-round-prior"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.out)
    if args.step == "eval":
        run_eval(out, args.seed, args.n)
        return
    if args.step == "simulate":
        run_simulations(out)
    report(out)
    print(out / "report.md")


if __name__ == "__main__":
    main()
