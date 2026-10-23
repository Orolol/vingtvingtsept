"""Étude : options structurelles du second tour (décisions 3 du journal de backtest).

    uv run python studies/second_round_v2.py eval [--out runs/<date>-second-round-v2]
    uv run python studies/second_round_v2.py simulate [--out ...] [--sim-out runs/<date>-simulate-subst-v2]
    uv run python studies/second_round_v2.py report [--out ...]

Trois options du modèle de second tour, évaluées en leave-one-election-out et en comparaison
appariée (vvs.eval.second_round.run_structural, mêmes tirages de premier tour et même graine
par cas pour toutes les variantes, trois graines) :

1. `bias_by_horizon` : biais RN des sondages de duel estimé par âge des sondages (courbe
   rétrécie et monotone) au lieu d'un biais constant, ou de son atténuation linéaire dans la
   simulation ;
2. `couple_far_right` : erreur de duel corrélée à la composante d'extrême droite de η ;
3. `second_round_turnout` : participation du second tour tirée d'une loi ajustée sur les
   seconds tours, au lieu de la participation implicite des reports.

`simulate` relance la simulation 2027 (ajustement runs/2026-10-20-fit-2027-subst, 40 000
tirages, graine 0) sans option (contrôle : identique à runs/2026-10-20-simulate-subst), avec
chaque option seule et avec les options retenues ; la dernière est écrite en entier dans
`--sim-out`. `report` écrit `report.md` à partir des sorties des deux étapes.
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
from vvs.model.run import _git_commit

SEEDS = (0, 1, 2)
REFERENCE_ELECTIONS = (2012, 2017, 2022)
FIT_DIR = paths.RUNS / "2026-10-20-fit-2027-subst"
OLD_SIM = paths.RUNS / "2026-10-20-simulate-subst"
OLD_EVAL = paths.RUNS / "2026-10-20-second-round"
ASOF = dt.date(2026, 10, 20)
HORIZON_LABEL = {
    "r2_eve": "veille du 2d tour",
    "r1_30": "30 j",
    "r1_90": "90 j",
    "r1_180": "180 j",
}
HORIZON_ORDER = ("r1_180", "r1_90", "r1_30", "r2_eve")
METHOD_LABEL = {
    "base": "biais constant (référence `model`)",
    "bias_linear": "biais constant, atténuation linéaire (simulation actuelle)",
    "bias_h": "biais par horizon",
    "turnout2": "participation du 2d tour tirée",
    "bias_h_turnout2": "biais par horizon + participation",
    "eta": "premier tour η (référence du couplage)",
    "eta_coupled": "η + couplage",
    "eta_all": "η + les trois options",
}
# (variante, référence appariée, option)
COMPARISONS = (
    ("bias_h", "base", "biais par horizon contre biais constant"),
    ("bias_h", "bias_linear", "biais par horizon contre atténuation linéaire"),
    ("eta_coupled", "eta", "couplage"),
    ("turnout2", "base", "participation du 2d tour"),
)
# simulations : nom -> options de `simulate`
SIM_VARIANTS = {
    "off": {"rn_bias_curve": False, "couple_far_right": False, "second_round_turnout": False},
    "curve": {"rn_bias_curve": True, "couple_far_right": False, "second_round_turnout": False},
    "coupled": {"rn_bias_curve": False, "couple_far_right": True, "second_round_turnout": False},
    "turnout2": {"rn_bias_curve": False, "couple_far_right": False, "second_round_turnout": True},
    "all": {"rn_bias_curve": True, "couple_far_right": True, "second_round_turnout": True},
}
DUELS = (
    ("le_pen_marine", "philippe_edouard"),
    ("le_pen_marine", "melenchon_jean_luc"),
    ("le_pen_marine", "glucksmann_raphael"),
    ("bardella_jordan", "philippe_edouard"),
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


def run_eval(out: Path, n: int = 8000) -> None:
    frames, folds = [], None
    for seed in SEEDS:
        r = ev.run_structural(n=n, seed=seed)
        frames.append(r["metrics"].with_columns(pl.lit(seed).alias("seed")))
        if folds is None:
            folds = r["folds"]
        logging.getLogger(__name__).info("graine %d faite", seed)
    out.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(out / "metrics.parquet")
    folds.write_parquet(out / "folds.parquet")


def aggregate(m: pl.DataFrame) -> pl.DataFrame:
    return (
        m.group_by("horizon", "method")
        .agg(
            pl.len().alias("n"),
            pl.col("abs_error").mean(),
            pl.col("crps").mean(),
            pl.col("cover50").cast(pl.Float64).mean(),
            pl.col("cover90").cast(pl.Float64).mean(),
            pl.col("log_score").mean(),
            (pl.col("turnout2_median") - pl.col("turnout2_obs")).abs().mean().alias("turnout_err"),
        )
        .sort("horizon", "method")
    )


def paired(m: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    """Différences appariées a − b par élection, horizon et graine."""
    keys = ["election", "horizon", "seed"]
    cols = ["crps", "log_score", "abs_error", "cover50", "cover90"]
    x = m.filter(pl.col("method") == a).select(keys + cols)
    y = m.filter(pl.col("method") == b).select(keys + cols)
    j = x.join(y, on=keys, suffix="_ref")
    return j.select(
        keys
        + [(pl.col(c).cast(pl.Float64) - pl.col(f"{c}_ref").cast(pl.Float64)).alias(f"d_{c}") for c in cols]
    )


def admission(m: pl.DataFrame, a: str, b: str) -> dict:
    """Règle d'admission (docs/07-evaluation.md) sur les horizons 180 et 90 jours, moyennes sur
    les graines : log-score et CRPS moyens améliorés, gain de même signe sur au moins deux des
    trois élections de référence, couverture non dégradée."""
    d = paired(m, a, b)
    out = {}
    for h in ("r1_180", "r1_90"):
        x = d.filter(pl.col("horizon") == h)
        by_e = x.group_by("election").agg(pl.col("d_crps").mean(), pl.col("d_log_score").mean())
        ref = by_e.filter(pl.col("election").is_in(list(REFERENCE_ELECTIONS)))
        out[h] = {
            "d_crps": float(x["d_crps"].mean()),
            "d_log_score": float(x["d_log_score"].mean()),
            "d_cover50": float(x["d_cover50"].mean()),
            "d_cover90": float(x["d_cover90"].mean()),
            "ref_crps_gains": int((ref["d_crps"] < -1e-9).sum()),
            "ref_crps_losses": int((ref["d_crps"] > 1e-9).sum()),
            "ref_log_gains": int((ref["d_log_score"] > 1e-9).sum()),
            "ref_log_losses": int((ref["d_log_score"] < -1e-9).sum()),
        }
    ok = all(
        o["d_crps"] < 0 and o["d_log_score"] > 0 and o["ref_crps_gains"] >= 2
        and o["d_cover50"] >= 0 and o["d_cover90"] >= 0
        for o in out.values()
    )
    out["admitted"] = ok
    return out


# --- simulation


def run_simulations(out: Path, sim_out: Path, final: str, n_draws: int = 40_000) -> None:
    rows, duel_rows, man = [], [], {}
    for name, opts in SIM_VARIANTS.items():
        target = sim_out if name == final else None
        res = sim.simulate(FIT_DIR, ASOF, out=target, n_draws=n_draws, seed=0, **opts)
        mix = res.summary.filter(pl.col("condition_id") == "mix")
        rows.append(mix.select("candidate_id", "p_qualify", "p_win").with_columns(pl.lit(name).alias("variant")))
        pm = res.pairs.filter(pl.col("condition_id") == "mix")
        duel_rows.append(pm.with_columns(pl.lit(name).alias("variant")))
        srm = res.manifest["second_round_model"]
        man[name] = {k: srm.get(k) for k in (
            "bias_mode", "bias_curve_applied", "rn_bias_scale", "bias_far_right_applied", "coupling",
            "turnout2", "turnout2_draws", "sigma0", "sigma1")}
        if name == "off":
            old = pl.read_parquet(OLD_SIM / "draws.parquet")
            same = all(
                old[c].equals(res.draws[c]) if old[c].dtype == pl.Utf8 else bool(np.allclose(
                    old[c].fill_null(-1).to_numpy().astype(float),
                    res.draws[c].fill_null(-1).to_numpy().astype(float), atol=1e-9))
                for c in old.columns
            )
            man[name]["identical_to_previous_run"] = same
            man[name]["turnout2_draws"] = sim.quantile_dict(res.draws["turnout_second_round"].to_numpy())
        logging.getLogger(__name__).info("simulation %s faite", name)
    pl.concat(rows).write_parquet(out / "sim_mix.parquet")
    pl.concat(duel_rows, how="diagonal_relaxed").write_parquet(out / "sim_pairs.parquet")
    (out / "sim_manifest.json").write_text(json.dumps({"final": final, "variants": man}, indent=1,
                                                      ensure_ascii=False, default=str))


# --- rapport


def report(out: Path) -> str:
    m = pl.read_parquet(out / "metrics.parquet")
    folds = pl.read_parquet(out / "folds.parquet")
    m0 = m.filter(pl.col("seed") == 0)
    L = [
        "# Second tour : biais RN par horizon, couplage avec le premier tour, participation",
        "",
        (f"Étude du {dt.date.today().isoformat()} (`back/studies/second_round_v2.py`, commit "
         f"{_git_commit()} + modifications non commitées). Évaluation leave-one-election-out "
         "2002-2022, duel réellement joué orienté sur le vainqueur, 8000 tirages par cas, "
         f"graines {', '.join(map(str, SEEDS))} (tableaux par élection : graine 0). Comparaison "
         "appariée : pour un cas, toutes les variantes d'une même famille partagent les mêmes "
         "tirages de premier tour et la même graine d'échantillonnage du second tour."),
        "",
        ("Familles de premier tour : `dirichlet` (bruit de `noisy_first_round`, celui de "
         "runs/2026-10-20-second-round) ; `eta` (modèle d'erreur η de la simulation, calibré "
         "hors élection, nécessaire au couplage qui a besoin de la composante d'extrême droite "
         "de chaque tirage). À la veille du second tour, premier tour officiel dans les deux "
         "familles ; le couplage y utilise l'erreur d'extrême droite du premier tour observée."),
        "",
    ]
    # paramètres par pli
    L += [
        "## Paramètres par pli",
        "",
        ("Biais RN par âge des sondages (points, finaliste RN surestimé si positif) : nœuds à "
         "l'âge moyen des fenêtres de poll_errors, 7 j (14 derniers jours, veille), 22 j "
         "(horizon 7), 45 (30), 105 (90), 195 (180). Moyenne pondérée par la précision, rétrécie "
         "vers 0 par un prior qui vaut une élection, puis isotone (|biais| non croissant avec "
         "l'âge). ρ : corrélation non centrée entre l'erreur du bloc RN au premier tour "
         "(log sondages / résultat) et le résidu de l'erreur RN des sondages de duel (biais par "
         "horizon estimé sans l'élection retiré), même nœud ; brute puis × n / (n + 2)."),
        "",
        "| pli exclu | σ0 / σ1 constant | biais constant | σ0 / σ1 courbe | biais 7 / 22 / 45 / 105 / 195 j | ρ brut | ρ retenu | élections du ρ | participation 2d tour (moyenne, σ logit) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in folds.sort("election").iter_rows(named=True):
        curve = " / ".join(_f(r[f"bias_age_{a}"]) for a in (7, 22, 45, 105, 195))
        L.append(
            f"| {r['election']} | {_f(r['sigma0'])} / {_f(r['sigma1'])} | {_f(r['bias_far_right'])} | "
            f"{_f(r['sigma0_h'])} / {_f(r['sigma1_h'])} | {curve} | {_f(r['rho_raw'])} | "
            f"{_f(r['rho'])} | {r['rho_elections']} | {_f(100 * r['turnout2_mean'], 1)} %, "
            f"{_f(r['turnout2_logit_sd'])} |"
        )
    sm_path = out / "sim_manifest.json"
    if sm_path.exists():
        sm = json.loads(sm_path.read_text())
        c = sm["variants"].get("curve", {}).get("bias_curve_applied")
        cp = sm["variants"].get("coupled", {}).get("coupling") or {}
        t2 = sm["variants"].get("turnout2", {}).get("turnout2") or {}
        if c:
            L += ["", ("Calibrage 2027 (toutes élections) : biais " + " / ".join(
                f"{_f(b)} à {int(a)} j" for a, b in c) + f" ; ρ = {_f(cp.get('rho'))} (brut "
                f"{_f(cp.get('rho_raw'))}, {cp.get('n_elections')} élections) ; participation du "
                f"second tour logit-normale de moyenne {_f(t2.get('logit_mean'))}, σ "
                f"{_f(t2.get('logit_sd'))} (seconds tours {', '.join(f'{k} : {_f(100 * v, 1)} %' for k, v in (t2.get('elections') or {}).items())}).")]
    # synthèse
    ref = pl.read_parquet(OLD_EVAL / "metrics.parquet").filter(pl.col("method") == "model")
    ref = ref.with_columns(pl.lit("2026-10-20 `model`").alias("method"), pl.lit(None, pl.Float64).alias("turnout2_median"),
                           pl.lit(None, pl.Float64).alias("turnout2_obs"))
    agg_all = aggregate(m)
    agg_ref = aggregate(ref.select(m0.select("horizon", "method", "abs_error", "crps", "cover50",
                                             "cover90", "log_score", "turnout2_median",
                                             "turnout2_obs").columns))
    L += [
        "",
        "## Synthèse par horizon (moyennes sur les cinq élections et les trois graines)",
        "",
        ("Erreur abs. et CRPS en points de la part du vainqueur ; log-score : log P(vainqueur "
         "gagne) ; « participation » : écart absolu moyen entre la participation du second tour "
         "(exprimés / inscrits) médiane simulée et l'observée. Première ligne de chaque "
         "horizon : le run de référence runs/2026-10-20-second-round (méthode `model`, une "
         "graine, non apparié)."),
        "",
        "| horizon | variante | erreur abs. | CRPS | couv. 50 % | couv. 90 % | log-score | participation (pts) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = ["base", "bias_linear", "bias_h", "turnout2", "bias_h_turnout2", "eta", "eta_coupled", "eta_all"]
    for h in HORIZON_ORDER:
        rr = agg_ref.filter(pl.col("horizon") == h)
        for r in rr.iter_rows(named=True):
            L.append(f"| {HORIZON_LABEL[h]} | {r['method']} | {_f(r['abs_error'])} | {_f(r['crps'])} | "
                     f"{_f(r['cover50'])} | {_f(r['cover90'])} | {_f(r['log_score'], 3)} | – |")
        for meth in order:
            r = agg_all.filter((pl.col("horizon") == h) & (pl.col("method") == meth))
            if r.is_empty():
                continue
            r = r.row(0, named=True)
            L.append(f"| {HORIZON_LABEL[h]} | {METHOD_LABEL[meth]} | {_f(r['abs_error'])} | {_f(r['crps'])} | "
                     f"{_f(r['cover50'])} | {_f(r['cover90'])} | {_f(r['log_score'], 3)} | "
                     f"{_f(100 * r['turnout_err'], 1)} |")
    # comparaisons appariées
    L += [
        "",
        "## Comparaisons appariées (variante − référence ; CRPS : négatif = mieux ; log-score : positif = mieux)",
        "",
        ("Moyennes sur les élections et les trois graines. « Réf. + / − » : nombre d'élections de "
         "référence (2012, 2017, 2022) où le CRPS moyen s'améliore / se dégrade. L'écart entre "
         "graines donne l'ordre du bruit Monte-Carlo."),
        "",
        "| option | horizon | Δ CRPS | Δ CRPS par graine | Δ log-score | Δ couv. 50 | Δ couv. 90 | réf. + / − |",
        "|---|---|---|---|---|---|---|---|",
    ]
    verdicts = {}
    for a, b, label in COMPARISONS:
        d = paired(m, a, b)
        adm = admission(m, a, b)
        verdicts[label] = adm
        for h in HORIZON_ORDER:
            x = d.filter(pl.col("horizon") == h)
            per_seed = x.group_by("seed").agg(pl.col("d_crps").mean()).sort("seed")["d_crps"].to_list()
            ref_e = x.filter(pl.col("election").is_in(list(REFERENCE_ELECTIONS))).group_by(
                "election").agg(pl.col("d_crps").mean())
            gains = int((ref_e["d_crps"] < -1e-9).sum())
            losses = int((ref_e["d_crps"] > 1e-9).sum())
            L.append(
                f"| {label} | {HORIZON_LABEL[h]} | {_s(x['d_crps'].mean())} | "
                f"{' ; '.join(_s(v) for v in per_seed)} | {_s(x['d_log_score'].mean(), 4)} | "
                f"{_s(x['d_cover50'].mean(), 2)} | {_s(x['d_cover90'].mean(), 2)} | {gains} / {losses} |"
            )
    L += ["", "### Règle d'admission (180 et 90 jours)", "",
          "| option | 180 j : Δ CRPS, Δ log, réf. + | 90 j : Δ CRPS, Δ log, réf. + | couverture | verdict |",
          "|---|---|---|---|---|"]
    for label, adm in verdicts.items():
        cells = []
        for h in ("r1_180", "r1_90"):
            o = adm[h]
            cells.append(f"{_s(o['d_crps'])}, {_s(o['d_log_score'], 4)}, {o['ref_crps_gains']}/3")
        cov = "non dégradée" if all(adm[h]["d_cover50"] >= 0 and adm[h]["d_cover90"] >= 0 for h in ("r1_180", "r1_90")) else "dégradée"
        L.append(f"| {label} | {cells[0]} | {cells[1]} | {cov} | {'admise' if adm['admitted'] else 'non admise'} |")
    # par élection
    L += [
        "",
        "## Par élection et horizon (graine 0)",
        "",
        "Médiane de la part du vainqueur (CRPS).",
        "",
        "| élection | horizon | résultat | " + " | ".join(METHOD_LABEL[x].split(" (")[0] for x in order) + " |",
        "|---|---|---|" + "---|" * len(order),
    ]
    for e in sorted(m0["election"].unique().to_list()):
        for h in HORIZON_ORDER:
            x = m0.filter((pl.col("election") == e) & (pl.col("horizon") == h))
            if x.is_empty():
                continue
            cells = []
            for meth in order:
                r = x.filter(pl.col("method") == meth)
                cells.append("–" if r.is_empty() else f"{_f(r['median'][0], 1)} ({_f(r['crps'][0])})")
            L.append(f"| {e} | {HORIZON_LABEL[h]} | {_f(x['result'][0], 1)} | " + " | ".join(cells) + " |")
    # participation
    L += [
        "",
        "## Participation du second tour (exprimés / inscrits, %)",
        "",
        ("Sans option, elle est implicite : Σ w (t_i + t_j) / Σ w, où w est le poids de "
         "premier tour des électorats (abstentionnistes, blancs et nuls compris, rapportés aux "
         "exprimés du premier tour) et t leurs reports. Avec l'option, elle est tirée d'une "
         "logit-normale ajustée sur les seconds tours hors élection évaluée et imposée par un "
         "décalage commun des parts « ni l'un ni l'autre »."),
        "",
        "| élection | horizon | observée | implicite (médiane [5–95 %]) | tirée (médiane [5–95 %]) |",
        "|---|---|---|---|---|",
    ]
    for e in sorted(m0["election"].unique().to_list()):
        for h in ("r1_90", "r2_eve"):
            b = m0.filter((pl.col("election") == e) & (pl.col("horizon") == h) & (pl.col("method") == "base"))
            t = m0.filter((pl.col("election") == e) & (pl.col("horizon") == h) & (pl.col("method") == "turnout2"))
            if b.is_empty() or t.is_empty():
                continue
            L.append(
                f"| {e} | {HORIZON_LABEL[h]} | {_f(100 * b['turnout2_obs'][0], 1)} | "
                f"{_f(100 * b['turnout2_median'][0], 1)} [{_f(100 * b['turnout2_q05'][0], 1)}–{_f(100 * b['turnout2_q95'][0], 1)}] | "
                f"{_f(100 * t['turnout2_median'][0], 1)} [{_f(100 * t['turnout2_q05'][0], 1)}–{_f(100 * t['turnout2_q95'][0], 1)}] |"
            )
    # simulation
    if sm_path.exists():
        L += simulation_section(out, json.loads(sm_path.read_text()))
    text = "\n".join(L) + "\n" + CONCLUSIONS
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def simulation_section(out: Path, sm: dict) -> list[str]:
    mix = pl.read_parquet(out / "sim_mix.parquet")
    pairs = pl.read_parquet(out / "sim_pairs.parquet")
    old = pl.read_parquet(OLD_SIM / "summary.parquet").filter(pl.col("condition_id") == "mix")
    old_pairs = pl.read_parquet(OLD_SIM / "pairs.parquet").filter(pl.col("condition_id") == "mix")
    variants = list(SIM_VARIANTS)
    L = [
        "",
        "## Simulation 2027 (au 2026-10-20, 180 jours du premier tour)",
        "",
        (f"Ajustement {FIT_DIR.name}, 40 000 tirages, graine 0. `off` : aucune option, "
         f"identique tirage à tirage à {OLD_SIM.name} : "
         f"{'oui' if sm['variants']['off'].get('identical_to_previous_run') else 'NON'}. "
         f"Run retenu (`{sm['final']}`) écrit en entier dans runs/2026-10-20-simulate-subst-v2. "
         "Le premier tour est identique dans toutes les variantes (mêmes tirages) : seules les "
         "victoires changent."),
        "",
        "Biais RN appliqué : " + "; ".join(
            f"`{v}` : {sm['variants'][v]['bias_mode']}"
            + (f" (courbe {', '.join(_f(b) for _, b in sm['variants'][v]['bias_curve_applied'])})"
               if sm['variants'][v]['bias_curve_applied'] else
               f" (×{_f(sm['variants'][v]['rn_bias_scale'])}, {_f(sm['variants'][v]['bias_far_right_applied'])} pt)")
            for v in variants) + ".",
        "",
        "Participation du second tour simulée (médiane [5–95 %]) : " + "; ".join(
            f"`{v}` : {_f(100 * sm['variants'][v]['turnout2_draws']['q50'], 1)} % "
            f"[{_f(100 * sm['variants'][v]['turnout2_draws']['q05'], 1)}–{_f(100 * sm['variants'][v]['turnout2_draws']['q95'], 1)}]"
            for v in variants if sm["variants"][v].get("turnout2_draws")) + ".",
        "",
        "### Mélange des scénarios : P(victoire), %",
        "",
        "| candidat | P(qualif.) | 2026-10-20 | " + " | ".join(f"`{v}`" for v in variants) + " |",
        "|---|---|---|" + "---|" * len(variants),
    ]
    top = old.filter(pl.col("p_win") >= 0.005).sort("p_win", descending=True)
    for r in top.iter_rows(named=True):
        c = r["candidate_id"]
        cells = []
        for v in variants:
            x = mix.filter((pl.col("variant") == v) & (pl.col("candidate_id") == c))
            cells.append(_f(100 * x["p_win"][0], 1) if x.height else "–")
        L.append(f"| {display_name(c)} | {_f(100 * r['p_qualify'], 1)} | {_f(100 * r['p_win'], 1)} | " + " | ".join(cells) + " |")
    L += [
        "",
        "### Duels principaux (mélange)",
        "",
        "P(a gagne | duel), % ; entre parenthèses, médiane de la part de a. P(duel) identique dans toutes les variantes.",
        "",
        "| duel | P(duel) | 2026-10-20 | " + " | ".join(f"`{v}`" for v in variants) + " |",
        "|---|---|---|" + "---|" * len(variants),
    ]
    for a, b in DUELS:
        o = old_pairs.filter((pl.col("candidate_a") == a) & (pl.col("candidate_b") == b))
        if o.is_empty():
            continue
        cells = []
        for v in variants:
            x = pairs.filter((pl.col("variant") == v) & (pl.col("candidate_a") == a) & (pl.col("candidate_b") == b))
            cells.append(f"{_f(100 * x['p_a_wins'][0], 1)} ({_f(x['q50'][0], 1)})" if x.height else "–")
        L.append(
            f"| {display_name(a)} – {display_name(b)} | {_f(100 * o['p_pair'][0], 1)} | "
            f"{_f(100 * o['p_a_wins'][0], 1)} ({_f(o['q50'][0], 1)}) | " + " | ".join(cells) + " |"
        )
    return L


CONCLUSIONS = """
## Ce que faisait le code pour la participation

- La part du second tour parmi les exprimés est calculée directement, part_i = Σ w t_i /
  Σ w (t_i + t_j) : aucune conversion n'utilise la participation du premier tour. Celle-ci ne
  sert qu'au poids des abstentionnistes, blancs et nuls du premier tour (w = (inscrits −
  exprimés) / exprimés), ce qui est correct.
- La participation du second tour n'était tirée nulle part : elle est implicite, Σ w (t_i + t_j)
  / Σ w, et dépend des parts « ni l'un ni l'autre » des reports. Elle est mal calibrée : 78,6 %
  à la veille du second tour 2017 contre 66,0 % observés (tableau ci-dessus). **L'écart « 78 %
  contre 66 % » cité dans docs/09-roadmap.md et docs/backtest-log.md est donc bien la
  participation implicite du second tour, pas celle du premier tour simulé** (premier tour
  2017 : 75,8 %) ; la précision du commit f0804be est à corriger.
- Seul usage d'une participation du second tour pour convertir : les plafonds du potentiel
  électoral (option `use_potential`, désactivée), avec celle du second tour précédent ; avec
  l'option `second_round_turnout`, ils utilisent la participation tirée de chaque tirage.
- La colonne `turnout_second_round` (implicite ou tirée) est ajoutée aux tirages de la
  simulation ; les sorties de premier tour sont inchangées (vérifié tirage à tirage).

## Conclusions

1. **Biais RN par horizon** : courbe estimée 3,0 points à la veille, 2,9 à 22 jours, 1,6 à
   45, 1,4 à 105, 1,3 au-delà de 195 jours du second tour (calibrage 2027). Elle bat
   l'atténuation linéaire actuelle à 180 jours (Δ CRPS −0,23, trois graines, deux élections de
   référence sur trois) et fait jeu égal à 90 jours (−0,01) ; contre le biais constant, −0,12 et
   −0,03. Mais elle perd à 30 jours (+0,11) et à la veille (+0,035), surtout en 2017 (sans
   2017, un seul duel informe les nœuds au-delà de 45 jours), et le log-score du vainqueur
   baisse légèrement à tous les horizons (−0,001 à −0,004, porté par 2007 et 2012 via σ0 et
   σ1 réestimés sur les résidus à la courbe). **Non admise** par la règle stricte (log-score).
   Couverture : non dégradée, améliorée à 90 jours (90 % : +0,13). Recommandation : c'est une
   correction de structure qui remplace un jugement jamais évalué (l'atténuation linéaire), et
   elle est meilleure que lui à l'horizon de la prévision actuelle ; l'activer pour la
   prévision d'octobre 2026 (`rn_bias_curve=True`) est défendable, à condition de la
   réévaluer avant 30 jours du premier tour. L'option reste désactivée par défaut en attendant
   la décision.
2. **Couplage premier tour / duel** : ρ n'est pas identifiable avec trois élections. Brut, il
   change de signe d'un pli à l'autre (−0,36 à +0,70 ; retenu après rétrécissement −0,18 à
   +0,35 ; 0,07 sur 2002-2022). L'historique contredit d'ailleurs l'hypothèse d'un biais RN
   commun aux deux types de sondages : en 2002 le FN a été sous-estimé de 5 points au premier
   tour et surestimé de 3 points dans les duels ; en 2017 et 2022, surestimé dans les duels
   (4 à 5 points) quelle que soit l'erreur du premier tour. Ce qui est commun d'une élection à
   l'autre est le biais des duels lui-même, déjà corrigé. Effet nul aux horizons du premier
   tour (Δ CRPS +0,005 à +0,013, sous le bruit entre graines), dégradation à la veille du
   second tour (+0,14, 2002). **Non admis**, option désactivée.
3. **Participation du second tour tirée à part** : la loi logit-normale des seconds tours
   (72 à 75 % selon le pli) ignore la baisse de 2017-2022 (66 %) ; l'écart moyen à la
   participation observée baisse un peu avant le premier tour (7,8 à 6,9 points) mais monte à
   la veille (4,0 à 6,7 : les enquêtes de reports en savent plus que la loi historique). La
   part du vainqueur se dégrade à 180 et 90 jours (Δ CRPS +0,10 et +0,03). **Non admise**,
   option désactivée. Le défaut de calibration de la participation implicite reste ouvert ; il
   pèse peu sur la part (les sondages de duel la contraignent).

Options par défaut : `bias_by_horizon` / `rn_bias_curve`, `couple_far_right`,
`second_round_turnout` toutes désactivées (`simulate.RN_BIAS_CURVE`, `COUPLE_FAR_RIGHT`,
`SECOND_ROUND_TURNOUT`). Le run runs/2026-10-20-simulate-subst-v2 est la configuration
recommandée (`curve`), pas le défaut du code.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["eval", "simulate", "report"])
    ap.add_argument("--out", default=str(paths.RUNS / f"{dt.date.today().isoformat()}-second-round-v2"))
    ap.add_argument("--sim-out", default=str(paths.RUNS / f"{dt.date.today().isoformat()}-simulate-subst-v2"))
    ap.add_argument("--final", default="curve", choices=list(SIM_VARIANTS))
    ap.add_argument("--n", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.out)
    if args.step == "eval":
        run_eval(out, args.n)
    elif args.step == "simulate":
        run_simulations(out, Path(args.sim_out), args.final)
    report(out)
    print(out / "report.md")


if __name__ == "__main__":
    main()
