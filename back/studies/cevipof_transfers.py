"""Étude : reports de voix, duels hypothétiques et anticipations dans les enquêtes du Cevipof.

    uv run python studies/cevipof_transfers.py [--out runs/<date>-cevipof] [--bootstrap 2000]

Entrées : data/curated/cevipof_votes.parquet, cevipof_duels.parquet,
cevipof_expectations.parquet (`vvs parse cevipof`), les codebooks DDI et questionnaires
publics de data/raw/cevipof/ (`vvs ingest cevipof`), data/registry/transfers.yaml.

Critères fixés avant lecture des résultats (2026-10-22) :

(a) Matrices de reports, en % de l'électorat de premier tour (finalistes, blancs et nuls,
    abstention), refus et « ne sait pas » exclus et rapportés. Un électorat n'est publié seul
    que s'il compte au moins `N_ROW_MIN` = 30 répondants ; sinon il est regroupé avec les
    petits candidats de son bloc. Il n'entre dans la proposition pour transfers.yaml qu'à
    `N_PROPOSE_MIN` = 50 répondants. Intervalles bootstrap à 90 % (2 000 tirages ; par unité
    primaire pour les enquêtes en face à face). Comparaison aux instituts : une cellule
    d'institut est « compatible » si elle tombe dans l'intervalle à 90 % du Cevipof ; on
    compare aussi l'écart absolu moyen Cevipof - moyenne des instituts à l'écart moyen de
    chaque institut à la moyenne des autres (dispersion entre instituts). Deux bases : part
    du premier finaliste parmi les exprimés (toutes sources) et part « ni l'un ni l'autre »
    en % de l'électorat (sources split et combined).
(a') Concentration κ : pour chaque électorat ayant au moins trois sources (Cevipof compris),
    κ̂ = m(1 - m) / s² - 1 sur la répartition parmi les exprimés (moment d'une Beta). C'est
    une dispersion ENTRE SOURCES d'une même élection : elle borne par en dessous
    l'incertitude d'un report futur, elle ne la mesure pas.
(b) Duels hypothétiques : reports par électorat d'intention de premier tour (même vague),
    publiés pour les électorats d'au moins 30 répondants ; sinon inventaire seul.
(c) Anticipations : part qui juge probable (note 6 à 10 sur 10) la victoire ou la
    qualification du candidat effectivement vainqueur ou qualifié, par vague, contre la part
    d'intentions du même candidat dans la même vague (base exprimés). Sans microdonnées :
    marges brutes NON PONDÉRÉES des codebooks, signalées comme telles.
(d) Certitude et participation différentielle par candidat contre l'erreur finale des
    sondages : seulement avec les microdonnées (croisement intention x certitude).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from vvs import paths
from vvs.ingest.cevipof.codebook import load_codebook
from vvs.ingest.cevipof.parse import default_aliases, normalize
from vvs.ingest.cevipof.registry import default_inventory
from vvs.ingest.cevipof.transfers import (
    expressed_split,
    group_rows,
    none_share,
    pair_sources,
    transfer_matrix,
)
from vvs.registry import default_registry
from vvs.registry.transfers import default_transfers

N_ROW_MIN = 30
N_PROPOSE_MIN = 50
LEVEL = 0.9
LIKELY = 6  # note minimale (sur 10) pour « juge probable »
UNLIKELY = 4

# Petits candidats regroupés par bloc quand ils sont sous N_ROW_MIN (ids des groupes : ceux
# de transfers.yaml, avec `members`)
SMALL_GROUPS = {
    2017: {
        "small_far_left": ["arthaud_nathalie", "poutou_philippe"],
        "small_other": ["lassalle_jean", "asselineau_francois", "cheminade_jacques"],
    },
    2022: {
        "small_far_left": ["arthaud_nathalie", "poutou_philippe", "roussel_fabien"],
        "small_other": ["lassalle_jean"],
    },
}
SPECIAL_ROW = {"abstention": "abstain", "blank": "blank_null"}


def _f(x, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",")


def finalists(election: int) -> tuple[str, str]:
    e = default_registry().elections[election]
    a, b = (r.candidate_id for r in e.round_(2).results)
    return a, b


# --- (a) matrices


def institute_vectors(election: int, pair: tuple[str, str]) -> pd.DataFrame:
    """Une ligne par (source, électorat) : split parmi les exprimés, part ni-l'un-ni-l'autre."""
    rows = []
    a, b = pair
    for s in default_transfers():
        if s.election != election or s.base == "other":
            continue
        for r in s.rows:
            to = r.to
            if a not in to or b not in to:
                continue
            split = 100 * to[a] / (to[a] + to[b]) if to[a] + to[b] else np.nan
            none = np.nan
            if s.columns == "split" and {"blank_null", "abstain"} <= set(to):
                none = to["blank_null"] + to["abstain"]
            elif s.columns == "combined" and "blank_or_abstain" in to:
                none = to["blank_or_abstain"]
            rows.append({"source": s.id, "from": r.from_, "split": split, "none": none})
    return pd.DataFrame(rows)


def kappa_moment(vals_pct: list[float]) -> float:
    """κ d'une Beta par la méthode des moments : m(1 - m) / s² - 1 (parts en %)."""
    m = np.mean(vals_pct) / 100
    s2 = np.var(vals_pct, ddof=1) / 1e4
    return m * (1 - m) / s2 - 1 if s2 > 0 else np.inf


def run_matrices(
    votes: pl.DataFrame, out: Path, bootstrap: int
) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    results, comp_rows, kappa_rows = [], [], []
    for election in sorted(votes["election"].drop_nulls().unique().to_list()):
        pair = finalists(election)
        for src in pair_sources(votes, election):
            raw = transfer_matrix(src, pair)
            small = [c for c, n in raw.n.items() if n < N_ROW_MIN and c not in SPECIAL_ROW]
            groups = {
                g: [c for c in members if c in small]
                for g, members in SMALL_GROUPS.get(election, {}).items()
            }
            groups = {g: m for g, m in groups.items() if m}
            m = (
                group_rows(raw, groups, src, bootstrap=bootstrap)
                if groups
                else transfer_matrix(src, pair, bootstrap)
            )
            results.append((src, m, groups))
            tab = m.shares.round(2).copy()
            tab.columns = [f"pct_{c}" for c in tab.columns]
            for c in m.columns:
                tab[f"low_{c}"] = m.low[c].round(2) if m.low is not None else np.nan
                tab[f"high_{c}"] = m.high[c].round(2) if m.high is not None else np.nan
            tab["n"] = m.n
            tab["n_eff"] = m.n_eff.round(1)
            tab["excluded_pct"] = m.excluded_share.reindex(tab.index).round(2)
            tab.index.name = "from"
            tab.reset_index().to_csv(
                out / f"transfers_{election}_{src.label.replace(' ', '')}.csv", index=False
            )
            # comparaison aux instituts
            inst = institute_vectors(election, pair)
            split = m.interval(expressed_split(pair))
            none = m.interval(none_share)
            for frm in m.shares.index:
                key = SPECIAL_ROW.get(frm, frm)
                sub = inst[inst["from"] == key]
                for metric, (est, lo, hi) in (("split", split), ("none", none)):
                    vals = sub[["source", metric]].dropna()
                    for _, r in vals.iterrows():
                        comp_rows.append(
                            {
                                "election": election,
                                "cevipof_source": src.label,
                                "from": key,
                                "metric": metric,
                                "n": int(m.n[frm]),
                                "cevipof": est[frm],
                                "low": lo[frm],
                                "high": hi[frm],
                                "institute": r["source"],
                                "institute_value": r[metric],
                                "compatible": bool(lo[frm] <= r[metric] <= hi[frm]),
                            }
                        )
                vals = sub["split"].dropna().tolist() + [split[0][frm]]
                if len(vals) >= 3:
                    mu, s2 = np.mean(vals) / 100, np.var(vals, ddof=1) / 1e4
                    kappa_rows.append(
                        {
                            "election": election,
                            "from": key,
                            "n_sources": len(vals),
                            "mean_split": 100 * mu,
                            "sd_split": 100 * np.sqrt(s2),
                            "kappa_hat": kappa_moment(vals),
                            "kappa_hat_without_cevipof": kappa_moment(vals[:-1])
                            if len(vals) >= 4
                            else np.nan,
                        }
                    )
    comp = pd.DataFrame(comp_rows)
    kap = pd.DataFrame(kappa_rows)
    if len(comp):
        comp.to_csv(out / "comparison_institutes.csv", index=False)
    if len(kap):
        kap.to_csv(out / "kappa_between_sources.csv", index=False)
    return results, comp, kap


def dispersion_summary(comp: pd.DataFrame) -> pd.DataFrame:
    """Écart absolu moyen Cevipof - moyenne des instituts, contre l'écart de chaque institut à
    la moyenne des autres, par élection et métrique."""
    rows = []
    for (el, metric), g in comp.groupby(["election", "metric"]):
        cev, inst = [], []
        for frm, h in g.groupby("from"):
            v = h["institute_value"].to_numpy()
            cev.append(abs(h["cevipof"].iloc[0] - v.mean()))
            if len(v) >= 2:
                inst += [abs(v[i] - np.delete(v, i).mean()) for i in range(len(v))]
        rows.append(
            {
                "election": el,
                "metric": metric,
                "electorates": g["from"].nunique(),
                "cells": len(g),
                "compatible_share": g["compatible"].mean(),
                "mad_cevipof_vs_institutes": np.mean(cev),
                "mad_institute_vs_others": np.mean(inst) if inst else np.nan,
            }
        )
    return pd.DataFrame(rows)


def proposal(results, out: Path) -> dict:
    """Entrées au format de transfers.yaml (base split), électorats d'au moins N_PROPOSE_MIN."""
    entries = []
    inv = default_inventory()
    for src, m, groups in results:
        ds = inv.get(src.datasets[-1])
        a, b = m.finalists
        rows = []
        r1 = {
            r.candidate_id: r.share
            for r in default_registry().elections[src.election].round_(1).results
        }
        order = sorted(m.shares.index, key=lambda c: (c in SPECIAL_ROW, -r1.get(c, 0.0), c))
        for frm in order:
            if m.n[frm] < N_PROPOSE_MIN:
                continue
            s = m.shares.loc[frm]
            row = {"from": SPECIAL_ROW.get(frm, frm)}
            if frm in groups:
                row["members"] = groups[frm]
            row["to"] = {
                a: round(float(s[a]), 1),
                b: round(float(s[b]), 1),
                "blank_null": round(float(s["blank"]), 1),
                "abstain": round(float(s["abstention"]), 1),
            }
            rows.append(row)
        n_tot = int(m.n.sum())
        entries.append(
            {
                "id": f"cevipof_{src.election}"
                if len(results) == 1
                else f"cevipof_{src.election}_{src.label}",
                "election": src.election,
                "pollster": "cevipof",
                "title": f"Cevipof, {ds.title} : reports de voix recalculés sur les microdonnées",
                "url": ds.url,
                "pages": f"microdonnées {', '.join(src.datasets)} ; calcul vvs.ingest.cevipof.transfers (étude back/studies/cevipof_transfers.py, {dt.date.today().isoformat()})",
                "fieldwork_start": ds.fieldwork_start,
                "fieldwork_end": ds.fieldwork_end,
                "sample_size": n_tot,
                "mode": None,  # face à face (CAPI) : hors des modes admis par transfers.yaml
                "base": "first_round_electorate",
                "base_note": (
                    "En % des électeurs de chaque électorat de premier tour (vote déclaré), lignes "
                    "sommant à 100 avec blancs ou nuls et abstention au second tour (vote déclaré). "
                    "Pondération de l'enquête calée sur les votes des deux tours ; répondants qui "
                    "refusent ou ne savent pas à l'un des deux tours exclus. Effectif : répondants "
                    "retenus dans la matrice."
                ),
                "columns": "split",
                "notes": (
                    f"Valeurs à une décimale, calculées et non recopiées. Électorats de moins de "
                    f"{N_PROPOSE_MIN} répondants omis ; effectifs et intervalles bootstrap à 90 % "
                    "dans runs/2026-10-22-cevipof/. Enquête post-électorale en face à face "
                    "(mode CAPI, `mode` laissé vide) : souvenir du vote quelques semaines après, "
                    "sujet à la reconstruction du vote. Le poids n'est calé que sur les votes "
                    "exprimés : l'abstention au second tour est sous-représentée et le vote blanc "
                    "ou nul surdéclaré (marges pondérées contre résultats officiels dans le "
                    "rapport) ; la répartition entre finalistes (base `expressed` de "
                    "transfer_prior) est la partie la plus fiable de la matrice."
                ),
                "rows": rows,
            }
        )
    return {"transfers": entries}


class _ProposalDumper(yaml.SafeDumper):
    """Cellules `to` en style compact, comme dans transfers.yaml."""


def _represent_dict(dumper, data):
    flow = set(data) <= {"macron_emmanuel", "le_pen_marine", "blank_null", "abstain"} or (
        "blank_null" in data and "abstain" in data
    )
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=flow)


_ProposalDumper.add_representer(dict, _represent_dict)


# --- (b) duels


def duel_inventory(out: Path) -> pd.DataFrame:
    """Duels posés : variables configurées (codebooks ENEF 2017, PEF) et questionnaires
    ENEF 2022 ; marges brutes non pondérées quand le codebook est public."""
    inv = default_inventory()
    al = default_aliases()
    rows = []
    for key, spec in al.datasets.items():
        ds = inv.get(key)
        cb = {v.name: v for v in load_codebook(ds)}
        for d in spec.duels:
            v = cb.get(d.var)
            counts = {}
            if v is not None:
                for c in v.categories:
                    ok, code = al.code(c.label, spec)
                    if ok and code is not None and c.freq is not None:
                        counts[code] = counts.get(code, 0) + c.freq
            a, b = d.options[0], d.options[1]
            ab = counts.get(a, 0) + counts.get(b, 0)
            rows.append(
                {
                    "dataset": key,
                    "election": ds.election,
                    "fieldwork_start": ds.fieldwork_start,
                    "office": d.office,
                    "variable": d.var,
                    "option_a": a,
                    "option_b": b,
                    "n_raw": sum(counts.values()) or None,
                    "a_share_expressed_unweighted": 100 * counts.get(a, 0) / ab if ab else None,
                    "source": "codebook DDI",
                }
            )
    rows += enef2022_questionnaire_duels()
    t = pd.DataFrame(rows)
    t.to_csv(out / "duels_inventory.csv", index=False)
    return t


NAME_TO_ID = {
    "emmanuel macron": "macron_emmanuel",
    "marine le pen": "le_pen_marine",
    "valerie pecresse": "pecresse_valerie",
    "eric zemmour": "zemmour_eric",
    "jean luc melenchon": "melenchon_jean_luc",
    "xavier bertrand": "bertrand_xavier",
    "yannick jadot": "jadot_yannick",
    "anne hidalgo": "hidalgo_anne",
}


def enef2022_questionnaire_duels() -> list[dict]:
    """Blocs IV2x des questionnaires ENEF 2022 (PDF publics) : duel posé et mention « hypothèse
    non traitée » (absente de la base)."""
    import pdfplumber

    inv = default_inventory()
    rows = []
    for ds in inv.datasets:
        if ds.series != "enef2022":
            continue
        q = [f for f in ds.files if f.role == "questionnaire" and not f.restricted]
        if not q:
            continue
        path = paths.RAW / "cevipof" / ds.key / q[0].name.strip()
        if not path.exists():
            continue
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            name = re.sub(r"\s*\[NEW\]$", "", line.strip())
            if not re.fullmatch(r"IV2[A-Z]{1,2}", name):
                continue
            block = lines[i + 1 : i + 16]
            opts = [
                normalize(re.sub(r"^\d+\.\s*", "", x))
                for x in block
                if re.match(r"^[12]\.\s", x.strip())
            ]
            ids = [NAME_TO_ID.get(o, o) for o in opts[:2]]
            dropped = any("pas été traitée" in x for x in block)
            if len(ids) == 2:
                rows.append(
                    {
                        "dataset": ds.key,
                        "election": 2022,
                        "fieldwork_start": ds.fieldwork_start,
                        "office": "president",
                        "variable": name,
                        "option_a": ids[0],
                        "option_b": ids[1],
                        "n_raw": None,
                        "a_share_expressed_unweighted": None,
                        "source": "questionnaire PDF"
                        + (" (hypothèse non traitée, absente de la base)" if dropped else ""),
                    }
                )
    return rows


def duel_transfers(duels: pl.DataFrame, votes: pl.DataFrame, out: Path) -> pd.DataFrame | None:
    """Reports par électorat d'intention de premier tour dans chaque duel posé (microdonnées)."""
    if duels.is_empty():
        return None
    v = votes.select("dataset", "respondent_id", "vote_round1", "vote_round1_kind").to_pandas()
    d = duels.to_pandas().merge(v, on=["dataset", "respondent_id"], how="left")
    rows = []
    for (key, var), g in d.groupby(["dataset", "variable"]):
        g = g[~g["choice"].isin(["refused", "undecided"]) & g["vote_round1"].notna()]
        for frm, h in g.groupby("vote_round1"):
            if len(h) < N_ROW_MIN:
                continue
            w = h.groupby("choice")["weight"].sum()
            tot = w.sum()
            rows.append(
                {
                    "dataset": key,
                    "variable": var,
                    "option_a": h["option_a"].iloc[0],
                    "option_b": h["option_b"].iloc[0],
                    "from": frm,
                    "n": len(h),
                    **{f"pct_{k}": 100 * x / tot for k, x in w.items()},
                }
            )
    t = pd.DataFrame(rows)
    t.to_csv(out / "duel_transfers.csv", index=False)
    return t


# --- (c) anticipations


def expectations_from_codebooks(out: Path) -> pd.DataFrame:
    """Marges brutes (non pondérées) des questions d'anticipation 0-10 configurées."""
    inv = default_inventory()
    al = default_aliases()
    reg = default_registry()
    rows = []
    for key, spec in al.datasets.items():
        ds = inv.get(key)
        cb = {v.name: v for v in load_codebook(ds)}
        el = reg.elections.get(ds.election)
        winner = el.round_(2).results[0].candidate_id if el else None
        qualified = {r.candidate_id for r in el.round_(2).results} if el else set()
        intention = {}
        if spec.vote_round1 and spec.vote_round1.kind == "intention" and spec.vote_round1.var in cb:
            tot = 0
            for c in cb[spec.vote_round1.var].categories:
                ok, code = al.code(c.label, spec)
                if (
                    ok
                    and code
                    and code not in ("abstention", "blank", "undecided", "refused")
                    and c.freq
                ):
                    intention[code] = intention.get(code, 0) + c.freq
                    tot += c.freq
            intention = {k: 100 * x / tot for k, x in intention.items()} if tot else {}
        duel_share = {}
        for d in spec.duels:
            if d.var not in cb:
                continue
            cnt = {}
            for c in cb[d.var].categories:
                ok, code = al.code(c.label, spec)
                if ok and code in d.options[:2] and c.freq:
                    cnt[code] = cnt.get(code, 0) + c.freq
            if sum(cnt.values()):
                duel_share[frozenset(d.options[:2])] = {
                    k: 100 * x / sum(cnt.values()) for k, x in cnt.items()
                }
        for e in spec.expectations:
            v = cb.get(e.var)
            if v is None:
                continue
            vals = {
                int(c.value): c.freq
                for c in v.categories
                if c.value.isdigit() and int(c.value) <= 10 and c.freq is not None
            }
            n = sum(vals.values())
            if not n:
                continue
            mean = sum(k * f for k, f in vals.items()) / n
            likely = 100 * sum(f for k, f in vals.items() if k >= LIKELY) / n
            unlikely = 100 * sum(f for k, f in vals.items() if k <= UNLIKELY) / n
            if e.question == "win_duel":
                truth = e.target == winner if {e.target, e.opponent} == qualified else None
                ref = duel_share.get(frozenset((e.target, e.opponent)), {}).get(e.target)
            else:
                truth = e.target in qualified if el else None
                ref = intention.get(e.target)
            rows.append(
                {
                    "dataset": key,
                    "fieldwork_start": ds.fieldwork_start,
                    "variable": e.var,
                    "question": e.question,
                    "target": e.target,
                    "opponent": e.opponent,
                    "n_raw": n,
                    "mean_0_10": mean,
                    "pct_likely_6_10": likely,
                    "pct_unlikely_0_4": unlikely,
                    "happened": truth,
                    "intention_share_unweighted": ref,
                }
            )
    t = pd.DataFrame(rows)
    t.to_csv(out / "expectations_codebook_unweighted.csv", index=False)
    return t


# --- rapport console


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=Path, default=paths.RUNS / f"{dt.date.today().isoformat()}-cevipof"
    )
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    votes_path = paths.CURATED / "cevipof_votes.parquet"
    votes = pl.read_parquet(votes_path) if votes_path.exists() else pl.DataFrame()
    print(
        f"cevipof_votes : {votes.height} lignes, jeux {sorted(set(votes['dataset'])) if votes.height else []}"
    )

    if votes.height:
        results, comp, kap = run_matrices(votes, out, args.bootstrap)
        for src, m, groups in results:
            print(
                f"\n== (a) {src.election} {src.label} ({int(m.n.sum())} répondants retenus) ; groupes {groups}"
            )
            t = m.shares.round(1)
            if m.low is not None:
                for c in m.columns:
                    t[c] = [
                        f"{_f(x)} [{_f(lo, 0)}-{_f(hi, 0)}]"
                        for x, lo, hi in zip(m.shares[c], m.low[c], m.high[c], strict=True)
                    ]
            t["n"] = m.n
            t["excl %"] = m.excluded_share.reindex(t.index).round(1)
            print(t.sort_values("n", ascending=False).to_string())
        if len(comp):
            print("\n== (a) comparaison aux instituts")
            show = comp.copy()
            for c in ("cevipof", "low", "high", "institute_value"):
                show[c] = show[c].round(1)
            print(show.to_string(index=False))
            summ = dispersion_summary(comp)
            summ.to_csv(out / "comparison_summary.csv", index=False)
            print(summ.round(2).to_string(index=False))
        if len(kap):
            print("\n== (a') κ entre sources")
            print(kap.round(2).to_string(index=False))
        prop = proposal(results, out)
        (out / "transfers_proposal.yaml").write_text(
            "# Proposition de sources Cevipof pour data/registry/transfers.yaml (à relire avant\n"
            "# intégration). Générée par back/studies/cevipof_transfers.py ; validée par\n"
            "# vvs.registry.transfers.load_transfers.\n\n"
            + yaml.dump(
                prop, Dumper=_ProposalDumper, sort_keys=False, allow_unicode=True, width=100
            ),
            encoding="utf-8",
        )
        from vvs.registry.transfers import load_transfers

        load_transfers(out / "transfers_proposal.yaml")
        print(f"\ntransfers_proposal.yaml : {len(prop['transfers'])} source(s), validée")

    print("\n== (b) duels hypothétiques")
    inv_d = duel_inventory(out)
    print(inv_d.round(1).to_string(index=False))
    duels_path = paths.CURATED / "cevipof_duels.parquet"
    if duels_path.exists():
        t = duel_transfers(pl.read_parquet(duels_path), votes, out)
        if t is not None:
            print(t.round(1).to_string(index=False))
    else:
        print("pas de microdonnées de duel (cevipof_duels.parquet absent)")

    print("\n== (c) anticipations (marges brutes non pondérées des codebooks)")
    exp = expectations_from_codebooks(out)
    print(exp.round(1).to_string(index=False))

    print("\n== (d) certitude et participation différentielle")
    if votes.height and votes["certainty_round1"].drop_nulls().len():
        print("certitude disponible : voir cevipof_votes.parquet")
    else:
        print("non réalisable : aucune microdonnée d'intention avec certitude accessible")


if __name__ == "__main__":
    main()
