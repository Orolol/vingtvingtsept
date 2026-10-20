"""Run de simulation jouet pour test_simulate.py et test_export.py : un faux ajustement du
premier tour (tirages de θ écrits comme par `run_fit(save_trajectories=True)`), trois
dimensions de scénarios, un modèle d'erreur η simplifié. Le second tour est le vrai modèle
(registre des reports et sondages de duel curated), avec des erreurs de duel par défaut."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from vvs.model.second_round import ErrorParams
from vvs.registry.scenarios import ScenariosFile

ASOF = dt.date(2026, 10, 20)
FIT_CANDIDATES = {
    # candidat : (bloc, part de référence en %, première semaine testée)
    "arthaud_nathalie": ("far_left", 1.0, 1),
    "bardella_jordan": ("far_right", 33.0, 2),
    "le_pen_marine": ("far_right", 34.0, 1),
    "melenchon_jean_luc": ("far_left", 16.0, 1),
    "philippe_edouard": ("centre", 18.0, 1),
    "retailleau_bruno": ("right", 9.0, 1),
}
T, ASOF_WEEK, S = 6, 4, 300
POLLSTERS = ["elabe", "ifop"]


def toy_scenarios() -> ScenariosFile:
    def state(i, label, p, cands, placeholders=()):
        return {"id": i, "label": label, "probability": p, "description": "-",
                "candidates": list(cands), "placeholders": list(placeholders)}

    return ScenariosFile.model_validate(
        {
            "election": 2027,
            "as_of": "2026-10-20",
            "dimensions": [
                {"id": "rn", "label": "RN", "question": "?", "states": [
                    state("le_pen", "Le Pen", 0.8, ["le_pen_marine"]),
                    state("bardella", "Bardella", 0.2, ["bardella_jordan"])]},
                {"id": "centre", "label": "Centre", "question": "?", "states": [
                    state("philippe", "Philippe", 0.9, ["philippe_edouard"]),
                    state("other", "Autre", 0.1, [], ["centre"])]},
                {"id": "lr", "label": "LR", "question": "?", "states": [
                    state("retailleau", "Retailleau", 0.7, ["retailleau_bruno"]),
                    state("rally", "Ralliement", 0.3, [])]},
            ],
            "small_candidates": [
                {"candidate_id": "melenchon_jean_luc", "probability": 0.95, "bloc": "far_left",
                 "rationale": "-"},
                {"candidate_id": "arthaud_nathalie", "probability": 0.9, "bloc": "far_left",
                 "rationale": "-"},
                {"candidate_id": "roussel_fabien", "probability": 0.5, "bloc": "far_left",
                 "rationale": "-"},
                {"candidate_id": "lisnard_david", "probability": 0.4, "bloc": "right",
                 "rationale": "-"},
            ],
            "constraints": [
                {"id": "c", "when": {"lr": "rally"}, "absent": ["lisnard_david"],
                 "rationale": "-"},
            ],
            "revisions": [{"date": "2026-10-20", "author": "test", "summary": "-",
                           "rationale": "-"}],
        }
    )


def make_fit_dir(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    fit_dir = Path(root) / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    cands = sorted(FIT_CANDIDATES)
    base = np.log([FIT_CANDIDATES[c][1] for c in cands])
    theta = base + 0.05 * rng.standard_normal((S, 1, len(cands)))
    theta = theta + np.cumsum(0.02 * rng.standard_normal((S, T, len(cands))), axis=1)
    alpha = 0.03 * rng.standard_normal((S, len(POLLSTERS), len(cands)))
    # milieu de semaine, la semaine ASOF_WEEK contenant ASOF (comme PollData.week_date)
    weeks = [ASOF + dt.timedelta(days=1 + 7 * (k - ASOF_WEEK)) for k in range(T)]
    np.savez_compressed(
        fit_dir / "draws.npz",
        theta_asof=theta[:, ASOF_WEEK].astype(np.float32),
        theta_vote=theta[:, -1].astype(np.float32),
        candidates=np.array(cands),
        blocs=np.array([FIT_CANDIDATES[c][0] for c in cands]),
        theta=theta.astype(np.float32),
        alpha=alpha.astype(np.float32),
        pollsters=np.array(POLLSTERS),
        week_dates=np.array([w.isoformat() for w in weeks]),
        asof_week=np.array(ASOF_WEEK),
    )
    rows = []
    for w in range(1, ASOF_WEEK + 1):
        for j, pol in enumerate(POLLSTERS):
            present = [c for c in cands if FIT_CANDIDATES[c][2] <= w and not (
                c == "bardella_jordan" and j == 0)]
            sh = np.array([FIT_CANDIDATES[c][1] for c in present])
            sh = sh / sh.sum()
            pid = f"2027-{pol}-w{w}"
            for c, s in zip(present, sh):
                rows.append({"poll_id": pid, "hypothesis_id": "+".join(present), "pollster": pol,
                             "week": w, "date": weeks[w], "candidate_id": c, "share": float(s)})
    pl.DataFrame(rows).write_parquet(fit_dir / "observed.parquet")
    manifest = {
        "commit": None,
        "data": {"n_polls": 2 * ASOF_WEEK, "n_hypotheses": 2 * ASOF_WEEK,
                 "reference": [c for c in cands if c != "bardella_jordan"]},
        "diagnostics": {"r_hat_max": 1.0, "ess_min": 1000.0, "divergences": 0},
        "posterior": {},
    }
    (fit_dir / "manifest.json").write_text(json.dumps(manifest))
    return fit_dir


class StubErrorModel:
    """η gaussien de bloc et de candidat (échelle logit), sans dépendance aux données."""

    def sample_eta(self, horizon, blocs, p, rng):
        n, k = np.atleast_2d(p).shape
        ub = sorted(set(blocs))
        u = 0.15 * rng.standard_normal((n, len(ub)))
        return u[:, [ub.index(b) for b in blocs]] + 0.2 * rng.standard_normal((n, k))

    def to_dict(self):
        return {"params": {"stub": True}}


def run_toy(root: Path, n: int = 3000, seed: int = 0, out: Path | None = None):
    from vvs.model.simulate import simulate

    fit_dir = make_fit_dir(root, seed)
    return simulate(
        fit_dir,
        ASOF,
        out=out,
        n_draws=n,
        seed=seed,
        scenarios=toy_scenarios(),
        error_model=StubErrorModel(),
        second_round_errors=ErrorParams(bias_far_right=3.0),
        trajectory_draws=200,
    )
