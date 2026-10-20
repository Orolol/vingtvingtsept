"""Point d'entrée `uv run vvs …`. Les sous-commandes s'enregistrent ici au fil des phases."""

from __future__ import annotations

from typing import Annotated

import typer

app = typer.Typer(help="vingtvingtsept — prévision de la présidentielle 2027", no_args_is_help=True)
ingest_app = typer.Typer(help="Téléchargement des sources vers data/raw", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
parse_app = typer.Typer(help="Extraction data/raw -> data/curated", no_args_is_help=True)
app.add_typer(parse_app, name="parse")


@app.command()
def version() -> None:
    from vvs import __doc__ as doc

    typer.echo(doc)


@ingest_app.command("commission")
def ingest_commission(
    since: str = typer.Option("2025-01-01", help="Mois de dépôt minimal (AAAA-MM-JJ)"),
) -> None:
    """Notices de la Commission des sondages (présidentielle 2027) -> data/raw/commission."""
    from datetime import date

    from vvs.ingest.commission.listing import ingest

    stats = ingest(date.fromisoformat(since))
    typer.echo(
        f"{stats['listed']} notices présidentielles listées, {stats['kept']} depuis {since} : "
        f"{stats['downloaded']} téléchargées, {stats['skipped']} déjà présentes, "
        f"{len(stats['failed'])} en échec"
    )
    for file_id, title, status in stats["failed"]:
        typer.echo(f"  échec HTTP {status} : {file_id} {title}", err=True)


@parse_app.command("commission")
def parse_commission(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Détail par notice"),
) -> None:
    """Notices brutes -> polls_commission.parquet et poll_results_commission.parquet.

    Code de sortie 1 si un tableau d'intentions de vote contient un libellé inconnu (la
    notice est alors exclue ; compléter la table d'alias de vvs.ingest.commission.aliases).
    """
    from vvs.ingest.commission.parse import run

    report = run()
    typer.echo(report.summary(verbose=verbose))
    if report.has_unknown_labels:
        raise typer.Exit(code=1)


@ingest_app.command("wikipedia")
def ingest_wikipedia(
    election: int = typer.Option(..., help="Année de l'élection (ex. 2022)"),
) -> None:
    """Télécharge la liste de sondages Wikipédia et la stocke brute dans data/raw."""
    from vvs.ingest.wikipedia.common import fetch_page, store_raw

    path = store_raw(election, fetch_page(election))
    typer.echo(f"HTML brut : {path}")


@parse_app.command("wikipedia")
def parse_wikipedia(
    election: int = typer.Option(..., help="Année de l'élection (ex. 2022)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Liste chaque ligne ignorée"),
) -> None:
    """Parse le dernier HTML brut Wikipédia et écrit polls/poll_results en parquet."""
    from vvs import paths
    from vvs.ingest.wikipedia import parse_with_report
    from vvs.ingest.wikipedia.common import latest_raw

    raw = latest_raw(election)
    polls, results, report = parse_with_report(election, raw.read_text(encoding="utf-8"))
    paths.CURATED.mkdir(parents=True, exist_ok=True)
    polls_path = paths.CURATED / f"polls_wikipedia_{election}.parquet"
    results_path = paths.CURATED / f"poll_results_wikipedia_{election}.parquet"
    polls.write_parquet(polls_path)
    results.write_parquet(results_path)
    typer.echo(f"source : {raw}")
    typer.echo(report.summary())
    if verbose:
        for s in report.skipped_rows:
            typer.echo(f"  ignorée : {s}")
    typer.echo(f"écrit : {polls_path} ({polls.height} sondages)")
    typer.echo(f"écrit : {results_path} ({results.height} lignes)")


@ingest_app.command("interieur")
def ingest_interieur(
    election: str = typer.Option(..., help="Clé de data/registry/interieur_sources.yaml (2022, euro2019…)"),
) -> None:
    """Résultats officiels du ministère de l'Intérieur -> data/raw/results/<election>."""
    from vvs.ingest.interieur import ingest

    for name, status in ingest(election):
        typer.echo(f"{status:10s} {name}")


@parse_app.command("interieur")
def parse_interieur(
    election: str = typer.Option(..., help="Clé de data/registry/interieur_sources.yaml (2022, euro2019…)"),
) -> None:
    """data/raw/results/<election> -> data/curated/results[_lists]_<election>.parquet."""
    from vvs.ingest.interieur import parse

    out, df = parse(election)
    counts = df.group_by("geo_level").len().sort("geo_level").iter_rows()
    typer.echo(f"écrit : {out} ({df.height} lignes : " + ", ".join(f"{k} {n}" for k, n in counts) + ")")



@ingest_app.command("macro")
def ingest_macro(
    series: Annotated[
        list[str] | None,
        typer.Option("--series", help="series_id de data/registry/macro_series.yaml (répétable ; toutes par défaut)"),
    ] = None,
) -> None:
    """Séries économiques et de popularité -> data/raw/macro/<série>/<date>.<ext>."""
    from vvs.ingest.macro import ingest

    ingest(series or None, log=typer.echo)


@parse_app.command("macro")
def parse_macro(
    series: Annotated[
        list[str] | None,
        typer.Option("--series", help="series_id à reparser (les autres lignes du parquet sont conservées)"),
    ] = None,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Détaille trous, désaccords et PDF Ifop"),
) -> None:
    """Derniers fichiers bruts de data/raw/macro -> data/curated/macro.parquet."""
    from vvs import paths
    from vvs.ingest.macro import OUTPUT_NAME, parse

    df, report = parse(series or None)
    typer.echo(report.summary())
    if verbose:
        for sid, gaps in report.gaps.items():
            if gaps:
                typer.echo(f"  trous {sid} : " + ", ".join(d.isoformat() for d in gaps))
        if report.ifop is not None:
            for c in report.ifop.conflicts:
                typer.echo(f"  désaccord Ifop : {c}")
            for r in report.ifop.report.rows_out_of_tenure:
                typer.echo(f"  hors mandat : {r}")
            for r in report.ifop.report.headlines_unreadable:
                typer.echo(f"  chiffre du mois illisible : {r}")
            for p in report.ifop.publications:
                typer.echo(f"  PDF {p['month']} {p['mode']:7s} {p['sponsor']:12s} tableaux {p['n_table']:4d} titre {p['n_headline']} {p['file']}")
    typer.echo(f"écrit : {paths.CURATED / OUTPUT_NAME} ({df.height} lignes)")


@ingest_app.command("mieuxvoter")
def ingest_mieuxvoter() -> None:
    """Dépôt MieuxVoter/presidentielle2027 (JSON + polls.csv) -> data/raw/mieuxvoter/<date>.*"""
    from vvs.ingest.mieuxvoter import ingest

    path, written = ingest()
    typer.echo(f"{'écrit' if written else 'déjà présent (non réécrit)'} : {path}")


@parse_app.command("mieuxvoter")
def parse_mieuxvoter(
    compare: bool = typer.Option(False, "--compare", help="Contrôle croisé avec Wikipédia 2027"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Détail du contrôle croisé"),
) -> None:
    """Dernier instantané MieuxVoter -> polls_mieuxvoter.parquet et poll_results_mieuxvoter.parquet."""
    from vvs.ingest.mieuxvoter import compare_with_wikipedia, run

    polls_path, results_path, report = run()
    typer.echo(report.summary())
    typer.echo(f"écrit : {polls_path}")
    typer.echo(f"écrit : {results_path}")
    if compare:
        typer.echo(compare_with_wikipedia().summary(verbose=verbose))


@ingest_app.command("jennings-wlezien")
def ingest_jennings_wlezien() -> None:
    """Base Jennings et Wlezien (Dataverse) -> data/raw/jennings_wlezien (idempotent)."""
    from vvs.ingest.jennings_wlezien import ingest

    for name, status, digest in ingest():
        typer.echo(f"{status:10s} {name} sha256={digest}")


@parse_app.command("jennings-wlezien")
def parse_jennings_wlezien(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Liste les partis sans libellé"),
    summary: bool = typer.Option(False, "--summary", help="Affiche l'erreur par horizon"),
) -> None:
    """data/raw/jennings_wlezien -> data/curated/jw_polls.parquet (+ erreur par horizon)."""
    from vvs.ingest.jennings_wlezien import error_summary, parse

    out, df, report = parse()
    typer.echo(report.summary(verbose=verbose))
    typer.echo(f"écrit : {out} ({df.height} lignes)")
    if summary:
        for title, kw in [
            ("tous pays, toutes élections", {}),
            ("France, premier tour", {"country": "FR", "round": 1}),
            ("France, second tour", {"country": "FR", "round": 2}),
        ]:
            typer.echo(f"\n{title} :\n{error_summary(df=df, **kw)}")


@ingest_app.command("press")
def ingest_press(
    source: str = typer.Option(..., help="gdelt, ina ou mediacloud"),
    query: Annotated[
        list[str] | None,
        typer.Option("--query", help="query_id de press_queries.yaml (répétable ; candidats par défaut)"),
    ] = None,
    mode: str = typer.Option("timeline", help="gdelt : timeline ou artlist ; mediacloud : counts ou list"),
    start: str = typer.Option(None, help="Début AAAA-MM-JJ (gdelt timeline : 2017-01-01 par défaut)"),
    end: str = typer.Option(None, help="Fin incluse AAAA-MM-JJ (défaut : hier)"),
    window_months: int = typer.Option(12, help="gdelt timeline : taille des fenêtres en mois"),
    tone: bool = typer.Option(True, help="gdelt timeline : demander aussi timelinetone"),
    coverage: str = typer.Option(None, help="mediacloud : années à mesurer, ex. 2012,2017,2022"),
) -> None:
    """Presse -> data/raw/press/<source> (cache immuable ; relancer reprend où on s'est arrêté)."""
    from datetime import date, timedelta

    from vvs.ingest.press.outlets import load_outlets, load_queries

    queries = load_queries()
    qids = query or list(queries.candidates)
    d0 = date.fromisoformat(start) if start else None
    d1 = date.fromisoformat(end) if end else date.today() - timedelta(days=1)
    if source == "ina":
        from vvs.ingest.press import ina

        path, status = ina.ingest()
        typer.echo(f"{status} : {path}")
    elif source == "gdelt":
        from vvs.ingest.press import gdelt

        if mode == "timeline":
            modes = gdelt.TIMELINE_MODES if tone else ("timelinevolraw",)
            stats = gdelt.ingest_timelines(
                queries, qids, modes, d0, d1, window_months=window_months, log=typer.echo
            )
        elif mode == "artlist":
            if d0 is None or end is None:
                raise typer.BadParameter("artlist exige --start et --end (fenêtres ciblées)")
            stats = gdelt.ingest_articles(queries, qids, d0, d1, log=typer.echo)
        else:
            raise typer.BadParameter(f"mode gdelt inconnu : {mode}")
        typer.echo(stats.summary())
        for qid, window, err in stats.failed:
            typer.echo(f"  échec {qid} {window} : {err}", err=True)
        if stats.rate_limited:  # code 3 : relancer plus tard reprend depuis le cache
            raise typer.Exit(code=3)
        if stats.failed:
            raise typer.Exit(code=1)
    elif source == "mediacloud":
        from vvs.ingest.press import mediacloud as mc

        try:
            client = mc.MediaCloudClient(log=typer.echo)
        except mc.MediaCloudKeyMissing as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=2) from None
        if coverage:
            years = [int(y) for y in coverage.split(",")]
            df = mc.measure_coverage(years, client=client, outlets=load_outlets())
            typer.echo(f"écrit : {mc.write_coverage(df)} ({df.height} lignes)")
        elif mode == "counts":
            for qid in qids:
                q = queries.all_queries()[qid]
                df = mc.daily_counts(client, q, d0 or date(d1.year, 1, 1), d1)
                typer.echo(f"{qid} : {df['count'].sum()} titres sur {df.height} jours")
        elif mode == "list":
            if d0 is None:
                raise typer.BadParameter("list exige --start")
            n = mc.ingest_story_lists(queries, qids, d0, d1, client=client, log=typer.echo)
            typer.echo(f"{n} titres listés")
        else:
            raise typer.BadParameter(f"mode mediacloud inconnu : {mode}")
    else:
        raise typer.BadParameter(f"source inconnue : {source}")


@parse_app.command("press")
def parse_press(
    source: Annotated[
        list[str] | None,
        typer.Option("--source", help="gdelt, ina, mediacloud (répétable ; toutes par défaut)"),
    ] = None,
) -> None:
    """data/raw/press -> press_timelines, press_ina et press_items (data/curated)."""
    from vvs.ingest.press import parse_all

    typer.echo(parse_all(source or None).summary())


@app.command("fit")
def fit_cmd(
    election: int = typer.Option(..., help="Année de l'élection (ex. 2022)"),
    asof: str = typer.Option(..., help="Date de prévision AAAA-MM-JJ (sondages terminés à cette date)"),
    out: str = typer.Option(None, help="Dossier de sortie (défaut runs/<aujourd'hui>-fit-<élection>)"),
    weeks: int = typer.Option(78, help="Profondeur de la grille, en semaines avant le vote"),
    step_days: int = typer.Option(7, help="Pas de temps de la grille, en jours"),
    method: str = typer.Option("nuts", help="nuts (référence) ou svi (approximation rapide)"),
    chains: int = typer.Option(4, help="Chaînes NUTS"),
    warmup: int = typer.Option(1000, help="Itérations de chauffe par chaîne"),
    samples: int = typer.Option(1000, help="Tirages conservés par chaîne"),
    seed: int = typer.Option(0, help="Graine"),
    centered: bool = typer.Option(
        False, "--centered", help="Marches en paramétrisation centrée (2× plus rapide, voir fit)"
    ),
    progress: bool = typer.Option(False, "--progress", help="Barre de progression numpyro"),
) -> None:
    """Agrégation des sondages de premier tour (docs/04-modele.md §1) -> runs/…"""
    from datetime import date
    from pathlib import Path

    from vvs import paths
    from vvs.model.run import run_fit

    out_dir = Path(out) if out else paths.RUNS / f"{date.today().isoformat()}-fit-{election}"
    m = run_fit(
        election,
        date.fromisoformat(asof),
        out_dir,
        weeks=weeks,
        step_days=step_days,
        method=method,
        chains=chains,
        warmup=warmup,
        samples=samples,
        seed=seed,
        centered=centered,
        progress_bar=progress,
    )
    d, diag = m["data"], m["diagnostics"]
    typer.echo(
        f"{d['n_polls']} sondages, {d['n_hypotheses']} hypothèses, {d['n_candidates']} candidats "
        f"(source {d['source']}) ; inférence {m['duration_s']['inference']} s"
    )
    if "r_hat_max" in diag:
        typer.echo(
            f"r_hat max {diag['r_hat_max']:.3f}, ESS min {diag['ess_min']:.0f}, "
            f"divergences {diag['divergences']}"
        )
    if d["dropped_candidates"]:
        typer.echo(f"candidats hors registre ignorés : {d['dropped_candidates']}", err=True)
    typer.echo(f"écrit : {out_dir}")


@app.command("backtest")
def backtest_cmd(
    elections: str = typer.Option("2012,2017,2022", help="Élections, séparées par des virgules"),
    horizons: str = typer.Option("180,90,30,7", help="Horizons en jours avant le premier tour"),
    draws: int = typer.Option(10_000, help="Tirages de parts au jour du vote"),
    jobs: int = typer.Option(3, help="Ajustements en parallèle (4 chaînes chacun)"),
    out: str = typer.Option(None, help="Dossier de sortie (défaut runs/<aujourd'hui>-backtest)"),
    variant: str = typer.Option("floor", help="Variante du modèle (vvs.eval.backtest.VARIANTS)"),
    stage: str = typer.Option("all", help="fit, score ou all"),
    method: str = typer.Option("nuts", help="nuts ou svi (essais rapides)"),
    warmup: int = typer.Option(500, help="Chauffe NUTS par chaîne"),
    samples: int = typer.Option(500, help="Tirages NUTS conservés par chaîne"),
) -> None:
    """Backtest du premier tour (docs/07-evaluation.md) -> runs/<date>-backtest/."""
    from pathlib import Path

    from vvs.eval import backtest as bt

    out_dir = Path(out) if out else bt.default_out()
    els = [int(x) for x in elections.split(",") if x]
    hs = [int(x) for x in horizons.split(",") if x]
    if stage in ("fit", "all"):
        plan = bt.plan_fits(out_dir, els, hs, variant)
        settings = bt.FitSettings(method=method, warmup=warmup, samples=samples)
        failed = bt.run_fits(plan, out_dir, settings, n_parallel=jobs)
        if failed:
            typer.echo(f"{len(failed)} ajustements en échec (voir {out_dir}/backtest.log)", err=True)
    if stage in ("score", "all"):
        res = bt.score(out_dir, els, hs, variant=variant, n_draws=draws)
        typer.echo(res.summary())
    typer.echo(f"écrit : {out_dir}")


@app.command("second-round")
def second_round_cmd(
    asof: str = typer.Option(..., help="Date de prévision AAAA-MM-JJ (sondages terminés à cette date)"),
    election: int = typer.Option(2027, help="Année de l'élection"),
    horizon: int = typer.Option(None, help="Jours avant le premier tour (défaut : premier tour − asof)"),
    n: int = typer.Option(10000, help="Tirages par paire"),
    seed: int = typer.Option(0, help="Graine"),
    out: str = typer.Option(None, help="Dossier de sortie (défaut runs/<aujourd'hui>-second-round-<élection>)"),
) -> None:
    """Second tour par reports de voix (docs/04-modele.md §4) pour chaque paire testée -> runs/…"""
    from datetime import date
    from pathlib import Path

    from vvs import paths
    from vvs.model.second_round import forecast, write_forecast

    out_dir = Path(out) if out else paths.RUNS / f"{date.today().isoformat()}-second-round-{election}"
    pairs, transfers = forecast(election, date.fromisoformat(asof), horizon=horizon, n=n, seed=seed)
    write_forecast(pairs, transfers, out_dir)
    for r in pairs.sort("p_i_wins", descending=True).iter_rows(named=True):
        typer.echo(
            f"{r['candidate_i']:>20s} – {r['candidate_j']:<20s} {r['median']:5.1f} "
            f"[{r['q05']:4.1f}–{r['q95']:4.1f}]  P={r['p_i_wins']:.2f}  sondages {r['n_polls']}"
            + ("  (inédit)" if r["inedit"] else "")
        )
    typer.echo(f"écrit : {out_dir}")


@app.command("curate")
def curate_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Compteurs détaillés de la fusion"),
    dry_run: bool = typer.Option(False, "--dry-run", help="N'écrit rien dans data/curated"),
) -> None:
    """Fusion des sources de sondages et tables dérivées -> data/curated (voir vvs.curate).

    Code de sortie 1 si une divergence connue n'est pas retrouvée dans le journal.
    """
    from vvs.curate import run

    report = run(write=not dry_run)
    typer.echo(report.summary(verbose=verbose))
    if not all(report.known_divergences.values()):
        raise typer.Exit(code=1)


@app.command("simulate")
def simulate_cmd(
    asof: str = typer.Option(..., help="Date de prévision AAAA-MM-JJ (sondages terminés à cette date)"),
    election: int = typer.Option(2027, help="Année de l'élection"),
    fit: str = typer.Option(None, help="Ajustement du premier tour (défaut runs/<asof>-fit-<élection>, créé s'il manque)"),
    out: str = typer.Option(None, help="Dossier de sortie (défaut runs/<aujourd'hui>-simulate)"),
    draws: int = typer.Option(40_000, help="Tirages Monte-Carlo (10 000 au minimum)"),
    seed: int = typer.Option(0, help="Graine"),
    rn_bias: float = typer.Option(
        None, help="Part du biais RN du second tour appliquée (défaut : atténuation par horizon)"
    ),
    refit: bool = typer.Option(False, "--refit", help="Refaire l'ajustement même s'il existe"),
    substitution: bool = typer.Option(
        False, "--substitution", help="Substitution intra-bloc (logit emboîté) dans l'ajustement"
    ),
) -> None:
    """Simulation de l'élection (docs/04-modele.md § 5) : scénarios, premier tour, second tour."""
    from datetime import date
    from pathlib import Path

    import polars as pl

    from vvs import paths
    from vvs.model.simulate import fit_for_simulation, simulate

    if draws < 10_000:
        raise typer.BadParameter("au moins 10 000 tirages (docs/04-modele.md § 5)")
    d = date.fromisoformat(asof)
    suffix = "-subst" if substitution else ""
    fit_dir = Path(fit) if fit else paths.RUNS / f"{d.isoformat()}-fit-{election}{suffix}"
    if refit or not (fit_dir / "draws.npz").exists():
        typer.echo(f"ajustement du premier tour -> {fit_dir}")
        kw = {}
        if substitution:
            from dataclasses import replace

            from vvs.eval.backtest import DEFAULT_VARIANT, VARIANTS

            kw["priors"] = replace(VARIANTS[DEFAULT_VARIANT], substitution=True)
        m = fit_for_simulation(election, d, fit_dir, **kw)
        typer.echo(f"r_hat max {m['diagnostics'].get('r_hat_max')}, "
                   f"divergences {m['diagnostics'].get('divergences')}")
    out_dir = Path(out) if out else paths.RUNS / f"{date.today().isoformat()}-simulate"
    res = simulate(fit_dir, d, out_dir, election=election, n_draws=draws, seed=seed, rn_bias=rn_bias)
    mix = res.summary.filter(pl.col("condition_id") == "mix")
    for r in mix.filter(pl.col("p_qualify") >= 0.005).iter_rows(named=True):
        typer.echo(
            f"{r['candidate_id']:>24s}  présence {r['p_present']:4.2f}  qualif. {r['p_qualify']:5.3f}"
            f"  victoire {r['p_win']:5.3f}  part {r['q50']:4.1f} [{r['q05']:4.1f}–{r['q95']:4.1f}]"
        )
    typer.echo(f"durée {res.manifest['duration_s']} s ; écrit : {out_dir}")


@app.command("export")
def export_cmd(
    run: str = typer.Option(..., help="Run de simulation (runs/<date>-simulate)"),
    out: str = typer.Option(None, help="Dossier du site (défaut front/static/data)"),
) -> None:
    """Fichiers JSON du site (docs/08-front.md) ; valider ensuite avec `npm run validate-data`."""
    from pathlib import Path

    from vvs import paths
    from vvs.export.export import export_run

    out_dir = Path(out) if out else paths.ROOT / "front" / "static" / "data"
    rep = export_run(Path(run), out_dir)
    typer.echo(
        f"{rep['n_polls']} sondages, {rep['n_candidates']} candidats, commit {rep['commit']}, "
        f"historique {'écrit' if rep['history_written'] else 'inchangé'} "
        f"({rep['history_entries']} entrées)"
    )
    for name, size in sorted(rep["sizes"].items()):
        typer.echo(f"  {name:20s} {size / 1024:7.1f} Ko")
    typer.echo(f"écrit : {out_dir}")


if __name__ == "__main__":
    app()
