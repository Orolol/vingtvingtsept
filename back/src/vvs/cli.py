"""Point d'entrée `uv run vvs …`. Les sous-commandes s'enregistrent ici au fil des phases."""

from __future__ import annotations

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
def parse_commission() -> None:
    """Notices brutes -> polls_commission.parquet et poll_results_commission.parquet."""
    from vvs.ingest.commission.parse import run

    report = run()
    typer.echo(report.summary())


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
    series: list[str] = typer.Option(None, "--series", help="series_id de data/registry/macro_series.yaml (répétable ; toutes par défaut)"),
) -> None:
    """Séries économiques et de popularité -> data/raw/macro/<série>/<date>.<ext>."""
    from vvs.ingest.macro import ingest

    ingest(series or None, log=typer.echo)


@parse_app.command("macro")
def parse_macro(
    series: list[str] = typer.Option(None, "--series", help="series_id à reparser (les autres lignes du parquet sont conservées)"),
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
            for p in report.ifop.publications:
                typer.echo(f"  PDF {p['month']} {p['mode']:7s} {p['sponsor']:12s} tableaux {p['n_table']:4d} titre {p['n_headline']} {p['file']}")
    typer.echo(f"écrit : {paths.CURATED / OUTPUT_NAME} ({df.height} lignes)")

if __name__ == "__main__":
    app()
