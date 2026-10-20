"""Point d'entrée `uv run vvs …`. Les sous-commandes s'enregistrent ici au fil des phases."""

from __future__ import annotations

import typer

app = typer.Typer(help="vingtvingtsept — prévision de la présidentielle 2027", no_args_is_help=True)
ingest_app = typer.Typer(help="Téléchargement des sources vers data/raw", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")


@app.command()
def version() -> None:
    from vvs import __doc__ as doc

    typer.echo(doc)


if __name__ == "__main__":
    app()
