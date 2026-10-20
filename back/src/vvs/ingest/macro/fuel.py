"""Prix des carburants : base des prix des produits pétroliers du ministère de la Transition
écologique (DGEC), https://www.ecologie.gouv.fr/politiques-publiques/prix-produits-petroliers.

La base n'a pas d'API : la page embarque un formulaire Drupal (« Suivre l'évolution ») chargé
en AJAX. On rejoue ce que fait le navigateur :

1. la page donne l'URL AJAX du simulateur (`/ajax/simulator/load/…/%23simulator_<id>`) ;
2. un GET de cette URL renvoie le formulaire et son `form_build_id` ;
3. un POST (action « monitor », carburants routiers, un carburant, dates de début et de fin)
   renvoie la page de résultat, qui contient le lien d'export
   `/simulator-energies/monitoring/export/<hash>` ;
4. ce lien sert un xlsx à huit feuilles : hebdo, mensuel, semestriel, annuel, en HTT et TTC.

Une requête par carburant, pour que chaque série ait son fichier brut.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta

import httpx

from vvs.ingest.macro import raw, xlsx
from vvs.ingest.macro.registry import Series

BASE = "https://www.ecologie.gouv.fr"
PAGE = BASE + "/politiques-publiques/prix-produits-petroliers"
FORM_ID = "webform_submission_fuel_simulator_add_form"
# code du formulaire -> début du libellé de colonne dans l'export (« Super sans plomb 95 €/l
# ttc », « Gazole €/l ttc ») ; l'espace final écarte « Super sans plomb 95-E10 ».
FUEL_LABELS = {"sp95": "Super sans plomb 95 ", "gazole": "Gazole "}
AJAX_HEADERS = {"X-Requested-With": "XMLHttpRequest"}


def ingest(
    http: httpx.Client, series: Series, start: date, end: date | None = None
) -> raw.RawFile | None:
    content, url = fetch_export(http, series.source_id, start, end or date.today())
    parse(content, series.source_id, series.params["sheet"])  # valide avant d'écrire
    return raw.write_raw(series.raw_dir, "xlsx", content, url)


def fetch_export(http: httpx.Client, fuel: str, start: date, end: date) -> tuple[bytes, str]:
    if fuel not in FUEL_LABELS:
        raise ValueError(f"carburant inconnu {fuel!r} (connus : {sorted(FUEL_LABELS)})")
    page = raw.get(http, PAGE).text
    m = re.search(
        r'href="(/ajax/simulator/load/customer_simulator_energies_fuel_block/[^"]+)"', page
    )
    if not m:
        raise ValueError("lien AJAX du simulateur introuvable sur la page du ministère")
    ajax_url = BASE + m.group(1)
    params = {"_wrapper_format": "drupal_ajax"}
    form_html = _inserted_html(raw.get(http, ajax_url, params=params, headers=AJAX_HEADERS))
    build_id = re.search(r'name="form_build_id" value="([^"]+)"', form_html)
    if not build_id:
        raise ValueError("form_build_id introuvable dans le formulaire du simulateur")
    data = {
        "action_select": "monitor",
        "fuel_type_select": "road_fuels",
        f"monitor_road_fuels[{fuel}]": fuel,
        "monitor_start_date": start.isoformat(),
        "monitor_end_date": end.isoformat(),
        "form_build_id": build_id.group(1),
        "form_id": FORM_ID,
        "op": "Soumettre",
        "_triggering_element_name": "op",
        "_triggering_element_value": "Soumettre",
        "_drupal_ajax": "1",
    }
    r = http.post(ajax_url, params={**params, "ajax_form": "1"}, data=data, headers=AJAX_HEADERS)
    r.raise_for_status()
    result_html = _inserted_html(r)
    link = re.search(r'href="(/simulator-energies/monitoring/export/[0-9a-f]+)"', result_html)
    if not link:
        raise ValueError("lien d'export absent de la réponse du simulateur (formulaire modifié ?)")
    export_url = BASE + link.group(1)
    export = raw.get(http, export_url)
    if not export.content.startswith(b"PK"):
        raise ValueError(f"l'export {export_url} n'est pas un xlsx")
    return export.content, export_url


def _inserted_html(r: httpx.Response) -> str:
    """Contenu HTML de la commande AJAX Drupal `insert` la plus longue."""
    commands = json.loads(r.text)
    inserts = [
        c["data"]
        for c in commands
        if c.get("command") == "insert" and isinstance(c.get("data"), str)
    ]
    if not inserts:
        raise ValueError("réponse AJAX sans commande insert")
    return max(inserts, key=len)


def parse(content: bytes, fuel: str, sheet: str) -> list[tuple[date, float]]:
    """[(lundi de la semaine, prix €/l)] ; la date source est le vendredi du relevé."""
    rows = xlsx.read_sheet(content, sheet)
    if not rows:
        raise ValueError(f"feuille {sheet!r} vide")
    header = [str(h or "") for h in rows[0]]
    if not header or header[0].strip().lower() != "date":
        raise ValueError(f"en-tête inattendu {header}")
    label = FUEL_LABELS[fuel]
    cols = [i for i, h in enumerate(header) if h.startswith(label)]
    if len(cols) != 1:
        raise ValueError(f"colonne {label!r} introuvable ou ambiguë dans {header}")
    col = cols[0]
    out = {}
    for row in rows[1:]:
        if not row or row[0] is None:
            continue
        friday = datetime.strptime(str(row[0]).strip(), "%d/%m/%Y").date()
        value = row[col] if col < len(row) else None
        if value is None or value == "":
            continue
        week = friday - timedelta(days=friday.weekday())
        if week in out:
            raise ValueError(f"deux relevés dans la semaine du {week}")
        out[week] = float(str(value).replace(",", "."))
    return sorted(out.items())
