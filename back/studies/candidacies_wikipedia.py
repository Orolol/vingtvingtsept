"""Collecte des candidatures historiques dans les révisions datées de Wikipédia FR.

Lot 1 de docs/12 : liste de collecte du registre `data/registry/candidacies_<année>.yaml`,
définie par des sources et des périodes fixées d'avance, jamais par la liste des candidats
finalement présents.

Sources (`PAGES`) : pour chaque élection, la page « Élection présidentielle française de
AAAA » (sections dont le titre contient « candidat » ou « primaire »), la page
« Candidatures à l'élection présidentielle française de AAAA » (page entière hors notes,
sondages et articles connexes) et la page anglaise « AAAA French presidential election »
(sections « candidates » et « primary »), dans la révision en vigueur à 00:00 UTC de chaque
date de collecte. Les deux éditions ont des règles d'inclusion différentes (la française
de 2021 ne liste pas les personnalités non déclarées, par exemple) : leurs écarts sont
rapportés par l'audit. Les révisions sont celles que tout lecteur voyait ce jour-là : leur horodatage
borne par excès la date à laquelle l'information était publique.

Périodes (`PERIODS`) : du 1er janvier de l'année précédant le scrutin (primaires de
l'automne comprises) au premier tour. Cadence : le 1er et le 15 de chaque mois, plus les
douze dates du backtest et la date de publication de la liste officielle.

Extraction, volontairement large : tous les liens internes des sections retenues, plus le
nom en tête des puces et des légendes de galerie sans lien. Les liens sont filtrés par
Wikidata (instance de « être humain », Q5) et regroupés par identifiant Wikidata. Le tri
final (personne envisagée comme candidate, ou soutien, organisateur, parrain cité) reste
manuel et se fait dans le registre. La première trace d'une personne est datée par la
révision du premier instantané qui la contient : précision d'environ quinze jours, exacte
aux douze dates du backtest, qui sont des dates d'instantané.

    uv run python studies/candidacies_wikipedia.py --out ../runs/2026-10-23-candidacies
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import time
from pathlib import Path

import httpx

API = {"fr": "https://fr.wikipedia.org/w/api.php", "en": "https://en.wikipedia.org/w/api.php"}
WIKIDATA = "https://www.wikidata.org/w/api.php"
USER_AGENT = "vingtvingtsept/0.1 (French 2027 presidential forecast; https://github.com/Orolol)"

# clé de page -> (langue, nature de la page, titre)
PAGES = {
    e: {
        "election": ("fr", "election", f"Élection présidentielle française de {e}"),
        "candidacies": ("fr", "candidacies", f"Candidatures à l'élection présidentielle française de {e}"),
        "en_election": ("en", "election", f"{e} French presidential election"),
    }
    for e in (2012, 2017, 2022)
}
PERIODS = {
    2012: (dt.date(2011, 1, 1), dt.date(2012, 4, 22)),
    2017: (dt.date(2016, 1, 1), dt.date(2017, 4, 23)),
    2022: (dt.date(2021, 1, 1), dt.date(2022, 4, 10)),
}
AUDIT_DATES = {
    2012: ["2011-10-25", "2012-01-23", "2012-03-23", "2012-04-15"],
    2017: ["2016-10-25", "2017-01-23", "2017-03-24", "2017-04-16"],
    2022: ["2021-10-12", "2022-01-10", "2022-03-11", "2022-04-03"],
}
LIST_DATES = {2012: "2012-03-19", 2017: "2017-03-18", 2022: "2022-03-07"}

SECTION_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
LINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]*)?(?:\|([^\]]*))?\]\]")
DROP_SECTIONS = re.compile(r"notes|références|sondage|articles connexes|liens externes|bibliographie"
                           r"|résultat|voir aussi|polls|polling|references|external links|results",
                           re.IGNORECASE)
KEEP_ELECTION = re.compile(r"candidat|primaire|primary|primaries", re.IGNORECASE)
NAME_RE = re.compile(r"^[A-ZÉÈÀÂÎÔÛÇ][\w'’\-\.]+(?:\s+(?:de|du|des|d'|le|la|van|von|[A-ZÉÈÀÂÎÔÛÇ]"
                     r"[\w'’\-\.]*)){1,4}$")


def dates_for(election: int) -> list[dt.date]:
    start, end = PERIODS[election]
    out = set()
    d = start
    while d <= end:
        out.add(d)
        out.add(d.replace(day=15))
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    out.update(dt.date.fromisoformat(x) for x in AUDIT_DATES[election])
    out.add(dt.date.fromisoformat(LIST_DATES[election]))
    return sorted(x for x in out if start <= x <= end)


class Wiki:
    def __init__(self, cache: Path):
        self.client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60)
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, **params) -> dict:
        params["format"] = "json"
        for attempt in range(5):
            r = self.client.get(url, params=params)
            if r.status_code == 200:
                time.sleep(0.2)
                return r.json()
            time.sleep(2 * (attempt + 1))
        r.raise_for_status()
        raise RuntimeError("inaccessible")

    def revision_at(self, lang: str, title: str, when: dt.date) -> dict | None:
        """Révision en vigueur à 00:00 UTC de `when` (None si la page n'existait pas)."""
        q = self.get(API[lang], action="query", titles=title, redirects=1, prop="revisions",
                     rvlimit=1, rvdir="older", rvstart=f"{when}T00:00:00Z",
                     rvprop="ids|timestamp")
        page = next(iter(q["query"]["pages"].values()))
        revs = page.get("revisions")
        if not revs:
            return None
        return {"revid": revs[0]["revid"], "timestamp": revs[0]["timestamp"],
                "title": page["title"]}

    def content(self, lang: str, revid: int) -> str:
        path = self.cache / f"{lang}_{revid}.wiki"
        if path.exists():
            return path.read_text()
        q = self.get(API[lang], action="query", revids=revid, prop="revisions",
                     rvprop="content", rvslots="main")
        page = next(iter(q["query"]["pages"].values()))
        text = page["revisions"][0]["slots"]["main"]["*"]
        path.write_text(text)
        return text


def sections(text: str, kind: str) -> list[tuple[str, str]]:
    """(chemin de titres, texte) des sections retenues."""
    heads = list(SECTION_RE.finditer(text))
    out, path = [], {}
    intro = text[: heads[0].start()] if heads else text
    if kind == "candidacies":
        out.append(("(intro)", intro))
    keep_level = None
    for i, h in enumerate(heads):
        level, title = len(h.group(1)), h.group(2)
        path = {k: v for k, v in path.items() if k < level}
        path[level] = title
        body = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        if keep_level is not None and level <= keep_level:
            keep_level = None
        if level == 2 and DROP_SECTIONS.search(title):
            keep_level = None
            continue
        if kind == "election" and keep_level is None:
            if level == 2 and KEEP_ELECTION.search(title):
                keep_level = 2
            else:
                continue
        if kind == "candidacies" and any(DROP_SECTIONS.search(t) for t in path.values()):
            continue
        out.append((" > ".join(path[k] for k in sorted(path)), body))
    return out


def clean(s: str) -> str:
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.DOTALL)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.DOTALL)
    return s


def mentions(text: str, kind: str) -> list[dict]:
    rows = []
    for path, body in sections(text, kind):
        body = clean(body)
        for m in LINK_RE.finditer(body):
            target = m.group(1).strip()
            if ":" in target:
                continue
            rows.append({"section": path, "name": target, "linked": True})
        for line in body.splitlines():
            line = line.strip()
            lead = None
            if line.startswith("*"):
                lead = line.lstrip("*: ").strip()
            elif "<center>" in line:
                lead = line.split("<center>", 1)[1]
            if not lead or lead.startswith("[["):
                continue
            lead = re.sub(r"'{2,}", "", lead)
            name = re.split(r"[,(;:«]| est | a | s'| se | et | déjà ", lead, maxsplit=1)[0].strip()
            if NAME_RE.match(name):
                rows.append({"section": path, "name": name, "linked": False})
    return rows


def humans(wiki: Wiki, links: list[tuple[str, str]]) -> dict[str, dict]:
    """(langue, titre de lien) -> {"title", "qid", "human"}, clé "langue:titre"."""
    out = {}
    for lang in ("fr", "en"):
        titles = sorted({t for lg, t in links if lg == lang})
        for i in range(0, len(titles), 50):
            batch = titles[i: i + 50]
            q = wiki.get(API[lang], action="query", titles="|".join(batch), redirects=1,
                         prop="pageprops", ppprop="wikibase_item")["query"]
            norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
            redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
            pages = {p["title"]: p for p in q["pages"].values()}
            for t in batch:
                resolved = redir.get(norm.get(t, t), norm.get(t, t))
                p = pages.get(resolved, {})
                out[f"{lang}:{t}"] = {"title": resolved,
                                      "qid": p.get("pageprops", {}).get("wikibase_item")}
    qids = sorted({v["qid"] for v in out.values() if v["qid"]})
    is_human, labels = {}, {}
    for i in range(0, len(qids), 50):
        q = wiki.get(WIKIDATA, action="wbgetentities", ids="|".join(qids[i: i + 50]),
                     props="claims|labels", languages="fr|en")
        for qid, ent in q["entities"].items():
            p31 = ent.get("claims", {}).get("P31", [])
            is_human[qid] = any(
                c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") == "Q5" for c in p31
            )
            lab = ent.get("labels", {})
            labels[qid] = (lab.get("fr") or lab.get("en") or {}).get("value")
    for v in out.values():
        v["human"] = bool(v["qid"] and is_human.get(v["qid"]))
        v["label"] = labels.get(v["qid"]) or v["title"]
    return out


def snapshot(wiki: Wiki, election: int) -> tuple[list[dict], list[dict]]:
    manifest, rows = [], []
    for when in dates_for(election):
        for key, (lang, kind, title) in PAGES[election].items():
            rev = wiki.revision_at(lang, title, when)
            if rev is None:
                continue
            text = wiki.content(lang, rev["revid"])
            manifest.append({"election": election, "date": str(when), "page": key,
                             "lang": lang, "kind": kind, "title": rev["title"],
                             "revid": rev["revid"], "timestamp": rev["timestamp"],
                             "sha256": hashlib.sha256(text.encode()).hexdigest()})
            for m in mentions(text, kind):
                rows.append({"election": election, "date": str(when), "page": key, "lang": lang,
                             "revid": rev["revid"], "timestamp": rev["timestamp"], **m})
    return manifest, rows


def summarise(rows: list[dict], who: dict[str, dict]) -> dict[str, dict]:
    """Personnes : clé QID Wikidata (liens) ou nom (sans lien), première et dernière trace."""
    persons: dict[str, dict] = {}
    for r in rows:
        if r["linked"]:
            info = who[f"{r['lang']}:{r['name']}"]
            if not info["human"]:
                continue
            key, label = info["qid"], info["label"]
        else:
            key = label = r["name"]
        p = persons.setdefault(key, {"name": label, "qid": key if r["linked"] else "",
                                     "dates": set(), "pages": set(), "sections": set(),
                                     "first": None, "last": None})
        p["dates"].add(r["date"])
        p["pages"].add(r["page"])
        p["sections"].add(f"{r['page']}: {r['section']}")
        first = (r["date"], r["timestamp"], r["revid"], r["page"])
        if p["first"] is None or first < p["first"]:
            p["first"] = first
        if p["last"] is None or r["date"] > p["last"]:
            p["last"] = r["date"]
    return persons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--elections", type=int, nargs="*", default=[2012, 2017, 2022])
    args = ap.parse_args()
    base = args.out / "wikipedia"
    wiki = Wiki(base / "revisions")
    for election in args.elections:
        manifest, rows = snapshot(wiki, election)
        for name, table in (("manifest", manifest), ("mentions", rows)):
            with (base / f"{name}_{election}.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(table[0]))
                w.writeheader()
                w.writerows(table)
        who = humans(wiki, sorted({(r["lang"], r["name"]) for r in rows if r["linked"]}))
        persons = summarise(rows, who)
        audit = AUDIT_DATES[election]
        with (base / f"persons_{election}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "name", "first_snapshot", "first_page", "first_revid",
                        "first_timestamp", "last_snapshot", "n_snapshots", "pages",
                        *[f"at_{d}" for d in audit], "sections"])
            for key in sorted(persons, key=lambda k: (persons[k]["first"][0], persons[k]["name"])):
                p = persons[key]
                w.writerow([key, p["name"], p["first"][0], p["first"][3], p["first"][2],
                            p["first"][1], p["last"], len(p["dates"]), "|".join(sorted(p["pages"])),
                            *[d in p["dates"] for d in audit], " | ".join(sorted(p["sections"]))])
        (base / f"humans_{election}.json").write_text(json.dumps(who, ensure_ascii=False,
                                                                  indent=1))
        print(election, len(manifest), "instantanés,", len(persons), "personnes")


REF_RE = re.compile(r"<ref[^>/]*>(.*?)</ref>", re.DOTALL)
FIELD_RE = re.compile(r"\|\s*(url|titre|title|date|site|périodique|éditeur|lire en ligne)\s*=\s*([^|}]*)")


def refs_of(raw_line: str) -> list[dict]:
    """Références {{lien web}} / {{article}} d'une ligne brute : url, titre, date, média."""
    out = []
    for body in REF_RE.findall(raw_line):
        fields = {k: v.strip() for k, v in FIELD_RE.findall(body)}
        url = fields.get("url") or fields.get("lire en ligne")
        if not url:
            m = re.search(r"https?://[^\s\]|}]+", body)
            url = m.group(0) if m else None
        if url:
            out.append({"url": url, "title": fields.get("titre") or fields.get("title"),
                        "date": fields.get("date"),
                        "outlet": fields.get("site") or fields.get("périodique")
                        or fields.get("éditeur")})
    return out


def context(base: Path, election: int, names: list[str], max_lines: int = 6) -> dict:
    """Lignes (nettoyées) qui citent chaque nom, dans l'ordre de leur première apparition, avec
    la révision et les références de la ligne brute (aide au tri manuel et au choix des
    sources contemporaines)."""
    manifest = list(csv.DictReader((base / f"manifest_{election}.csv").open()))
    who = json.loads((base / f"humans_{election}.json").read_text())
    needles = {n: {n} | {k.split(":", 1)[1] for k, v in who.items() if v["label"] == n}
               for n in names}
    out: dict[str, list] = {n: [] for n in names}
    seen: dict[str, set] = {n: set() for n in names}
    for m in manifest:
        text = (base / "revisions" / f"{m['lang']}_{m['revid']}.wiki").read_text()
        for path, body in sections(text, m["kind"]):
            for raw in body.splitlines():
                line = clean(raw).strip()
                if "<center>" in line or line.lower().startswith(("fichier:", "file:")):
                    continue
                for n in names:
                    if len(out[n]) >= max_lines or not any(x in line for x in needles[n]):
                        continue
                    key = re.sub(r"\W+", "", line)[:200]
                    if key in seen[n]:
                        continue
                    seen[n].add(key)
                    out[n].append({"date": m["date"], "revid": m["revid"], "lang": m["lang"],
                                   "timestamp": m["timestamp"], "page": m["page"],
                                   "section": path, "line": line[:700], "refs": refs_of(raw)})
    return out


def digest(base: Path, election: int, name: str, width: int = 220) -> str:
    """Chronologie d'une personne dans les instantanés : à chaque changement des sections où
    elle apparaît, la révision (page, oldid, horodatage), les sections et le début des lignes ;
    puis les références de presse de ces lignes (date, média, titre, url). Outil de tri."""
    manifest = list(csv.DictReader((base / f"manifest_{election}.csv").open()))
    who = json.loads((base / f"humans_{election}.json").read_text())
    needles = {name} | {k.split(":", 1)[1] for k, v in who.items() if v["label"] == name}
    out, refs, last = [], {}, {}
    for m in manifest:
        text = (base / "revisions" / f"{m['lang']}_{m['revid']}.wiki").read_text()
        found = []
        for path, body in sections(text, m["kind"]):
            for raw in body.splitlines():
                line = clean(raw).strip()
                if "<center>" in line or line.lower().startswith(("fichier:", "file:")):
                    continue
                if any(x in line for x in needles):
                    found.append((path, line))
                    for r in refs_of(raw):
                        refs.setdefault(r["url"], {**r, "seen": m["date"]})
        sig = tuple(sorted({p for p, _ in found}))
        if sig != last.get(m["page"]):
            last[m["page"]] = sig
            out.append(f"{m['date']} {m['page']} oldid={m['revid']} ({m['timestamp']}) :"
                       + (" ABSENT" if not found else ""))
            for path, line in found[:4]:
                out.append(f"    [{path[:70]}] {line[:width]}")
    out.append("références :")
    for r in sorted(refs.values(), key=lambda r: r["seen"]):
        out.append(f"    {r['date']} | {r['outlet']} | {(r['title'] or '')[:110]} | {r['url']}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "digest":
        # uv run python studies/candidacies_wikipedia.py digest RUN_DIR ELECTION "Nom" …
        for n in sys.argv[4:]:
            print(f"===== {n}")
            print(digest(Path(sys.argv[2]) / "wikipedia", int(sys.argv[3]), n))
    else:
        main()
