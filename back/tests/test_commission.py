"""Notices de la Commission des sondages : listage, lecture des PDF, tables curated.

Trois notices figées dans tests/fixtures/commission/ couvrent les trois formats de tableaux
les plus différents : Elabe (publié seul), Cluster17 (« Brut / Redressé »), OpinionWay
(neuf colonnes, publié en tête). Les valeurs attendues ont été relues sur les PDF.
"""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from vvs import schemas
from vvs.ingest.commission import listing, notice
from vvs.ingest.commission import parse as cparse
from vvs.ingest.commission.aliases import candidate_id, pollster_id

FIXTURES = Path(__file__).parent / "fixtures" / "commission"


# ------------------------------------------------------------------------------ listage


@pytest.mark.parametrize(
    "title, month, expected",
    [
        (
            "10325 Pres Focus droite et centre CLUSTER17 Le Point 18 octobre",
            "Octobre 2026",
            {
                "number": "10325",
                "institute": "CLUSTER17",
                "sponsor": "Le Point",
                "date": "2026-10-18",
            },
        ),
        (
            "10246A Pres Indices de popularite IFOP LOpinion 31 août",
            "Août 2026",
            {"number": "10246A", "institute": "IFOP", "sponsor": "LOpinion", "date": "2026-08-31"},
        ),
        (
            "10286 Pres Budget 2027 ELABE 1er octobre",
            "Octobre 2026",
            {"number": "10286", "institute": "ELABE", "sponsor": None, "date": "2026-10-01"},
        ),
        # date de décembre déposée en janvier : année précédente
        (
            "10059 Pres bloc central droite ELABE BFMTV 28 decembre",
            "Janvier 2026",
            {"number": "10059", "institute": "ELABE", "sponsor": "BFMTV", "date": "2025-12-28"},
        ),
    ],
)
def test_parse_title(title, month, expected):
    assert listing.parse_title(title, month) == expected


def test_listing_union_of_tag_and_search():
    tagged = listing.parse_listing((FIXTURES / "listing_tag.html").read_text(), tagged=True)
    searched = listing.parse_listing((FIXTURES / "listing_search.html").read_text(), tagged=False)
    merged = {e.file_id: e for e in listing.merge_listings(tagged, searched)}
    # « Pres » hors thème gardé ; municipales (« Centre presse ») et titre sans jeton écartés
    assert 2302 in merged and not merged[2302].tagged_2027
    assert 2120 not in merged and 2191 not in merged
    assert merged[2196].institute == "THI" and merged[2026].date == "2025-12-12"
    assert merged[2339].month == "Octobre 2026"


@pytest.mark.parametrize(
    "label, expected",
    [
        ("TOLUNA HARRIS INTERACTIVE", "harris"),
        ("Pres Barometre confiance politique THI 29 mai", "harris"),
        ("Pres CommStrat L'Opinion Toluna HI 14 avril", "harris"),
        ("IPSOS BVA La Tribune", "ipsos"),
        ("Cluster 17 Le Point", "cluster17"),
        ("Mun Paris OPINION WAY JDD", "opinionway"),
    ],
)
def test_pollster_aliases(label, expected):
    assert pollster_id(label) == expected


class FakeClient:
    """Client hors réseau : listes figées, une notice = quelques octets PDF."""

    def __init__(self):
        self.downloads = 0

    def get(self, url):
        if url == listing.TAG_URL:
            return _Resp((FIXTURES / "listing_tag.html").read_text(), url)
        self.downloads += 1
        file_id = url.rsplit("/", 1)[1]
        return _Resp(b"%PDF-1.7 fake", f"http://x/notices/files/notices/2026/{file_id}.pdf")

    def post(self, url, data):
        return _Resp((FIXTURES / "listing_search.html").read_text(), url)


class _Resp:
    def __init__(self, body, url):
        import httpx

        self.text = body if isinstance(body, str) else ""
        self.content = body if isinstance(body, bytes) else body.encode()
        self.url = httpx.URL(url)
        self.headers = {"content-type": "application/pdf"}


def test_ingest_is_idempotent(tmp_path):
    client = FakeClient()
    first = listing.ingest(date(2026, 5, 1), client=client, raw_dir=tmp_path)
    assert first["kept"] == 4 and first["downloaded"] == 4  # 2341, 2339, 2302, 2196
    second = listing.ingest(date(2026, 5, 1), client=client, raw_dir=tmp_path)
    assert second["downloaded"] == 0 and second["skipped"] == 4
    rows = (tmp_path / "index.jsonl").read_text().splitlines()
    assert len(rows) == 4 and sorted(p.name for p in tmp_path.glob("*.pdf")) == [
        "2196.pdf",
        "2302.pdf",
        "2339.pdf",
        "2341.pdf",
    ]


# ------------------------------------------------------------------- lecture des lignes


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Marine LE PEN", "le_pen_marine"),
        ("EricZemmour", "zemmour_eric"),
        ("Dominique De Villepin", "de_villepin_dominique"),
        ("Marine Le Pen - Rassemblement National", "le_pen_marine"),
        ("Raphaël Glucksman", "glucksmann_raphael"),
        # deux tableaux côte à côte collés sur une ligne : pas un candidat
        ("Édouard Philippe - Horizons 42% 48% Marine Le Pen - Rassemblement National", None),
        ("Le candidat du RN, Marine Le Pen ou Jordan Bardella", None),
    ],
)
def test_candidate_id(label, expected):
    assert candidate_id(label) == expected


@pytest.mark.parametrize(
    "line, label, values",
    [
        ("Jean-Luc Mélenchon 15% +/- 2,2 points 13%", "Jean-Luc Mélenchon", [15.0, 13.0]),
        ("Karim Bouamrane <1% - - -", "Karim Bouamrane", [0.5, None, None, None]),
        ("• Nathalie Arthaud ................... 0,8 0,6", "Nathalie Arthaud", [0.8, 0.6]),
        ("Nicolas Dupont-Aignan 1 Non présent", "Nicolas Dupont-Aignan", [1.0, None]),
        ("Marine Le Pen 34,5% +/- 2,9 points 34%*", "Marine Le Pen", [34.5, 34.0]),
    ],
)
def test_parse_row(line, label, values):
    row = notice.parse_row(line, 0)
    assert (row.label, row.values) == (label, values)


def test_merge_wrapped_values():
    lines = [
        "Jean-Luc MÉLENCHON",
        "12,5",  # valeur à la ligne
        "Raphaël",
        "10,0",
        "Glucksmann",  # valeur au milieu du nom
        "Nicolas Dupont-",
        "Aignan 2",  # nom coupé
        "4% +/- 0,4 point",
        "Eric Zemmour",  # valeur avant le nom, en fin de tableau
        "N’expriment pas d’intention de",
        "17",
        "vote",
    ]
    merged = notice.merge_wrapped_values(lines)
    assert merged[:4] == [
        "Jean-Luc MÉLENCHON 12,5",
        "Raphaël Glucksmann 10,0",
        "Nicolas Dupont-Aignan 2",
        "Eric Zemmour 4% +/- 0,4 point",
    ]
    assert merged[4:] == lines[-3:]


def test_split_two_column_page():
    lines = [
        "Hypothèse 1 Hypothèse 2",
        "Bruts Pondérés Bruts Pondérés",
        "Édouard Philippe - Horizons 42% 48% Marine Le Pen - Rassemblement National 60% 55%",
        "Marine Le Pen - Rassemblement National 58% 52% Raphaël Glucksmann - Place Publique 40% 45%",
    ]
    page = "\n".join(lines)
    left, right = notice.split_two_column_page(page)
    assert left.splitlines()[1:] == [
        "Bruts Pondérés",
        "Édouard Philippe - Horizons 42% 48%",
        "Marine Le Pen - Rassemblement National 58% 52%",
    ]
    assert right.splitlines()[-1] == "Raphaël Glucksmann - Place Publique 40% 45%"


def _iv_page(rows: list[str], header: str = "Si le 1er tour avait lieu dimanche prochain") -> str:
    return "\n".join(["QUESTIONS PUBLIEES", header, *rows, "Fin de page"])


_FIRST_ROUND = [
    "Nathalie Arthaud 1",
    "Jean-Luc Mélenchon 15",
    "Raphaël Glucksmann 15",
    "Edouard Philippe 17",
    "Bruno Retailleau 15",
    "Marine Le Pen 33",
    "Eric Zemmour 4",
]


def test_unknown_label_in_vote_table_fails():
    rows = [*_FIRST_ROUND[:-1], "Jean Inconnu 4"]
    with pytest.raises(notice.UnknownLabelError, match="Jean Inconnu"):
        notice.classify_blocks(notice.find_blocks([_iv_page(rows)]), "elabe", "registered")


def test_recall_and_short_blocks_are_skipped():
    recall = _iv_page(
        [
            "Emmanuel Macron 27,8",
            "Marine Le Pen 23,2",
            "Jean-Luc Mélenchon 22,0",
            "Eric Zemmour 7,1",
            "Valérie Pécresse 4,8",
            "Yannick Jadot 4,6",
            "Autres 10,5",
        ],
        header="Vote au 1er tour de l'élection présidentielle 2022",
    )
    chart = _iv_page(["Nathalie Arthaud 36", "Jean-Luc Mélenchon 32", "Fabien Roussel 33"])
    res = notice.classify_blocks(notice.find_blocks([recall, chart]), "harris", "registered")
    assert res.hypotheses == []
    assert [o.status for o in res.outcomes] == ["recall_or_other_question", "too_few_candidates"]


def test_brut_header_sets_columns():
    page = _iv_page(
        [
            f"{r.rsplit(' ', 1)[0]} {int(r.rsplit(' ', 1)[1]) + 1}% {r.rsplit(' ', 1)[1]}%"
            for r in _FIRST_ROUND
        ],
        header="ITV 1er tour Hypothèse 1 Brut Redressé",
    )
    [hyp] = notice.classify_blocks(notice.find_blocks([page]), "cluster17", "registered").hypotheses
    assert hyp.results[0] == ("arthaud_nathalie", 1.0, 2.0)


# ------------------------------------------------------------------------- métadonnées


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Dates de réalisation de l'enquête : Du 14 au 16 octobre 2026",
            (date(2026, 10, 14), date(2026, 10, 16)),
        ),
        (
            "Les interviews ont été réalisées\nL'échantillon a été interrogé en ligne.\ndu10au11juin2026.",
            (date(2026, 6, 10), date(2026, 6, 11)),
        ),
        (
            "Enquête réalisée en ligne du 7 (après l'annonce du verdict) au 8 juillet 2026.",
            (date(2026, 7, 7), date(2026, 7, 8)),
        ),
        # mise en page en colonnes (Toluna Harris Interactive) : la plage de la vague en cours est
        # coupée, celle des rappels est complète mais vient après « rappels »
        (
            (
                "Enquête réalisée en ligne du 25 au 27 Échantillon de 1 744 personnes inscrites\n"
                "mai 2026. sur les listes électorales\n"
                "• Les rappels sont issus de la première vague réalisée en ligne du 28 au 30 avril"
                "\n2026,"
            ),
            (date(2026, 5, 25), date(2026, 5, 27)),
        ),
    ],
)
def test_find_fieldwork(text, expected):
    assert notice.find_fieldwork(text) == expected


def test_find_sample_size_ignores_reading_examples():
    text = (
        "L'enquête a été menée auprès d'un échantillon de 1 114 personnes inscrites, extraites "
        "d'un échantillon de 1 200 personnes. Exemple de lecture du tableau : dans le cas d'un "
        "échantillon de 2 000 personnes, si le pourcentage mesuré est de 10 %…"
    )
    assert notice.find_sample_size(text) == 1200


# ------------------------------------------------------------------ notices figées (PDF)


@pytest.fixture(scope="module")
def parsed():
    index = listing.load_index(FIXTURES / "index.jsonl")
    out = {}
    for file_id, entry in index.items():
        out[file_id] = cparse.parse_notice(FIXTURES / entry["file"], entry)
    return out


def _hyp(res, round_, must_have):
    return next(
        h for h in res.hypotheses if h.round == round_ and must_have <= {c for c, _, _ in h.results}
    )


def test_elabe_published_only(parsed):
    res = parsed[2339]
    m = res.meta
    assert (m.pollster, m.sponsor, m.sample_size, m.mode) == (
        "elabe",
        "BFMTV, La Tribune Dimanche",
        1701,
        "online",
    )
    assert (m.fieldwork_start, m.fieldwork_end, m.publish_date) == (
        date(2026, 10, 14),
        date(2026, 10, 16),
        date(2026, 10, 17),
    )
    assert [h.round for h in res.hypotheses] == [1] * 6 + [2] * 3
    h = _hyp(res, 1, {"polony_natacha"})
    assert {c: s for c, s, _ in h.results}["le_pen_marine"] == 33.5
    assert all(r is None for h in res.hypotheses for *_, r in h.results)


def test_cluster17_brut_and_redresse(parsed):
    res = parsed[2256]
    assert res.meta.fieldwork_end == date(2026, 7, 24) and res.meta.sample_size == 1506
    assert [h.round for h in res.hypotheses] == [1, 1, 2, 2, 2, 2, 2]
    h = _hyp(res, 1, {"philippe_edouard"})
    results = {c: (s, r) for c, s, r in h.results}
    assert results["le_pen_marine"] == (30.1, 26.0)
    assert results["philippe_edouard"] == (18.1, 16.9)
    duel = _hyp(res, 2, {"retailleau_bruno"})
    assert {c: (s, r) for c, s, r in duel.results}["le_pen_marine"] == (57.1, 56.1)


def test_opinionway_nine_columns(parsed):
    res = parsed[2334]
    assert res.meta.fieldwork_end == date(2026, 10, 14)
    [first] = [h for h in res.hypotheses if h.round == 1]
    results = {c: (s, r) for c, s, r in first.results}
    # « Publié » en première colonne ; brut des certains d'aller voter en sixième
    assert results["melenchon_jean_luc"] == (15.0, 12.0)
    assert results["bouamrane_karim"] == (0.5, None)
    assert first.population == "certain_voters"
    assert len([h for h in res.hypotheses if h.round == 2]) == 3


def test_build_tables_conform_and_merge(parsed):
    notices = list(parsed.values())
    # une notice complémentaire du même sondage (même institut, terrain, commanditaire)
    twin = cparse.NoticeParse(
        file_id=9999,
        entry={**parsed[2256].entry, "file_id": 9999},
        is_iv=True,
        meta=parsed[2256].meta,
        hypotheses=parsed[2256].hypotheses[:1],
    )
    polls, results = cparse.build_tables([*notices, twin])
    assert polls["poll_id"].to_list() == [
        "2027-cluster17-20260724-1",
        "2027-opinionway-20261014-1",
        "2027-elabe-20261016-1",
    ]
    assert (polls["source"] == "commission").all()
    assert results.columns == list(schemas.POLL_RESULTS_SCHEMA.keys())
    keys = ["poll_id", "round", "hypothesis_id", "candidate_id"]
    assert results.select(keys).unique().height == results.height
    sums = results.group_by(["poll_id", "round", "hypothesis_id"]).agg(pl.col("share").sum())
    assert sums["share"].is_between(97, 103).all()


# ---------------------------------------------------------------------------- réseau


@pytest.mark.slow
def test_live_listing_and_redirect():
    client = listing.Client()
    entries = listing.fetch_listing(client)
    ids = {e.file_id for e in entries}
    assert len(entries) > 150 and {2256, 2339} <= ids
    resp = client.get(listing.NOTICE_URL.format(file_id=2339))
    assert resp.content[:4] == b"%PDF" and str(resp.url).endswith(".pdf")
    json.dumps([e.__dict__ for e in entries[:3]])  # sérialisable tel quel dans l'index
