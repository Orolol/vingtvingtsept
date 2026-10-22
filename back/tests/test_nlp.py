"""Classification NLP des titres (vvs.nlp) : prompts figés, contrainte et validation des
sorties, client OpenAI-compatible simulé, cache sans reclassification, variables hebdo."""

import json
from datetime import date

import httpx
import polars as pl
import pytest
import yaml

from vvs import paths
from vvs.nlp import features as F
from vvs.nlp import labels as L
from vvs.nlp.client import ChatClient, Item, make_groups
from vvs.nlp.prompts import LabelError, PromptError, load_prompts

MODEL = "test-model@0"


@pytest.fixture(scope="module")
def prompts():
    return load_prompts()


def test_registry_digests_frozen(prompts, tmp_path):
    assert {"title-v1", "title-v3", "title-v3-alt"} <= set(prompts.prompts)
    raw = yaml.safe_load((paths.REGISTRY / "nlp_prompts.yaml").read_text(encoding="utf-8"))
    raw["prompts"]["title-v3"]["system"] += " Sois bref."
    p = tmp_path / "nlp_prompts.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(PromptError, match="title-v3"):
        load_prompts(p)


def test_lines_grammar_and_parser(prompts):
    v3, alt = prompts.get("title-v3"), prompts.get("title-v3-alt")
    g = v3.grammar(2)
    assert g.startswith('root ::= "1|" lab "\\n" "2|" lab "\\n"')
    assert 'lab ::= tone "|" stance "|" themes' in g
    assert 't9 ::= "scandals"' in g  # dernier thème : rien après
    assert v3.parse_lines("1|-1|scandal|institutions,scandals\n2|0|other|none\n", 2) == [
        {"tone": -1, "stance": "scandal", "themes": ["institutions", "scandals"]},
        {"tone": 0, "stance": "other", "themes": []}]
    assert alt.parse_lines("1|security|attack|1", 1) == [
        {"tone": 1, "stance": "attack", "themes": ["security"]}]
    for bad, k in [("1|0|other|none", 2), ("2|0|other|none", 1), ("1|0|rumeur|none", 1),
                   ("1|0|other|scandals,scandals", 1), ("1|2|other|none", 1),
                   ("1|0|other|météo", 1)]:
        with pytest.raises(LabelError):
            v3.parse_lines(bad, k)


def test_json_format_schema_and_parser(prompts):
    v1 = prompts.get("title-v1")
    s = v1.schema()
    assert list(s["properties"]) == ["tone", "stance", "themes"]
    assert s["properties"]["tone"]["enum"] == [-1, 0, 1]
    assert v1.parse('{"tone": 1, "stance": "poll", "themes": []}')["stance"] == "poll"
    with pytest.raises(LabelError):
        v1.parse('{"tone": true, "stance": "poll", "themes": []}')
    m = v1.messages("Valérie Pécresse", "Pécresse au Zénith", "2022-02-13", "")
    assert "Pécresse au Zénith" in m[1]["content"] and "inconnue" in m[1]["content"]


def items(n, cid="pecresse_valerie"):
    return [Item(f"s{i}", cid, "Valérie Pécresse", f"Titre {i}", "2022-02-13", "lci.fr")
            for i in range(n)]


def fake_server(answer, calls):
    """Réponse fabriquée à partir du corps de requête (nombre de titres = lignes numérotées)."""
    def h(request):
        body = json.loads(request.content)
        calls.append(body)
        k = body["messages"][1]["content"].count("\n") if "grammar" in body else 1
        text = answer(body, k)
        if text is None:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                         "usage": {"prompt_tokens": 10,
                                                   "completion_tokens": 5}})
    return h


def client(answer, calls):
    return ChatClient(http=httpx.Client(base_url="http://t",
                                        transport=httpx.MockTransport(fake_server(answer,
                                                                                  calls))))


def test_client_groups_and_falls_back_per_title(prompts):
    v3 = prompts.get("title-v3")
    calls = []

    def answer(body, k):
        if k > 1 and "Titre 3" in body["messages"][1]["content"]:
            return "1|0|other|none\n"  # groupe tronqué : refusé, puis repris titre par titre
        return "".join(f"{i}|-1|attack|security\n" for i in range(1, k + 1))

    out = client(answer, calls).label_many(items(10), v3, workers=2)
    assert len(out) == 10 and all(o["error"] is None for o in out)
    assert all(o["tone"] == -1 and o["themes"] == ["security"] for o in out)
    assert all(b["temperature"] == 0 and b["chat_template_kwargs"] == {"enable_thinking": False}
               for b in calls)
    sizes = sorted({o["group_size"] for o in out})
    assert sizes[0] == 1 and max(sizes) <= v3.batch  # repli titre par titre pour un groupe


def test_client_records_error_without_label(prompts):
    out = client(lambda body, k: None, []).label_many(items(1), prompts.get("title-v1"))
    assert out[0]["tone"] is None and "500" in out[0]["error"]


def test_make_groups_by_candidate_and_salt():
    its = items(9) + items(3, cid="zemmour_eric")
    g1 = make_groups(its, 4, salt="a")
    assert [len(g) for g in g1] == [4, 4, 1, 3]
    assert all(len({its[i].candidate_id for i in g}) == 1 for g in g1)
    assert make_groups(its, 4, salt="b") != g1  # autre version, autres voisins
    rev = list(reversed(its))  # les groupes ne dépendent que des story_id, pas de l'ordre
    assert ({tuple(its[i].story_id for i in g) for g in g1}
            == {tuple(rev[i].story_id for i in g) for g in make_groups(rev, 4, salt="a")})


def titles_frame(n=6):
    return pl.DataFrame({
        "story_id": [f"s{i}" for i in range(n)], "candidate_id": "pecresse_valerie",
        "cycle": 2022, "publish_date": [date(2022, 2, 7 + i) for i in range(n)],
        "outlet": "lci.fr", "title": [f"Titre {i}" for i in range(n)], "url": "u",
        "language": "fr", "in_title": [True] * (n - 1) + [False]})


def test_classify_missing_never_relabels(prompts, tmp_path):
    v3 = prompts.get("title-v3")
    calls = []
    fail = {"on": True}

    def answer(body, k):
        if fail["on"] and "Titre 0" in body["messages"][1]["content"] and k == 1:
            return None
        if fail["on"] and "Titre 0" in body["messages"][1]["content"]:
            return "garbage"
        return "".join(f"{i}|1|support|none\n" for i in range(1, k + 1))

    c = client(answer, calls)
    t = titles_frame()
    st = L.classify_missing(t, v3, c, MODEL, cache_dir=tmp_path, log=lambda _: None)
    assert st["labeled"] == 5 and st["errors"] == 1
    n_calls = len(calls)
    fail["on"] = False
    st = L.classify_missing(t, v3, c, MODEL, cache_dir=tmp_path, log=lambda _: None)
    assert st["todo"] == 1 and st["labeled"] == 1  # seule l'erreur repart
    assert len(calls) == n_calls + 1
    st = L.classify_missing(t, v3, c, MODEL, cache_dir=tmp_path, log=lambda _: None)
    assert st["todo"] == 0
    # autre modèle ou autre version : nouvelle clé
    assert L.classify_missing(t, v3, c, "autre@1", cache_dir=tmp_path,
                              log=lambda _: None)["todo"] == 6
    df = L.consolidate(tmp_path, write=False)
    assert df.height == 12 and df.select(L.KEY).is_duplicated().sum() == 0
    assert "raw" not in df.columns and df["prompt_digest"].unique().to_list() == [v3.digest]


def test_weekly_features(prompts):
    t = titles_frame().with_columns(
        pl.Series("publish_date", [date(2022, 2, 7)] * 3 + [date(2022, 2, 14)] * 3))
    t = pl.concat([t, t.with_columns(pl.lit("zemmour_eric").alias("candidate_id"),
                                     pl.lit(False).alias("in_title"))])
    lab = pl.DataFrame({"story_id": ["s0", "s1", "s2", "s3"],
                        "candidate_id": "pecresse_valerie", "tone": [-1, -1, 1, 0],
                        "stance": ["attack", "scandal", "support", "poll"],
                        "themes": [["scandals"], ["scandals", "security"], [], []]},
                       schema_overrides={"tone": pl.Int8})
    f = F.weekly_features(t, lab, prompts.stances, prompts.themes,
                          {2022: (date(2022, 2, 7), date(2022, 2, 16))})
    p = f.filter(pl.col("candidate_id") == "pecresse_valerie").sort("week")
    assert p["articles"].to_list() == [3, 3] and p["mentions"].to_list() == [3, 2]
    assert p["share_of_voice"].to_list() == [100.0, 100.0]
    w1 = p.row(0, named=True)
    assert w1["n_labeled"] == 3 and w1["tone"] == pytest.approx(-1 / 3)
    assert w1["attack_share"] == pytest.approx(100 / 3)
    assert w1["topic_scandals"] == pytest.approx(200 / 3)
    assert w1["topic_security"] == pytest.approx(100 / 3)
    assert p["days"].to_list() == [7, 3]
    z = f.filter(pl.col("candidate_id") == "zemmour_eric")
    assert z["n_labeled"].to_list() == [0, 0] and z["tone"].null_count() == 2
