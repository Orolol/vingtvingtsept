"""Client de l'API OpenAI-compatible d'un serveur d'inférence local (llama.cpp `llama-server`).

`/v1/chat/completions`, température 0, raisonnement désactivé
(`chat_template_kwargs.enable_thinking = false`), sortie contrainte au décodage :

- format `json` : un titre par requête, schéma JSON en `response_format` (llama-server le
  traduit en grammaire) ;
- format `lines` : les titres d'un même candidat sont groupés par `batch` (ordre tiré par
  une empreinte de story_id salée par la version de prompt, pour que deux versions ne voient
  pas les mêmes voisins), grammaire GBNF passée dans le champ `grammar` (extension de
  llama-server).

Les requêtes partent en parallèle (autant de fils que de slots du serveur). Une réponse non
conforme est retentée une fois ; au format `lines`, un groupe qui échoue encore est repris
titre par titre ; un titre qui échoue toujours est consigné comme erreur sans étiquette.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from vvs.nlp.prompts import LabelError, PromptSpec

BASE_URL = "http://127.0.0.1:8765"
TIMEOUT = httpx.Timeout(300.0, connect=10.0)
MAX_TOKENS = 96  # une sortie fait 20 à 50 jetons


@dataclass(frozen=True)
class Item:
    """Un titre à classer pour un candidat."""

    story_id: str
    candidate_id: str
    candidate: str  # nom affiché dans le prompt
    title: str
    date: str
    outlet: str


@dataclass
class ChatClient:
    base_url: str = BASE_URL
    model: str = "local"
    http: httpx.Client | None = None
    max_tokens: int = MAX_TOKENS
    retries: int = 1
    _own: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.http is None:
            self.http = httpx.Client(base_url=self.base_url, timeout=TIMEOUT)
            self._own = True

    def health(self) -> bool:
        try:
            return self.http.get("/health").status_code == 200
        except httpx.TransportError:
            return False

    def complete(self, messages: list[dict], schema: dict) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "label", "strict": True,
                                                "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = self.http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return r.json()

    def label(self, item: Item, spec: PromptSpec) -> dict:
        """Une étiquette validée, ou une erreur ; avec la latence et les jetons."""
        t0 = time.perf_counter()
        err, text, usage = None, None, {}
        for _ in range(1 + self.retries):
            try:
                out = self.complete(spec.messages(item.candidate, item.title, item.date,
                                                  item.outlet), spec.schema())
                text = out["choices"][0]["message"].get("content") or ""
                usage = out.get("usage") or {}
                lab = spec.parse(text)
                return {**lab, "error": None, "raw": text,
                        "latency_s": time.perf_counter() - t0,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens")}
            except (LabelError, httpx.HTTPError, KeyError, ValueError) as e:
                err = f"{type(e).__name__}: {e}"[:300]
        return {"tone": None, "stance": None, "themes": None, "error": err, "raw": text,
                "latency_s": time.perf_counter() - t0,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    def label_group(self, items: Sequence[Item], spec: PromptSpec) -> list[dict]:
        """Format `lines` : un groupe de titres d'un même candidat, une requête."""
        t0 = time.perf_counter()
        err, text, usage = None, None, {}
        msgs = spec.batch_messages(items[0].candidate,
                                   [(it.title, it.outlet, it.date) for it in items])
        for _ in range(1 + self.retries):
            try:
                body = {"model": self.model, "messages": msgs, "temperature": 0.0,
                        "max_tokens": self.max_tokens * len(items),
                        "grammar": spec.grammar(len(items)),
                        "chat_template_kwargs": {"enable_thinking": False}}
                r = self.http.post("/v1/chat/completions", json=body)
                r.raise_for_status()
                out = r.json()
                text = out["choices"][0]["message"].get("content") or ""
                usage = out.get("usage") or {}
                labs = spec.parse_lines(text, len(items))
                meta = {"error": None, "raw": text, "latency_s": time.perf_counter() - t0,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "group_size": len(items)}
                return [{**lab, **meta} for lab in labs]
            except (LabelError, httpx.HTTPError, KeyError, ValueError) as e:
                err = f"{type(e).__name__}: {e}"[:300]
        if len(items) > 1:
            return [lab for it in items for lab in self.label_group([it], spec)]
        return [{"tone": None, "stance": None, "themes": None, "error": err, "raw": text,
                 "latency_s": time.perf_counter() - t0,
                 "prompt_tokens": usage.get("prompt_tokens"),
                 "completion_tokens": usage.get("completion_tokens"), "group_size": 1}]

    def label_many(self, items: Sequence[Item], spec: PromptSpec, workers: int = 8
                   ) -> list[dict]:
        """Étiquettes dans l'ordre de `items`, `workers` requêtes en vol."""
        out: list[dict | None] = [None] * len(items)
        if spec.format == "json":
            groups = [[i] for i in range(len(items))]
        else:
            groups = make_groups(items, spec.batch, salt=spec.version)

        def run(g: list[int]) -> None:
            if spec.format == "json":
                out[g[0]] = {**self.label(items[g[0]], spec), "group_size": 1}
            else:
                for i, lab in zip(g, self.label_group([items[i] for i in g], spec)):
                    out[i] = lab

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run, groups))
        return out  # type: ignore[return-value]

    def close(self) -> None:
        if self._own and self.http is not None:
            self.http.close()


def _rank(story_id: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}:{story_id}".encode()).hexdigest()


def make_groups(items: Sequence[Item], k: int, salt: str) -> list[list[int]]:
    """Indices de `items` groupés par candidat, k au plus par groupe, dans un ordre tiré
    par empreinte (déterministe, indépendant de l'ordre d'entrée)."""
    by_cand: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by_cand.setdefault(it.candidate_id, []).append(i)
    groups = []
    for cid in sorted(by_cand):
        idx = sorted(by_cand[cid], key=lambda i: _rank(items[i].story_id, salt))
        groups += [idx[j:j + k] for j in range(0, len(idx), k)]
    return groups
