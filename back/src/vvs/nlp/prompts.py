"""Prompts versionnés (data/registry/nlp_prompts.yaml), contrainte de sortie et validation.

Une version de prompt est figée par son empreinte (`prompt_digest` du système rendu, du
gabarit utilisateur et, au format `lines`, du gabarit de titre, des champs et de la taille
de lot) : le chargeur refuse une version dont le texte ne correspond plus au `sha256`
enregistré, pour que le cache des étiquettes ne mélange jamais deux textes sous un même nom.

Gabarits : `{stances}`, `{themes}` (système) ; `{candidate}`, `{date}`, `{outlet}`, `{title}`
(utilisateur, format `json`) ; `{candidate}`, `{titles}` (utilisateur) et `{i}`, `{title}`,
`{outlet}`, `{date}` (titre, format `lines`). Remplacement littéral, pas de `str.format` :
le système contient des accolades JSON.

Formats :
- `json` : un titre par requête, schéma JSON (`schema`) passé en `response_format` ;
- `lines` : `batch` titres par requête, une ligne « numéro|…|…|… » par titre dans l'ordre de
  `fields`, imposée par une grammaire GBNF (`grammar`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from vvs import paths

TONES = (-1, 0, 1)
FIELDS = ("tone", "stance", "themes")
NO_THEME = "none"


class PromptError(ValueError):
    pass


class LabelError(ValueError):
    """Réponse du modèle non conforme au format attendu."""


def prompt_digest(system: str, user: str, extra: str = "") -> str:
    payload = f"{system}\n\x00\n{user}" + (f"\n\x00\n{extra}" if extra else "")
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _bullets(d: dict[str, str]) -> str:
    return "\n".join(f"- {k} : {v}" for k, v in d.items())


def _sub(template: str, **kw: str) -> str:
    for k, v in kw.items():
        template = template.replace("{" + k + "}", v)
    return template


@dataclass(frozen=True)
class PromptSpec:
    version: str
    format: str  # json | lines
    system: str  # rendu (listes fermées insérées)
    user: str  # gabarit
    item: str  # gabarit d'un titre (format lines)
    batch: int
    fields: tuple[str, ...]
    themes_grammar: str  # free | ordered (format lines)
    stances: tuple[str, ...]
    themes: tuple[str, ...]
    digest: str

    # --- format json (un titre)

    def messages(self, candidate: str, title: str, date: str, outlet: str) -> list[dict]:
        user = _sub(self.user, candidate=candidate, date=date, outlet=outlet or "inconnue",
                    title=title)
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def schema(self) -> dict:
        props = {
            "tone": {"type": "integer", "enum": list(TONES)},
            "stance": {"type": "string", "enum": list(self.stances)},
            "themes": {"type": "array", "items": {"type": "string", "enum": list(self.themes)},
                       "uniqueItems": True, "maxItems": len(self.themes)},
        }
        return {"type": "object", "properties": {k: props[k] for k in self.fields},
                "required": list(self.fields), "additionalProperties": False}

    def parse(self, text: str) -> dict:
        """Objet JSON -> {'tone': int, 'stance': str, 'themes': [str]} validé."""
        try:
            d = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise LabelError(f"JSON illisible : {str(text)[:120]!r}") from e
        if not isinstance(d, dict) or set(d) != set(self.fields):
            raise LabelError(f"clés inattendues : {d!r}"[:200])
        return self._check(d["tone"], d["stance"], d["themes"])

    def _check(self, tone, stance, themes) -> dict:
        if isinstance(tone, bool) or tone not in TONES:
            raise LabelError(f"tone hors {TONES} : {tone!r}")
        if stance not in self.stances:
            raise LabelError(f"stance inconnue : {stance!r}")
        if not isinstance(themes, list) or any(t not in self.themes for t in themes):
            raise LabelError(f"thèmes hors liste : {themes!r}")
        return {"tone": int(tone), "stance": stance, "themes": sorted(set(themes))}

    # --- format lines (plusieurs titres)

    def batch_messages(self, candidate: str, rows: Sequence[tuple[str, str, str]]
                       ) -> list[dict]:
        """`rows` = (titre, source, date) d'un même candidat, numérotés à partir de 1."""
        titles = "\n".join(_sub(self.item, i=str(i), title=t, outlet=o or "inconnue", date=d)
                           for i, (t, o, d) in enumerate(rows, 1))
        user = _sub(self.user, candidate=candidate, titles=titles)
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def grammar(self, k: int) -> str:
        """GBNF : exactement k lignes « i|champ|champ|champ\\n », i = 1..k."""
        rules = {
            "tone": 'tone ::= "-1" | "0" | "1"',
            "stance": "stance ::= " + " | ".join(f'"{s}"' for s in self.stances),
        }
        if self.themes_grammar == "ordered":
            # chaque thème au plus une fois, dans l'ordre de la liste : t_i suivi de rien ou
            # d'une virgule et d'un t_j, j > i
            n = len(self.themes)
            th = [f'themes ::= "{NO_THEME}" | ' + " | ".join(f"t{i}" for i in range(n))]
            for i, t in enumerate(self.themes):
                nxt = " | ".join(f"t{j}" for j in range(i + 1, n))
                th.append(f't{i} ::= "{t}"' + (f' ("," ({nxt}))?' if nxt else ""))
            rules["themes"] = "\n".join(th)
        else:
            rules["themes"] = (f'themes ::= "{NO_THEME}" | theme ("," theme)*\n'
                               + "theme ::= " + " | ".join(f'"{t}"' for t in self.themes))
        lab = ' "|" '.join(self.fields)
        lines = " ".join(f'"{i}|" lab "\\n"' for i in range(1, k + 1))
        return "\n".join([f"root ::= {lines}", f"lab ::= {lab}", *rules.values()]) + "\n"

    def parse_lines(self, text: str, k: int) -> list[dict]:
        lines = [x for x in (text or "").strip().split("\n") if x.strip()]
        if len(lines) != k:
            raise LabelError(f"{len(lines)} lignes au lieu de {k} : {str(text)[:120]!r}")
        out = []
        for i, line in enumerate(lines, 1):
            parts = line.strip().split("|")
            if len(parts) != 1 + len(self.fields) or parts[0] != str(i):
                raise LabelError(f"ligne {i} mal formée : {line!r}")
            d = dict(zip(self.fields, parts[1:]))
            try:
                tone = int(d["tone"])
            except ValueError as e:
                raise LabelError(f"tone illisible : {d['tone']!r}") from e
            themes = [] if d["themes"] == NO_THEME else d["themes"].split(",")
            if len(set(themes)) != len(themes):
                raise LabelError(f"thème répété : {d['themes']!r}")
            out.append(self._check(tone, d["stance"], themes))
        return out


@dataclass(frozen=True)
class NlpPrompts:
    model: dict
    stances: dict[str, str]
    themes: dict[str, str]
    prompts: dict[str, PromptSpec]

    def get(self, version: str) -> PromptSpec:
        if version not in self.prompts:
            raise PromptError(f"version de prompt inconnue : {version}")
        return self.prompts[version]


def load_prompts(path: Path | None = None, check_digest: bool = True) -> NlpPrompts:
    path = path or paths.REGISTRY / "nlp_prompts.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    stances = {str(k): str(v) for k, v in raw["stances"].items()}
    themes = {str(k): str(v) for k, v in raw["themes"].items()}
    if NO_THEME in themes:
        raise PromptError(f"« {NO_THEME} » est réservé (aucun thème)")
    out: dict[str, PromptSpec] = {}
    errors = []
    for version, p in raw["prompts"].items():
        fmt = p.get("format", "json")
        fields = tuple(p.get("fields") or FIELDS)
        if fmt not in ("json", "lines") or sorted(fields) != sorted(FIELDS):
            errors.append(f"{version}: format {fmt!r} ou champs {fields} invalides")
            continue
        system = _sub(p["system"], stances=_bullets(stances), themes=_bullets(themes))
        batch = int(p.get("batch", 1))
        item = p.get("item", "")
        tg = p.get("themes_grammar", "free")
        if tg not in ("free", "ordered"):
            errors.append(f"{version}: themes_grammar {tg!r} invalide")
        extra = f"{item}\n{batch}\n{','.join(fields)}" if fmt == "lines" else ""
        if tg != "free":
            extra += f"\n{tg}"
        digest = prompt_digest(system, p["user"], extra)
        if check_digest and p.get("sha256") != digest:
            errors.append(f"{version}: empreinte {digest} différente de sha256 "
                          f"{p.get('sha256')!r} (texte modifié : créer une nouvelle version)")
        out[version] = PromptSpec(version, fmt, system, p["user"], item, batch, fields, tg,
                                  tuple(stances), tuple(themes), digest)
    if errors:
        raise PromptError("nlp_prompts.yaml :\n  " + "\n  ".join(errors))
    return NlpPrompts(dict(raw["model"]), stances, themes, out)
