"""Tables d'alias de la Commission : libellés d'instituts et de candidats -> ids du registre.

Les libellés sont comparés après normalisation (`norm_label` : ASCII minuscule, ponctuation
réduite à des espaces). Un libellé inconnu fait échouer le parseur : on n'invente pas d'id.
Les ids sont ceux de `data/registry/pollsters.yaml` et `candidates.yaml` ; ces tables
locales pourront céder la place au chargeur du registre quand il couvrira les alias vus ici.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

from unidecode import unidecode


def norm_label(text: str) -> str:
    text = unidecode(text).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Instituts : alias normalisés -> id court. L'ordre compte peu : la recherche prend
# l'occurrence la plus à gauche, puis la plus longue.
POLLSTER_ALIASES: dict[str, str] = {
    "ifop": "ifop",
    "ifop fiducial": "ifop",
    "ipsos": "ipsos",
    "ipsos bva": "ipsos",
    "ipsos sopra steria": "ipsos",
    "bva": "bva",
    "elabe": "elabe",
    "harris interactive": "harris",
    "toluna harris interactive": "harris",
    "toluna harris": "harris",
    "toluna hi": "harris",
    "harris interactive toluna": "harris",
    "harris": "harris",
    "opinionway": "opinionway",
    "opinion way": "opinionway",
    "opnionway": "opinionway",
    "odoxa": "odoxa",
    "cluster17": "cluster17",
    "cluster 17": "cluster17",
    "verian": "verian",
    "kantar": "kantar",
    "yougov": "yougov",
    "csa": "csa",
    "viavoice": "viavoice",
    "thi": "harris",  # « Toluna Harris Interactive » abrégé dans certains titres
}

_POLLSTER_RE = re.compile(
    r"\b("
    + "|".join(sorted((re.escape(k) for k in POLLSTER_ALIASES), key=len, reverse=True))
    + r")\b"
)


def _norm_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalise caractère par caractère en gardant l'offset d'origine de chaque caractère."""
    out, offsets = [], []
    for i, ch in enumerate(text):
        n = unidecode(ch).lower()
        n = "".join(c if c.isalnum() else " " for c in n)
        for c in n:
            out.append(c)
            offsets.append(i)
    return "".join(out), offsets


def find_pollster_in_title(title: str) -> tuple[str, int, int] | None:
    """(id, début, fin) de la première mention d'institut dans `title`, offsets d'origine."""
    normed, offsets = _norm_with_offsets(title)
    m = _POLLSTER_RE.search(normed)
    if not m:
        return None
    return POLLSTER_ALIASES[m.group(1)], offsets[m.start()], offsets[m.end() - 1] + 1


def pollster_id(label: str) -> str:
    found = find_pollster_in_title(label)
    if not found:
        raise KeyError(f"institut inconnu : {label!r}")
    return found[0]


# Candidats : id du registre (convention nom_prenom ASCII) -> libellés vus dans les notices.
# Les variantes de casse, d'accents et d'espaces (« EricZemmour ») sont absorbées par
# `candidate_key` ; on ne liste ici que les formes réellement différentes (coquilles,
# prénom abrégé). Inclut les candidats de 2022, nécessaires pour reconnaître les tableaux de
# reconstitution du vote 2022 et les écarter.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "arthaud_nathalie": ("Nathalie Arthaud", "Nathalie Artaud"),
    "poutou_philippe": ("Philippe Poutou",),
    "kazib_anasse": ("Anasse Kazib",),
    "besancenot_olivier": ("Olivier Besancenot",),
    "roussel_fabien": ("Fabien Roussel",),
    "melenchon_jean_luc": ("Jean-Luc Mélenchon",),
    "ruffin_francois": ("François Ruffin",),
    "autain_clementine": ("Clémentine Autain",),
    "bompard_manuel": ("Manuel Bompard",),
    "panot_mathilde": ("Mathilde Panot",),
    "castets_lucie": ("Lucie Castets",),
    "tondelier_marine": ("Marine Tondelier",),
    "rousseau_sandrine": ("Sandrine Rousseau",),
    "jadot_yannick": ("Yannick Jadot",),
    "piolle_eric": ("Éric Piolle",),
    "glucksmann_raphael": ("Raphaël Glucksmann", "Raphaël Glucksman"),
    "hollande_francois": ("François Hollande",),
    "faure_olivier": ("Olivier Faure", "Oliver Faure"),
    "vallaud_boris": ("Boris Vallaud",),
    "royal_segolene": ("Ségolène Royal",),
    "guedj_jerome": ("Jérôme Guedj",),
    "maurel_emmanuel": ("Emmanuel Maurel",),
    "delga_carole": ("Carole Delga",),
    "cazeneuve_bernard": ("Bernard Cazeneuve",),
    "bouamrane_karim": ("Karim Bouamrane",),
    "hidalgo_anne": ("Anne Hidalgo",),
    "polony_natacha": ("Natacha Polony",),
    "montebourg_arnaud": ("Arnaud Montebourg",),
    "macron_emmanuel": ("Emmanuel Macron",),
    "philippe_edouard": ("Édouard Philippe",),
    "attal_gabriel": ("Gabriel Attal",),
    "darmanin_gerald": ("Gérald Darmanin",),
    "lecornu_sebastien": ("Sébastien Lecornu",),
    "bayrou_francois": ("François Bayrou",),
    "borne_elisabeth": ("Élisabeth Borne", "Elizabeth Borne"),
    "le_maire_bruno": ("Bruno Le Maire",),
    "braun_pivet_yael": ("Yaël Braun-Pivet",),
    "de_villepin_dominique": ("Dominique de Villepin",),
    "retailleau_bruno": ("Bruno Retailleau",),
    "wauquiez_laurent": ("Laurent Wauquiez",),
    "bertrand_xavier": ("Xavier Bertrand",),
    "lisnard_david": ("David Lisnard",),
    "pecresse_valerie": ("Valérie Pécresse",),
    "barnier_michel": ("Michel Barnier",),
    "dati_rachida": ("Rachida Dati",),
    "sarkozy_nicolas": ("Nicolas Sarkozy",),
    "dupont_aignan_nicolas": ("Nicolas Dupont-Aignan",),
    "lassalle_jean": ("Jean Lassalle",),
    "asselineau_francois": ("François Asselineau",),
    "philippot_florian": ("Florian Philippot",),
    "le_pen_marine": ("Marine Le Pen",),
    "bardella_jordan": ("Jordan Bardella",),
    "zemmour_eric": ("Éric Zemmour",),
    "marechal_marion": ("Marion Maréchal",),
    "knafo_sarah": ("Sarah Knafo",),
    "ciotti_eric": ("Éric Ciotti",),
    "de_villiers_philippe": ("Philippe de Villiers",),
    "hanouna_cyril": ("Cyril Hanouna",),
    "bonnal_sebastien": ("Sébastien Bonnal",),  # candidat déclaré, sondage OpinionWay 09/2026
    "leclerc_michel_edouard": ("Michel-Édouard Leclerc",),
}


def candidate_key(label: str) -> str:
    """Clé de comparaison : ASCII minuscule sans aucun séparateur (« EricZemmour »)."""
    return norm_label(label).replace(" ", "")


CANDIDATE_BY_KEY: dict[str, str] = {
    candidate_key(label): cid for cid, labels in CANDIDATES.items() for label in labels
}

# Libellés de lignes non candidates rencontrés dans les tableaux d'intentions de vote. Ils
# ne produisent pas de ligne de résultat (bases différentes des exprimés) mais ne déclenchent
# pas d'erreur. Comparaison par préfixe sur `norm_label`.
NON_CANDIDATE_PREFIXES: tuple[str, ...] = (
    "total",
    "sous total",
    "ensemble",
    "n expriment",
    "nexpriment",
    "ne se prononce",
    "ne sait pas",
    "nsp",
    "vous ne savez pas",
    "je ne sais pas",
    "sans reponse",
    "non repondu",
    "abstention",
    "vous vous abstiendriez",
    "vous n iriez pas voter",
    "vous n irez pas voter",
    "je ne voterais pas",
    "vote blanc",
    "blanc",
    "nul",
    "vous voteriez blanc",
    "vous voterez blanc",
    "autre",
    "aucun",
    "base",
    "personnes",
    "rappel",
    "non pris en compte",
    "ponderation",
    "effectif",
    "n ont pas exprime",
    "aucune",
    "sans opinion",
    "une autre",
    "un autre",
)


# Candidats supplémentaires actifs pour la lecture des notices d'archive (2017, 2022) :
# {clé `candidate_key` -> id}. Vide par défaut, de sorte que le parseur 2027 ne voit que
# `CANDIDATES` ; `archive_candidates` les active le temps d'une lecture (voir `archive`).
_EXTRA_CANDIDATES: ContextVar[Mapping[str, str] | None] = ContextVar(
    "extra_candidates", default=None
)


@contextmanager
def archive_candidates(labels: Mapping[str, list[str] | tuple[str, ...]]) -> Iterator[None]:
    """Active des libellés supplémentaires {id: libellés} pour `candidate_id`."""
    extra = {candidate_key(label): cid for cid, ls in labels.items() for label in ls}
    token = _EXTRA_CANDIDATES.set(extra)
    try:
        yield
    finally:
        _EXTRA_CANDIDATES.reset(token)


def candidate_id(label: str) -> str | None:
    """Id du candidat pour un libellé de ligne, None si ce n'est pas un candidat connu.

    Un suffixe de parti (« Marine Le Pen - Rassemblement National ») est retiré.
    """
    extra = _EXTRA_CANDIDATES.get() or {}
    if label in CANDIDATES or label in extra.values():  # déjà un id (tableaux transposés)
        return label
    variants = [label]
    parts = re.split(r"\s+[-–]\s+", label)
    if len(parts) == 2 and not re.search(r"\d", parts[1]):
        variants.append(parts[0])  # « Marine Le Pen - Rassemblement National » (YouGov)
    supported = re.split(r",\s*soutenue? par\b", label)
    if len(supported) == 2:
        variants.append(supported[0])  # « Yannick Jadot, soutenu par EELV » (Ipsos 2022)
    for variant in variants:
        key = candidate_key(variant)
        cid = CANDIDATE_BY_KEY.get(key) or extra.get(key)
        if cid:
            return cid
    return None


def is_non_candidate(label: str) -> bool:
    n = norm_label(label)
    return any(n.startswith(p) for p in NON_CANDIDATE_PREFIXES)
