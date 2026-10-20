"""Conversion des codes géographiques du ministère de l'Intérieur en codes INSEE.

Le ministère code l'outre-mer et l'étranger en Z* (voir data/registry/interieur_sources.yaml).
Conventions de sortie :

- département : code INSEE sur 2 caractères (01…95, 2A, 2B) ou 3 (971…988) ; `ZZ` pour les
  Français établis hors de France ;
- commune : code INSEE sur 5 caractères ; à défaut de code INSEE (Wallis-et-Futuna, traité
  par le ministère comme une seule unité, et postes consulaires ou circonscriptions de
  l'étranger), code du ministère préfixé : `ZW001`, `ZZ001`… Ces codes ZZ ne désignent pas
  la même chose d'une élection à l'autre (circonscriptions en 2012, postes ensuite).

Saint-Martin et Saint-Barthélemy (ZX) sont regroupés par le ministère : on les sépare au
niveau département grâce au code commune (701 -> 977, 801 -> 978).
"""

from __future__ import annotations

import re

# Code ministère -> code département INSEE. ZX est résolu par la commune.
OVERSEAS = {
    "ZA": "971",
    "ZB": "972",
    "ZC": "973",
    "ZD": "974",
    "ZS": "975",
    "ZM": "976",
    "ZW": "986",
    "ZP": "987",
    "ZN": "988",
    "ZZ": "ZZ",
}
_ZX_COMMUNES = {"701": "977", "801": "978"}
# Unités sans code commune INSEE reconstructible
_NO_INSEE_COMMUNE = ("ZW", "ZZ")
_METRO = re.compile(r"^(0[1-9]|[1-8][0-9]|9[0-5]|2A|2B)$")


class GeoCodeError(ValueError):
    pass


def normalize_dept_src(code: object) -> str:
    """Code département tel que le ministère l'écrit, nettoyé : '1', 1.0 -> '01'."""
    if isinstance(code, float):
        if code != int(code):
            raise GeoCodeError(f"code département non entier : {code!r}")
        code = int(code)
    s = str(code).strip().upper()
    if s.isdigit() and len(s) == 1:
        s = "0" + s
    return s


def department(dept_src: object, commune_src: str | None = None) -> str:
    """Code département INSEE (ou ZZ) depuis le code ministère."""
    d = normalize_dept_src(dept_src)
    if _METRO.match(d):
        return d
    if d == "ZX":
        if commune_src is None or commune_src not in _ZX_COMMUNES:
            raise GeoCodeError(f"ZX sans commune 701/801 : {commune_src!r}")
        return _ZX_COMMUNES[commune_src]
    if d in OVERSEAS:
        return OVERSEAS[d]
    raise GeoCodeError(f"code département inconnu : {d!r}")


def commune(dept_src: object, commune_src: str) -> str:
    """Code commune INSEE (5 caractères) ou code ministère préfixé pour ZW et ZZ."""
    d = normalize_dept_src(dept_src)
    c = str(commune_src).strip()
    if not c.isdigit():
        raise GeoCodeError(f"code commune non numérique : {d} {c!r}")
    c = c.zfill(3)
    if _METRO.match(d):
        if len(c) != 3:
            raise GeoCodeError(f"code commune sur {len(c)} chiffres : {d} {c!r}")
        return d + c
    if d in _NO_INSEE_COMMUNE:
        return d + c
    # Outre-mer : département INSEE sur 3 chiffres + deux derniers chiffres du code ministère
    # (ZA 101 -> 97101, ZM 501 ou 601 -> 97601, ZP 011 ou 711 -> 98711, ZX 801 -> 97801).
    return department(d, c) + c[-2:]
