"""Lecteur minimal de classeurs Excel 97-2003 (.xls, BIFF8), sans dépendance.

Les résultats départementaux 2002 et 2007 du ministère ne sont publiés qu'en .xls, et
l'environnement n'a pas `xlrd`. Ce module lit juste ce qu'il faut : le conteneur OLE2,
la table de chaînes partagées (SST) et les cellules texte et numériques de chaque feuille.
Pas de formats, de dates ni de formules autres que leur résultat numérique ou texte.

Toute structure inattendue lève `XlsError` : on préfère un échec explicite à une lecture
silencieusement fausse.
"""

from __future__ import annotations

import struct
from pathlib import Path

_ENDOFCHAIN = 0xFFFFFFFE
_FREESECT = 0xFFFFFFFF
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# Enregistrements BIFF8 utilisés
_BOF = 0x0809
_EOF = 0x000A
_BOUNDSHEET = 0x0085
_SST = 0x00FC
_CONTINUE = 0x003C
_LABELSST = 0x00FD
_LABEL = 0x0204
_NUMBER = 0x0203
_RK = 0x027E
_MULRK = 0x00BD
_FORMULA = 0x0006
_STRING = 0x0207

Cell = str | float | None


class XlsError(ValueError):
    pass


# --- Conteneur OLE2 (Compound File Binary) ---------------------------------------------


def _ole_stream(data: bytes, name: str) -> bytes:
    """Renvoie le flux `name` du conteneur OLE2 `data`."""
    if data[:8] != _OLE_MAGIC:
        raise XlsError("pas un conteneur OLE2 (fichier .xls attendu)")
    sector_shift, mini_shift = struct.unpack_from("<HH", data, 0x1E)
    ssize, mssize = 1 << sector_shift, 1 << mini_shift
    (n_fat, first_dir) = struct.unpack_from("<II", data, 0x2C)
    (mini_cutoff, first_minifat, n_minifat, first_difat, n_difat) = struct.unpack_from(
        "<IIIII", data, 0x38
    )

    def sector(i: int) -> bytes:
        off = (i + 1) * ssize
        if off + ssize > len(data):
            raise XlsError(f"secteur {i} hors du fichier (fichier tronqué ?)")
        return data[off : off + ssize]

    # Liste des secteurs FAT : 109 entrées dans l'en-tête, puis chaîne DIFAT
    fat_sectors = list(struct.unpack_from("<109I", data, 0x4C))
    d = first_difat
    for _ in range(n_difat):
        entries = struct.unpack(f"<{ssize // 4}I", sector(d))
        fat_sectors.extend(entries[:-1])
        d = entries[-1]
    fat_sectors = [s for s in fat_sectors if s not in (_FREESECT, _ENDOFCHAIN)][:n_fat]
    fat: list[int] = []
    for s in fat_sectors:
        fat.extend(struct.unpack(f"<{ssize // 4}I", sector(s)))

    def chain(start: int) -> list[int]:
        out, s = [], start
        while s != _ENDOFCHAIN:
            if s >= len(fat) or len(out) > len(fat):
                raise XlsError("chaîne FAT invalide")
            out.append(s)
            s = fat[s]
        return out

    def read_chain(start: int) -> bytes:
        return b"".join(sector(s) for s in chain(start))

    directory = read_chain(first_dir)
    entries = []
    for off in range(0, len(directory), 128):
        e = directory[off : off + 128]
        name_len = struct.unpack_from("<H", e, 0x40)[0]
        ename = e[: max(name_len - 2, 0)].decode("utf-16-le", errors="replace")
        etype = e[0x42]
        start, size = struct.unpack_from("<II", e, 0x74)
        entries.append((ename, etype, start, size))
    root = entries[0]
    for ename, etype, start, size in entries:
        if etype != 2 or ename != name:
            continue
        if size >= mini_cutoff:
            return read_chain(start)[:size]
        # Petit flux : stocké dans le mini-flux de l'entrée racine
        ministream = read_chain(root[2])
        minifat: list[int] = []
        if n_minifat:
            raw = read_chain(first_minifat)
            minifat = list(struct.unpack(f"<{len(raw) // 4}I", raw))
        out, s = [], start
        while s != _ENDOFCHAIN:
            out.append(ministream[s * mssize : (s + 1) * mssize])
            s = minifat[s]
        return b"".join(out)[:size]
    raise XlsError(f"flux {name!r} absent")


# --- BIFF8 -------------------------------------------------------------------------------


def _records(stream: bytes, start: int = 0):
    pos = start
    n = len(stream)
    while pos + 4 <= n:
        rtype, rlen = struct.unpack_from("<HH", stream, pos)
        yield rtype, stream[pos + 4 : pos + 4 + rlen]
        pos += 4 + rlen


def _rk(v: int) -> float:
    if v & 2:
        x = float(v >> 2)
    else:
        x = struct.unpack("<d", struct.pack("<Q", (v & 0xFFFFFFFC) << 32))[0]
    if v & 1:
        x /= 100
    return x


def _unicode_str(data: bytes, pos: int, len_size: int) -> tuple[str, int]:
    """XLUnicodeString non fragmenté (longueur sur 1 ou 2 octets, puis drapeaux)."""
    if len_size == 1:
        nchars = data[pos]
    else:
        nchars = struct.unpack_from("<H", data, pos)[0]
    pos += len_size
    flags = data[pos]
    pos += 1
    if flags & 0x08:
        pos += 2  # runs de mise en forme : comptés mais non lus ici (rare hors SST)
    if flags & 0x04:
        pos += 4
    if flags & 0x01:
        s = data[pos : pos + 2 * nchars].decode("utf-16-le")
        pos += 2 * nchars
    else:
        s = data[pos : pos + nchars].decode("latin-1")
        pos += nchars
    return s, pos


def _parse_sst(chunks: list[bytes]) -> list[str]:
    """Table SST, éventuellement répartie sur des enregistrements CONTINUE.

    Une chaîne coupée par un CONTINUE reprend avec un octet de drapeaux (seul le bit
    « caractères sur 2 octets » compte) ; les runs de mise en forme et le bloc étendu qui
    suivent les caractères peuvent aussi déborder, sans octet de drapeaux.
    """
    data = chunks[0]
    _total, unique = struct.unpack_from("<II", data, 0)
    ci, pos = 0, 8
    out: list[str] = []
    for _ in range(unique):
        if pos >= len(data):
            ci += 1
            data, pos = chunks[ci], 0
        nchars, options = struct.unpack_from("<HB", data, pos)
        pos += 3
        rtcount = phosz = 0
        if options & 0x08:
            rtcount = struct.unpack_from("<H", data, pos)[0]
            pos += 2
        if options & 0x04:
            phosz = struct.unpack_from("<i", data, pos)[0]
            pos += 4
        parts: list[str] = []
        got = 0
        while True:
            need = nchars - got
            if options & 0x01:
                avail = min((len(data) - pos) // 2, need)
                parts.append(data[pos : pos + 2 * avail].decode("utf-16-le"))
                pos += 2 * avail
            else:
                avail = min(len(data) - pos, need)
                parts.append(data[pos : pos + avail].decode("latin-1"))
                pos += avail
            got += avail
            if got == nchars:
                break
            ci += 1
            data = chunks[ci]
            options = data[0]
            pos = 1
        pos += 4 * rtcount + phosz
        while pos > len(data):
            pos -= len(data)
            ci += 1
            data = chunks[ci]
        out.append("".join(parts))
    return out


def read_xls(path: Path | str) -> dict[str, list[list[Cell]]]:
    """Lit toutes les feuilles : {nom: lignes}, chaque ligne étant une liste de cellules
    (str, float ou None) ; toutes les lignes ont la largeur de la feuille, cellules vides à None."""
    stream = _ole_stream(Path(path).read_bytes(), "Workbook")
    sheets: list[tuple[str, int]] = []
    sst_chunks: list[bytes] = []
    in_sst = False
    for rtype, rec in _records(stream):
        if rtype == _BOF:
            version = struct.unpack_from("<H", rec, 0)[0]
            if version != 0x0600:
                raise XlsError(f"version BIFF {version:#x} non gérée (BIFF8 attendu)")
        elif rtype == _BOUNDSHEET:
            offset = struct.unpack_from("<I", rec, 0)[0]
            kind = rec[5]
            name, _ = _unicode_str(rec, 6, 1)
            if kind == 0:  # feuille de calcul (pas graphique ni macro)
                sheets.append((name, offset))
        elif rtype == _SST:
            sst_chunks = [rec]
            in_sst = True
            continue
        elif rtype == _CONTINUE and in_sst:
            sst_chunks.append(rec)
            continue
        elif rtype == _EOF:
            break
        in_sst = False
    sst = _parse_sst(sst_chunks) if sst_chunks else []

    result: dict[str, list[list[Cell]]] = {}
    for name, offset in sheets:
        cells: dict[tuple[int, int], Cell] = {}
        pending_formula: tuple[int, int] | None = None
        for rtype, rec in _records(stream, offset):
            if rtype == _EOF:
                break
            if rtype == _LABELSST:
                r, c, _xf, idx = struct.unpack_from("<HHHI", rec, 0)
                cells[r, c] = sst[idx]
            elif rtype == _NUMBER:
                r, c, _xf, v = struct.unpack_from("<HHHd", rec, 0)
                cells[r, c] = v
            elif rtype == _RK:
                r, c, _xf, v = struct.unpack_from("<HHHi", rec, 0)
                cells[r, c] = _rk(v)
            elif rtype == _MULRK:
                r, c0 = struct.unpack_from("<HH", rec, 0)
                n = (len(rec) - 6) // 6
                for k in range(n):
                    _xf, v = struct.unpack_from("<Hi", rec, 4 + 6 * k)
                    cells[r, c0 + k] = _rk(v)
            elif rtype == _LABEL:
                r, c, _xf = struct.unpack_from("<HHH", rec, 0)
                cells[r, c], _ = _unicode_str(rec, 6, 2)
            elif rtype == _FORMULA:
                r, c, _xf = struct.unpack_from("<HHH", rec, 0)
                if rec[12:14] == b"\xff\xff":
                    if rec[6] == 0:  # résultat texte : dans l'enregistrement STRING suivant
                        pending_formula = (r, c)
                else:
                    cells[r, c] = struct.unpack_from("<d", rec, 6)[0]
            elif rtype == _STRING and pending_formula is not None:
                cells[pending_formula], _ = _unicode_str(rec, 0, 2)
                pending_formula = None
        if cells:
            nrows = max(r for r, _ in cells) + 1
            ncols = max(c for _, c in cells) + 1
            rows: list[list[Cell]] = [[None] * ncols for _ in range(nrows)]
            for (r, c), v in cells.items():
                rows[r][c] = v
        else:
            rows = []
        result[name] = rows
    return result
