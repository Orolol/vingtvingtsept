"""Rebonds d'annonce et de primaire : dérive μ_c(τ) de la projection (docs/04-modele.md, § 2).

L'équivalent du « convention bounce » de Silver. Après une primaire (ou une désignation), une
déclaration, un retrait ou un choc, la part d'un candidat dans les sondages saute, puis une
partie du saut se résorbe. Si la forme est commune à un type d'événement, une prévision faite
peu après l'événement doit anticiper la part du saut qui ne tiendra pas jusqu'au vote :

    θ_{c,0} = θ_{c,τ} + μ_c(τ) + η_c(τ)

Forme (échelle log-part, temps s en semaines depuis l'événement) :

    y(s) = pre + J · d(s),   d(s) = r + (1 − r) · exp(−s / T)   (s ≥ 0)

J est le saut, r ∈ [0, 1] la part durable, T la constante de décroissance. Trois paramètres
par événement, dont deux (r, T) partagés par un type d'événement (`fit_shape`, moindres carrés
non linéaires sur les sondages, pre et J propres à chaque événement, intervalles par
rééchantillonnage des événements).

Dérive (`candidate_drift`) : à la date de prévision, s semaines après l'événement, on observe
D = y(s) − pre = J · d(s). Entre la prévision et le vote (Δ = τ / 7 semaines), la forme prévoit

    μ = y(s + Δ) − y(s) = −D · (1 − r) · e^{−s/T} · (1 − e^{−Δ/T}) / d(s),

r tronqué à [0, 1] (la dérive ne sert qu'au retour partiel vers le niveau d'avant). D est
mesuré sur les sondages bruts (moyenne des log-parts des 8 semaines avant l'événement contre
celle des 14 derniers jours), avec son erreur type ; (r, T) sont tirés dans leur distribution
de rééchantillonnage : μ porte l'incertitude sur l'amplitude. La dérive s'applique au logit du
candidat, avant η (`ErrorModel.sample(..., drift=…)`), seulement aux candidats de la
configuration qui ont eu un événement des types retenus dans les `window_weeks` semaines
avant la date de prévision.

Série d'un candidat (`unit_series`) : pour chaque sondage, l'hypothèse qui contient le candidat
la plus proche de la configuration connue à la fin du terrain (`known_configuration`), part
normalisée dans l'hypothèse ; série de bloc : somme des parts des candidats du bloc dans
l'hypothèse la plus proche de la configuration. Un sondage à cheval sur la date de l'événement
est écarté ; pour une primaire à deux tours (`first_round`), le niveau d'avant s'arrête au
premier tour et les sondages de l'entre-deux-tours sont écartés. Si le candidat est testé par
moins de `PRE_MIN_POLLS` sondages dans les 8 semaines d'avant, la fenêtre est portée à 16
semaines, puis, à défaut, la série est celle de son bloc (`focal_series`). Pour un retrait, l'unité suivie est le premier bénéficiaire désigné, à
défaut le bloc du retiré.

Variante du backtest (`BOUNCE_VARIANTS`, option désactivée par défaut) : la forme est estimée
sans l'élection évaluée (leave-one-election-out), sur les élections 2002-2022 du catalogue
`data/registry/events.yaml`.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from vvs import paths
from vvs.registry import Registry, default_registry

EVENT_TYPES = ("primary", "declaration", "withdrawal", "shock")
EVENTS_PATH = paths.REGISTRY / "events.yaml"
BEFORE_WEEKS = 8.0
AFTER_WEEKS = 12.0
PRE_MIN_POLLS = 2
PRE_EXTEND_WEEKS = 16.0  # fenêtre d'avant élargie si le candidat est trop peu testé
POST_MIN_POLLS = 3
NOW_DAYS = 14  # fenêtre de la mesure « à la date de prévision »
SD_FLOOR = 0.10  # écart type minimal d'une log-part de sondage (erreur type de D)
R_BOUNDS = (0.0, 1.0)  # niveau durable entre le niveau d'avant (0) et le pic (1)
T_BOUNDS = (0.5, 52.0)
CALIBRATION_ELECTIONS = (2002, 2007, 2012, 2017, 2022)


# --- catalogue


@dataclass(frozen=True)
class Event:
    id: str
    election: int
    type: str
    date: dt.date
    candidate: str
    bloc: str
    sign: int | None = None
    first_round: dt.date | None = None
    beneficiaries: tuple[str, ...] = ()
    new_entrant: bool = False
    sources: tuple[str, ...] = ()
    note: str = ""

    @property
    def focal(self) -> str | None:
        """Candidat suivi : le candidat, ou pour un retrait son premier bénéficiaire (None : le
        bloc du retiré)."""
        if self.type == "withdrawal":
            return self.beneficiaries[0] if self.beneficiaries else None
        return self.candidate


def load_events(path: Path | None = None, registry: Registry | None = None) -> list[Event]:
    """Lit et valide `events.yaml`. Lève ValueError avec la liste de tous les problèmes."""
    reg = registry or default_registry()
    raw = yaml.safe_load(Path(path or EVENTS_PATH).read_text())
    rows = raw.get("events") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError("events.yaml : clé `events` absente ou pas une liste")  # noqa: TRY004
    errors: list[str] = []
    out: list[Event] = []
    seen: set[str] = set()
    allowed = {"id", "election", "type", "date", "first_round", "candidate", "bloc", "sign",
               "beneficiaries", "new_entrant", "sources", "note"}
    for i, r in enumerate(rows):
        where = f"événement {i} ({r.get('id', '?') if isinstance(r, dict) else '?'})"
        if not isinstance(r, dict):
            errors.append(f"{where} : pas un dictionnaire")
            continue
        extra = set(r) - allowed
        if extra:
            errors.append(f"{where} : champs inconnus {sorted(extra)}")
        eid = r.get("id")
        if not isinstance(eid, str) or not eid:
            errors.append(f"{where} : id manquant")
        elif eid in seen:
            errors.append(f"{where} : id en double")
        seen.add(eid)
        typ = r.get("type")
        if typ not in EVENT_TYPES:
            errors.append(f"{where} : type {typ!r} hors de {EVENT_TYPES}")
        date = r.get("date")
        if not isinstance(date, dt.date):
            errors.append(f"{where} : date {date!r} invalide")
        e = r.get("election")
        if e not in reg.elections:
            errors.append(f"{where} : élection {e!r} absente du registre")
        elif isinstance(date, dt.date) and date >= reg.elections[e].round_(1).date:
            errors.append(f"{where} : date postérieure au premier tour {e}")
        cid = r.get("candidate")
        if cid not in reg.candidates:
            errors.append(f"{where} : candidat {cid!r} absent du registre")
        elif e in reg.elections:
            try:
                b = reg.bloc_of(cid, e)
            except KeyError as exc:
                errors.append(f"{where} : {exc}")
            else:
                if r.get("bloc") != b:
                    errors.append(f"{where} : bloc {r.get('bloc')!r}, le registre dit {b!r}")
        sign = r.get("sign")
        if typ == "shock" and sign not in (-1, 1):
            errors.append(f"{where} : un choc doit avoir sign = +1 ou -1")
        if typ != "shock" and sign is not None:
            errors.append(f"{where} : sign réservé aux chocs")
        fr = r.get("first_round")
        if fr is not None and (typ != "primary" or not isinstance(fr, dt.date)
                               or not isinstance(date, dt.date) or fr >= date):
            errors.append(f"{where} : first_round réservé aux primaires, avant `date`")
        ben = r.get("beneficiaries") or []
        for b in ben:
            if b not in reg.candidates:
                errors.append(f"{where} : bénéficiaire {b!r} absent du registre")
        src = r.get("sources") or []
        if not src or not all(isinstance(s, str) and s.startswith("http") for s in src):
            errors.append(f"{where} : sources manquantes ou pas des URL")
        if not errors or not any(x.startswith(where) for x in errors):
            out.append(
                Event(
                    id=eid,
                    election=int(e),
                    type=typ,
                    date=date,
                    candidate=cid,
                    bloc=r["bloc"],
                    sign=sign,
                    first_round=fr,
                    beneficiaries=tuple(ben),
                    new_entrant=bool(r.get("new_entrant", False)),
                    sources=tuple(src),
                    note=str(r.get("note") or "").strip(),
                )
            )
    if errors:
        raise ValueError("events.yaml invalide :\n" + "\n".join(errors))
    return sorted(out, key=lambda x: (x.election, x.date, x.id))


# --- forme


def decay(s, r, T):
    """Part du saut encore présente s semaines après l'événement (s < 0 : traité comme 0)."""
    s = np.maximum(np.asarray(s, dtype=float), 0.0)
    return r + (1.0 - r) * np.exp(-s / T)


def trajectory(t, pre, jump, r, T):
    """y(t) = pre avant l'événement, pre + J · d(t) après."""
    t = np.asarray(t, dtype=float)
    return pre + jump * np.where(t >= 0, decay(t, r, T), 0.0)


def drift_factor(s, horizon_weeks, r, T):
    """μ / D : variation prévue entre s et s + Δ, rapportée au saut observé D = J · d(s)."""
    r = np.clip(r, 0.0, 1.0)
    num = -(1.0 - r) * np.exp(-s / T) * (1.0 - np.exp(-horizon_weeks / T))
    return num / decay(s, r, T)


@dataclass
class Shape:
    """Forme commune (r, T) d'un type d'événement et sa distribution de rééchantillonnage."""

    type: str
    r: float
    T: float
    r_draws: np.ndarray
    T_draws: np.ndarray
    events: list[str]
    jumps: dict[str, float] = field(default_factory=dict)
    pres: dict[str, float] = field(default_factory=dict)
    rss: float = float("nan")
    n_obs: int = 0

    def interval(self, q=(0.05, 0.5, 0.95)) -> dict:
        return {
            "r": [float(x) for x in np.quantile(self.r_draws, q)],
            "T": [float(x) for x in np.quantile(self.T_draws, q)],
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["r_draws"] = [float(x) for x in self.r_draws]
        d["T_draws"] = [float(x) for x in self.T_draws]
        d["interval"] = self.interval()
        return d


@dataclass
class Series:
    """Log-parts d'une unité autour d'un événement : t en semaines (milieu du terrain)."""

    t: np.ndarray
    y: np.ndarray
    unit: str = ""
    kind: str = "candidate"  # candidate ou bloc

    def window(self, lo: float, hi: float) -> np.ndarray:
        return self.y[(self.t >= lo) & (self.t < hi)]

    @property
    def n_pre(self) -> int:
        return int((self.t < 0).sum())

    @property
    def n_post(self) -> int:
        return int((self.t >= 0).sum())


def _fit_pooled(series: list[Series], x0=None):
    from scipy.optimize import least_squares

    E = len(series)
    t = [s.t for s in series]
    y = [s.y for s in series]

    def resid(x):
        r, T = x[0], math.exp(x[1])
        out = []
        for k in range(E):
            out.append(y[k] - trajectory(t[k], x[2 + k], x[2 + E + k], r, T))
        return np.concatenate(out)

    if x0 is None:
        pre0 = [float(np.mean(s.window(-1e9, 0))) for s in series]
        j0 = [float(np.mean(s.window(0, 4)) - p) if len(s.window(0, 4)) else 0.0
              for s, p in zip(series, pre0)]
        x0 = np.array([0.5, math.log(6.0), *pre0, *j0])
    lo = np.array([R_BOUNDS[0], math.log(T_BOUNDS[0])] + [-np.inf] * 2 * E)
    hi = np.array([R_BOUNDS[1], math.log(T_BOUNDS[1])] + [np.inf] * 2 * E)
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    res = least_squares(resid, x0, bounds=(lo, hi))
    return res.x, float((res.fun**2).sum())


def fit_shape(
    series: dict[str, Series], type_: str = "", n_boot: int = 300, seed: int = 0
) -> Shape | None:
    """Forme commune (r, T) sur des séries d'événements (pre et J propres à chaque événement).

    Intervalles : rééchantillonnage des événements avec remise (`n_boot` tirages). Rend None
    s'il y a moins de deux événements."""
    keys = [k for k, s in series.items() if s.n_pre >= PRE_MIN_POLLS and s.n_post >= POST_MIN_POLLS]
    if len(keys) < 2:
        return None
    ss = [series[k] for k in keys]
    x, rss = _fit_pooled(ss)
    E = len(ss)
    rng = np.random.default_rng(seed)
    r_d, T_d = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, E, E)
        if len(set(idx.tolist())) < 2:
            continue
        xb, _ = _fit_pooled([ss[i] for i in idx])
        r_d.append(xb[0])
        T_d.append(math.exp(xb[1]))
    return Shape(
        type=type_,
        r=float(x[0]),
        T=float(math.exp(x[1])),
        r_draws=np.array(r_d),
        T_draws=np.array(T_d),
        events=keys,
        jumps={k: float(x[2 + E + i]) for i, k in enumerate(keys)},
        pres={k: float(x[2 + i]) for i, k in enumerate(keys)},
        rss=rss,
        n_obs=int(sum(len(s.t) for s in ss)),
    )


# --- sondages d'une élection


class PollBook:
    """Sondages d'une élection, un par sondage, avec la configuration connue à la fin du terrain.

    `polls` : liste de dicts (`baselines._poll_rows`) : poll_id, fieldwork_start, fieldwork_end,
    hyps {hid: {candidat: part}} ; `config` : candidats connus à la fin du terrain."""

    def __init__(self, election: int, registry: Registry | None = None, polls_df=None):
        from vvs.eval.baselines import _poll_rows
        from vvs.eval.configurations import known_configuration
        from vvs.model.data import load_polls

        self.election = election
        self.reg = registry or default_registry()
        df = load_polls(election) if polls_df is None else polls_df
        self.polls = sorted(_poll_rows(df), key=lambda p: p["fieldwork_end"])
        cache: dict[dt.date, list[str]] = {}
        for p in self.polls:
            d = p["fieldwork_end"]
            if d not in cache:
                try:
                    cache[d] = known_configuration(election, d, self.reg, df).candidates
                except ValueError:
                    cache[d] = []
            p["config"] = cache[d]
            p["mid"] = p["fieldwork_start"] + (p["fieldwork_end"] - p["fieldwork_start"]) / 2
        self._bloc: dict[str, str] = {}

    def bloc_of(self, cid: str) -> str:
        if cid not in self._bloc:
            try:
                self._bloc[cid] = self.reg.bloc_of(cid, self.election)
            except KeyError:
                self._bloc[cid] = "other"
        return self._bloc[cid]

    def value(self, poll: dict, unit: str, kind: str = "candidate",
              config: list[str] | None = None) -> float | None:
        """Log-part de `unit` (candidat ou bloc) dans un sondage, hypothèse la plus proche de
        `config` (défaut : configuration connue à la fin du terrain)."""
        from vvs.eval.configurations import closest_hypothesis

        cfg = poll["config"] if config is None else config
        hyps = poll["hyps"]
        if kind == "candidate":
            hyps = {h: s for h, s in hyps.items() if unit in s}
            if not hyps:
                return None
        hid, _ = closest_hypothesis(hyps, cfg)
        shares = hyps[hid]
        total = sum(shares.values())
        if kind == "candidate":
            x = shares[unit]
        else:
            x = sum(v for c, v in shares.items() if self.bloc_of(c) == unit)
        if total <= 0 or x <= 0:
            return None
        return math.log(x / total)

    def series(self, unit: str, kind: str, date: dt.date, before: float = BEFORE_WEEKS,
               after: float = AFTER_WEEKS, asof: dt.date | None = None,
               pre_end: dt.date | None = None) -> Series:
        """Série de `unit` de `before` semaines avant `pre_end` (défaut : `date`) à `after`
        après `date` ; sondages terminés au plus tard à `asof`. Les sondages qui ne sont ni
        terminés avant `pre_end` ni commencés à `date` (à cheval, ou entre les deux tours d'une
        primaire) sont écartés. t est compté depuis `date`."""
        pre_end = pre_end or date
        lo = -before - (date - pre_end).days / 7.0
        t, y = [], []
        for p in self.polls:
            if asof is not None and p["fieldwork_end"] > asof:
                break
            if p["fieldwork_end"] >= pre_end and p["fieldwork_start"] < date:
                continue
            tw = (p["mid"] - date).days / 7.0
            if p["fieldwork_end"] < pre_end:
                tw = min(tw, -1e-6)
            if tw < lo or tw > after:
                continue
            v = self.value(p, unit, kind)
            if v is not None:
                t.append(tw)
                y.append(v)
        return Series(np.array(t), np.array(y), unit, kind)


def focal_series(book: PollBook, ev: Event, before: float = BEFORE_WEEKS,
                 after: float = AFTER_WEEKS, asof: dt.date | None = None) -> Series:
    """Série de l'unité suivie (voir le docstring du module) : candidat, fenêtre d'avant
    élargie à `PRE_EXTEND_WEEKS` s'il est trop peu testé, puis son bloc à défaut. Pour une
    primaire à deux tours, le niveau d'avant s'arrête au premier tour."""
    unit = ev.focal
    pre_end = ev.first_round or ev.date
    if unit is not None:
        for b in (before, max(before, PRE_EXTEND_WEEKS)):
            s = book.series(unit, "candidate", ev.date, b, after, asof, pre_end)
            if s.n_pre >= PRE_MIN_POLLS:
                return s
        bloc = book.bloc_of(unit)
    else:
        bloc = ev.bloc
    return book.series(bloc, "bloc", ev.date, before, after, asof, pre_end)


# --- dérive pour la projection


@dataclass(frozen=True)
class BounceSpec:
    """Réglages de la dérive de rebond. `types` : types d'événements retenus ; `window_weeks` :
    âge maximal de l'événement à la date de prévision ; `truncate` : séries d'estimation coupées
    au prochain événement visant la même unité (analyse de sensibilité)."""

    types: tuple[str, ...] = ("primary",)
    window_weeks: float = 12.0
    truncate: bool = False
    n_boot: int = 300


# Variantes du backtest qui ajoutent la dérive à la projection (mêmes ajustements que floor).
BOUNCE_VARIANTS: dict[str, BounceSpec] = {"bounce": BounceSpec()}

_BOOKS: dict[int, PollBook] = {}
_SHAPES: dict[tuple, Shape | None] = {}


def poll_book(election: int, registry: Registry | None = None) -> PollBook:
    if election not in _BOOKS:
        _BOOKS[election] = PollBook(election, registry)
    return _BOOKS[election]


def next_event_date(ev: Event, events: list[Event]) -> dt.date | None:
    """Date du prochain événement du catalogue qui vise la même unité (candidat ou bloc)."""
    later = [
        x.date for x in events
        if x.election == ev.election and x.date > ev.date and x.id != ev.id
        and (x.candidate == ev.focal or x.focal == ev.focal or x.bloc == ev.bloc)
    ]
    return min(later) if later else None


def type_series(events: list[Event], type_: str, elections, truncate: bool = False,
                registry: Registry | None = None) -> dict[str, Series]:
    """Séries focales des événements d'un type sur `elections`."""
    out = {}
    for ev in events:
        if ev.type != type_ or ev.election not in elections:
            continue
        book = poll_book(ev.election, registry)
        s = focal_series(book, ev)
        if truncate:
            nxt = next_event_date(ev, events)
            if nxt is not None:
                keep = s.t < (nxt - ev.date).days / 7.0
                s = Series(s.t[keep], s.y[keep], s.unit, s.kind)
        out[ev.id] = s
    return out


def shape_for(type_: str, exclude: int | None, spec: BounceSpec | None = None,
              events: list[Event] | None = None, registry: Registry | None = None) -> Shape | None:
    """Forme d'un type estimée sur 2002-2022 privées de `exclude` (mise en cache)."""
    spec = spec or BounceSpec()
    key = (type_, exclude, spec.truncate, spec.n_boot)
    if key not in _SHAPES:
        events = events if events is not None else load_events(registry=registry)
        el = [e for e in CALIBRATION_ELECTIONS if e != exclude]
        ser = type_series(events, type_, el, spec.truncate, registry)
        _SHAPES[key] = fit_shape(ser, type_, n_boot=spec.n_boot)
    return _SHAPES[key]


def observed_jump(book: PollBook, ev: Event, asof: dt.date) -> dict | None:
    """Saut D observé à `asof` (log-part, 14 derniers jours contre 8 semaines avant) et son
    erreur type. None si l'unité n'a pas assez de sondages."""
    s = focal_series(book, ev, after=1e9, asof=asof)
    pre = s.y[s.t < 0]
    age = (asof - ev.date).days / 7.0
    now = s.window(age - NOW_DAYS / 7.0, age + 1e-9)
    if len(now) == 0:
        now = s.window(max(age - 2 * NOW_DAYS / 7.0, 0), age + 1e-9)
    if len(pre) < PRE_MIN_POLLS or len(now) == 0:
        return None
    sd_pre = max(float(pre.std(ddof=1)) if len(pre) > 1 else SD_FLOOR, SD_FLOOR)
    sd_now = max(float(now.std(ddof=1)) if len(now) > 1 else SD_FLOOR, SD_FLOOR)
    return {
        "D": float(now.mean() - pre.mean()),
        "se": math.sqrt(sd_pre**2 / len(pre) + sd_now**2 / len(now)),
        "n_pre": len(pre),
        "n_now": len(now),
        "unit": s.unit,
        "kind": s.kind,
        "age_weeks": age,
    }


def candidate_drift(
    election: int,
    asof: dt.date,
    horizon_days: float,
    config: list[str],
    n: int,
    rng: np.random.Generator,
    spec: BounceSpec | None = None,
    shapes: dict[str, Shape | None] | None = None,
    events: list[Event] | None = None,
    book: PollBook | None = None,
    registry: Registry | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Dérive μ (n, K) sur le logit des candidats de `config` et le détail par événement.

    `shapes` : forme par type (défaut : `shape_for(type, exclude=election)`)."""
    spec = spec or BounceSpec()
    events = events if events is not None else load_events(registry=registry)
    mu = np.zeros((n, len(config)))
    details: list[dict] = []
    for ev in events:
        if ev.election != election or ev.type not in spec.types:
            continue
        age_days = (asof - ev.date).days
        if age_days < 0 or age_days > spec.window_weeks * 7:
            continue
        target = ev.focal
        if target not in config:
            continue
        shape = (shapes or {}).get(ev.type) if shapes is not None else shape_for(
            ev.type, election, spec, None, registry)
        if shape is None or len(shape.r_draws) == 0:
            continue
        bk = book or poll_book(election, registry)
        obs = observed_jump(bk, ev, asof)
        if obs is None:
            continue
        k = rng.integers(0, len(shape.r_draws), n)
        r, T = shape.r_draws[k], shape.T_draws[k]
        D = obs["D"] + obs["se"] * rng.standard_normal(n)
        s = age_days / 7.0
        m = D * drift_factor(s, horizon_days / 7.0, r, T)
        mu[:, config.index(target)] += m
        details.append(obs | {
            "event": ev.id, "candidate": target, "mu_median": float(np.median(m)),
            "mu_q05": float(np.quantile(m, 0.05)), "mu_q95": float(np.quantile(m, 0.95)),
            "mu_point": float(obs["D"] * drift_factor(s, horizon_days / 7.0, shape.r, shape.T)),
        })
    return mu, details


def variant_drift(
    variant: str,
    election: int,
    asof: dt.date,
    horizon_days: float,
    config: list[str],
    n: int,
    seed: int = 0,
    registry: Registry | None = None,
) -> np.ndarray | None:
    """Dérive de la variante de backtest `variant` (None si ce n'est pas une variante de
    rebond). Générateur propre (graine [seed, election, horizon, 7]) : η reste celui de la
    référence."""
    spec = BOUNCE_VARIANTS.get(variant)
    if spec is None:
        return None
    rng = np.random.default_rng([seed, election, int(horizon_days), 7])
    mu, _ = candidate_drift(election, asof, horizon_days, list(config), n, rng, spec,
                            registry=registry)
    return mu
