"""Scénarios historiques assemblés par décision de candidature (lot 2 de docs/12).

Protocole déclaré le 23 octobre 2026, **avant toute simulation et toute lecture de score** de
cette variante ; paramètres dans `data/registry/historical_decisions_2026-10-23.yaml`. La
référence `reference-2026-10-22`, ses fichiers et ses tirages restent inchangés. Variante
expérimentale : aucune activation live, aucune modification de la projection 2027.

Idée
----
La référence (docs/10) tire des listes entières effectivement sondées dans les 30 derniers
jours ; docs/11 a montré qu'élargir la mémoire de ces listes ne donne pas de probabilités de
candidature acceptables (fréquences de test des instituts, listes complètes indissociables).
Ici, on sépare deux incertitudes : **la candidature** de chaque personne, déduite de son état
daté dans le registre du lot 1 (`vvs.registry.candidacies.state_at`), et **le soutien**
conditionnel à la candidature, pris dans les ajustements figés du gel quand la personne y
figure, ou dans un prior déclaré sinon. Aucune liste sondée n'est réutilisée telle quelle.

(a) Probabilités de candidature
-------------------------------
État à la date `asof` (événements publiés strictement avant `asof`) :

- `official` → 1 ; `designated` → 1 ;
- `declared` → p_declared = 0,9 ;
- `possible` → p_possible = 0,15, sauf membre d'un processus en attente (ci-dessous) ;
- `undetermined` → p_undetermined = 0,3 ;
- `excluded`, `withdrawn` → 0 ; une personne sans trace publiée avant la date n'existe pas.

Un **processus de désignation** du registre (primaire, congrès, consultation) est *visible* à la
date si au moins une entrée (`primary_entry`) y est publiée avant la date ; il est *en attente*
si aucune désignation n'est publiée. Ses **membres** à la date sont ses entrants encore actifs,
plus, tant qu'aucun tour n'a été publié, les personnes à l'état `possible` dont le parti
(registre du lot 1) figure parmi ses organisateurs : elles peuvent encore y entrer. Le processus
désigne exactement un membre, chacun avec probabilité 1/n (n membres à la date), et ce désigné
est candidat (probabilité 1). Les membres sont mutuellement exclusifs. Les champs du registre qui
révèlent l'avenir (`participants`, `winner`, tours non publiés) ne sont jamais lus ; un
processus sans entrant publié n'existe pas encore pour le modèle. Ces valeurs sont des choix de
protocole communs aux trois élections. Sensibilité annoncée, calculée et publiée sans
sélection : p_declared ∈ {0,8 ; 0,99} (0,9 + 0,1 = 1 est ramené à 0,99 : les probabilités des
états `declared`, `possible` et `undetermined` sont bornées à [0,01 ; 0,99]), p_possible ∈
{0,05 ; 0,25}. Les décisions indépendantes sont des Bernoulli indépendantes.

(b) Contraintes
---------------
- Au plus un candidat par processus en attente (règle ci-dessus).
- Engagements documentés : un engagement (`commitments`) publié avant la date sur un processus
  en attente force l'absence de la personne engagée dans les configurations où l'un des
  `winners` visés est désigné (présence ramenée à 0, comme les contraintes `absent` de
  scenarios_2027.yaml). Après la désignation, `state_at` applique déjà la règle.
- Plafonds par bloc tirés de l'histoire : pour l'élection E, le nombre maximal de candidats
  d'un bloc (et le nombre total) sur les listes officielles des présidentielles du registre
  strictement antérieures à E (elections.yaml : 2002 et 2007 pour 2012 ; plus 2012 pour 2017 ;
  plus 2017 pour 2022). Valeurs : extrême gauche 5, gauche 4, centre 1, droite 4, extrême
  droite 2, autres 1 (3 pour 2022), total 16. Le bloc d'une personne est celui du registre du
  lot 1. Si les personnes certaines d'un bloc (probabilité 1, désigné d'un processus) dépassent
  le plafond, il est porté à leur nombre. Les configurations au-delà des plafonds sont retirées
  et la masse renormalisée (loi des Bernoulli conditionnée aux plafonds).

(c) Assemblage
--------------
La loi est calculée exactement : énumération nominative des issues des processus, puis, par
bloc, des sous-ensembles de candidats à soutien estimé et des effectifs de candidats sans
soutien estimé (Poisson-binomiale), combinés sous les plafonds ; les combinaisons de
probabilité conditionnelle < 1e-12 sont élaguées et leur masse est publiée. Une
**configuration** est une *signature* : l'ensemble des candidats à soutien estimé présents et,
pour les candidats sans soutien estimé, leur nombre par *type* (bloc, centre du prior). À
signature égale, la simulation est identique en loi ; les noms des candidats sans soutien
estimé sont tirés tirage par tirage dans la loi conditionnelle exacte de la signature
(générateur séparé, graine [seed, 2]), ce qui conserve leur probabilité de présence nominative.

Plafond : on retient les signatures les plus probables jusqu'à couvrir 95 % de la masse, sans
dépasser 200 ; le reste est regroupé par état de présence des **gros candidats** (soutien estimé
et part médiane a posteriori ≥ 5 % parmi les actifs à soutien estimé). Les 40 000 tirages sont
répartis entre signatures retenues et groupes par plus forts restes (règle `allocate` du gel) ;
dans chaque groupe (strate), ils sont répartis entre ses signatures par tirage aléatoire
proportionnel à leur masse (graine [seed, 3]). L'audit publie la couverture, la masse regroupée
et l'écart de présence de chaque personne entre la loi exacte et la loi représentée.

Amendement 1 (23 octobre 2026, après un essai à blanc de construction des configurations, sans
aucune simulation ni score de la variante) : le texte initial représentait chaque groupe par sa
signature la plus probable, qui recevait toute sa masse. À 180 jours, les 200 premières
signatures ne couvrent que 42 % (2017) et 15 % (2022) de la masse et ce représentant déplaçait
la présence d'une personne jusqu'à 0,29 ; le tirage stratifié ci-dessus le remplace.

Soutien. Un candidat a un soutien estimé s'il figure dans les tirages de l'ajustement figé du cas
(au moins une hypothèse sondée dans sa fenêtre). Sinon, prior déclaré : log(part) ~ N(log c, 1),
où c est la médiane de ses parts dans les sondages disponibles du cycle (toutes hypothèses,
premier tour), à défaut sa médiane dans les sondages de la présidentielle précédente, à défaut
1 % ; parts bornées à [1e-5 ; 0,5], total des priors d'un tirage ramené à 0,9 au plus. Le
candidat reste présent avec sa probabilité de candidature : il ne reçoit jamais les voix d'un
autre candidat ni sa popularité future. Agrégateur bayésien : log-parts des candidats à soutien
estimé tirées comme dans le gel (`draws_log_shares`), multipliées par (1 − somme des priors) ;
tirages des priors par un générateur séparé (graine [seed, 1]). Témoin `mean14d` : règle du gel
appliquée à chaque nom (moyenne 14 jours, sinon médiane du cycle disponible, sinon 1 %). Puis,
pour tous : même erreur de premier tour (bloc et candidat), même qualification, même second tour
par configuration et par duel, mêmes scores, graine du gel. Les candidats sans soutien estimé
sont simulés sous des pseudo-noms de type ajoutés en mémoire au registre du gel (bloc seulement),
puis renommés. Sans candidat à prior, la fonction reproduit `simulate_chain` du gel (contrôlé
par rejeu des douze cas). Ajustements figés réutilisés, aucun NUTS relancé.

La liste officielle remplace les scénarios exactement à la même date que dans la référence ; à
30 et 7 jours, les tirages de la référence sont repris à l'identique.

(d) Critères de promotion (fixés avant calcul)
----------------------------------------------
Agrégateur bayésien, univers communs de `stage_diagnostics.compare_study` ; mean14d publié comme
témoin. Erreur Monte-Carlo : sqrt((1 − p)/(n p)) pour les événements binaires, sinon écart type
sur 20 lots aléatoires de 2 000 tirages divisé par sqrt(20) ; une différence combine les deux
erreurs en racine de la somme des carrés.

1. Présence (étape visée) : à 180 et 90 jours, log P(W présent), log P(paire présente) et Brier
   de présence (somme sur l'univers commun) meilleurs sur au moins 2 élections sur 3.
2. Aval : à 180 et 90 jours, moyennes par horizon de log P(paire | présence), Brier de
   qualification | présence, CRPS et couverture 90 % conditionnels non dégradés au-delà de
   2 erreurs Monte-Carlo.
3. P(W | paire réelle) inchangée à 2 erreurs Monte-Carlo dans chaque cas (sinon bug).
4. Probabilité qu'un fantôme se qualifie quand W est présent non croissante (au-delà de 2 erreurs
   Monte-Carlo) dans chaque cas à 180 et 90 jours.
5. Écarts exactement nuls à 30 et 7 jours.
6. Critère global de docs/07 : log-score moyen de la paire et CRPS améliorés à 180 et 90 jours,
   gain de même signe sur 2 élections sur 3, couverture 90 % non dégradée, aucun autre horizon
   dégradé.

Promotion seulement si tous sont satisfaits ; c'est de toute façon une étude exploratoire sur
les élections qui ont servi au diagnostic.

Limites connues au moment de la déclaration : registre reconstruit en 2026 (dates de première
trace bornées par les sources parcourues), blocs et partis rétrospectifs, plafonds fondés sur
deux à quatre listes, prior de soutien non validé, univers de score élargi (lissage de Jeffreys
du score global dépendant du nombre de noms).
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

ACTIVE = ("possible", "declared", "designated", "undetermined", "official")
SLOT_PREFIX = "~slot"


# --------------------------------------------------------------------------- réglages


@dataclass(frozen=True)
class Settings:
    declared: float = 0.9
    possible: float = 0.15
    undetermined: float = 0.3
    designated: float = 1.0
    official: float = 1.0
    clip_low: float = 0.01
    clip_high: float = 0.99
    prior_sigma: float = 1.0
    prior_default: float = 0.01
    prior_low: float = 1e-5
    prior_high: float = 0.5
    prior_total_max: float = 0.9
    max_configurations: int = 200
    coverage: float = 0.95
    large_share: float = 0.05
    prune: float = 1e-12

    @classmethod
    def from_protocol(cls, protocol: dict, variant: str) -> Settings:
        p = protocol["probabilities"]
        low, high = protocol["probability_clip"]
        a = protocol["assembly"]
        s = protocol["support"]
        base = cls(
            declared=p["declared"],
            possible=p["possible"],
            undetermined=p["undetermined"],
            designated=p["designated"],
            official=p["official"],
            clip_low=low,
            clip_high=high,
            prior_sigma=1.0,
            prior_low=s["prior_clip"][0],
            prior_high=s["prior_clip"][1],
            prior_total_max=s["prior_total_max"],
            max_configurations=a["max_configurations"],
            coverage=a["coverage"],
        )
        if variant == protocol["main_variant"]:
            return base
        for entry in protocol["sensitivity"]:
            if entry["id"] == variant:
                changes = {k: v for k, v in entry.items() if k != "id"}
                return cls(**(asdict(base) | changes))
        raise ValueError(f"variante inconnue : {variant}")


def variants(protocol: dict) -> list[str]:
    return [protocol["main_variant"], *(e["id"] for e in protocol["sensitivity"])]


def candidacy_probability(state: str, s: Settings) -> float:
    """Probabilité de candidature d'une décision indépendante selon l'état à la date."""
    if state in ("excluded", "withdrawn"):
        return 0.0
    if state == "official":
        return s.official
    if state == "designated":
        return s.designated
    raw = {"declared": s.declared, "possible": s.possible, "undetermined": s.undetermined}
    if state not in raw:
        raise ValueError(f"état inconnu : {state}")
    return min(max(raw[state], s.clip_low), s.clip_high)


# --------------------------------------------------------------------------- décisions


@dataclass(frozen=True)
class Pool:
    """Processus en attente : un seul membre désigné, chacun avec probabilité 1/n."""

    id: str
    members: tuple[str, ...]
    entrants: tuple[str, ...]
    joined_possible: tuple[str, ...]
    started: bool


@dataclass(frozen=True)
class Commitment:
    person: str
    process: str
    winners: frozenset[str]


@dataclass
class Decisions:
    asof: dt.date
    states: dict  # personne -> PersonState (actives et inactives tracées)
    independent: dict[str, float]  # personne -> probabilité de candidature
    pools: list[Pool]
    commitments: list[Commitment]
    blocs: dict[str, str]  # personne active -> bloc du registre du lot 1
    parties: dict[str, str | None]


def decision_units(cr, asof: dt.date, settings: Settings) -> Decisions:
    """Décisions de candidature à la date, d'après les seuls événements publiés avant elle."""
    states = cr.state_at(asof)
    active = {p: s for p, s in states.items() if s.state in ACTIVE}
    published = cr.published_before(asof)
    pools: list[Pool] = []
    taken: dict[str, str] = {}
    commitments: list[Commitment] = []
    for pid in sorted(cr.processes):
        proc = cr.processes[pid]
        entries = [e.person for e in published if e.type == "primary_entry" and e.process == pid]
        if not entries:
            continue  # processus pas encore visible
        if any(e.type == "designation" and e.process == pid for e in published):
            continue  # tranché : state_at a déjà appliqué ses exclusions
        started = any(r.published < asof for r in proc.rounds)
        entrants = tuple(sorted({p for p in entries if p in active}))
        joined = ()
        if not started:
            joined = tuple(
                sorted(
                    p
                    for p, s in active.items()
                    if s.state == "possible"
                    and cr.persons[p].party in proc.organizers
                    and p not in entrants
                )
            )
        members = entrants + joined
        if not members:
            continue
        for m in members:
            if m in taken:
                raise ValueError(f"{m} membre de deux processus en attente : {taken[m]}, {pid}")
            taken[m] = pid
        pools.append(Pool(pid, members, entrants, joined, started))
        for c in proc.commitments:
            if c.published < asof and c.person in active:
                commitments.append(Commitment(c.person, pid, frozenset(c.winners)))
    independent = {
        p: candidacy_probability(s.state, settings)
        for p, s in sorted(active.items())
        if p not in taken
    }
    return Decisions(
        asof=asof,
        states=states,
        independent=independent,
        pools=pools,
        commitments=commitments,
        blocs={p: cr.persons[p].bloc for p in active},
        parties={p: cr.persons[p].party for p in active},
    )


def bloc_caps(registry, election: int) -> tuple[dict[str, int], int]:
    """Plafonds par bloc et total : maxima des listes officielles strictement antérieures."""
    from collections import Counter

    past = [
        y
        for y, e in sorted(registry.elections.items())
        if y < election and e.candidates
    ]
    if not past:
        raise ValueError(f"aucune liste officielle antérieure à {election}")
    caps: dict[str, int] = defaultdict(int)
    total = 0
    for y in past:
        counts = Counter(c.bloc for c in registry.elections[y].candidates)
        for b, k in counts.items():
            caps[b] = max(caps[b], k)
        total = max(total, len(registry.elections[y].candidates))
    return dict(caps), total


# --------------------------------------------------------------------------- loi exacte


def poisson_binomial(ps: list[float], kmax: int) -> np.ndarray:
    """P(exactement k succès), k = 0..kmax (troncature : masse au-delà ignorée)."""
    pmf = np.zeros(kmax + 1)
    pmf[0] = 1.0
    for p in ps:
        nxt = pmf * (1 - p)
        nxt[1:] += pmf[:-1] * p
        pmf = nxt
    return pmf


def tail_table(ps: list[float]) -> np.ndarray:
    """T[i, r] = P(exactement r succès parmi ps[i:]) ; T[m, 0] = 1."""
    m = len(ps)
    table = np.zeros((m + 1, m + 1))
    table[m, 0] = 1.0
    for i in range(m - 1, -1, -1):
        table[i] = table[i + 1] * (1 - ps[i])
        table[i, 1:] += table[i + 1, :-1] * ps[i]
    return table


def conditional_inclusion(ps: list[float], k: int) -> np.ndarray:
    """P(i inclus | exactement k succès) pour des Bernoulli indépendantes."""
    m = len(ps)
    if k == 0:
        return np.zeros(m)
    total = poisson_binomial(ps, m)[k]
    if total <= 0:
        raise ValueError("effectif impossible pour ces probabilités")
    out = np.zeros(m)
    for i, p in enumerate(ps):
        if p == 0:
            continue
        others = poisson_binomial(ps[:i] + ps[i + 1 :], m)
        out[i] = p * others[k - 1] / total
    return out




@dataclass
class Law:
    """Loi des signatures : masse et composantes (indice de l'issue des processus, poids)."""

    signatures: dict  # clé -> {"mass": float, "components": {omega: poids}}
    omegas: list[dict[str, str]]
    pruned_bound: float  # borne de la masse élaguée, relative à la masse admissible
    admissible: float  # log10 de la probabilité des contraintes sous la loi sans contrainte
    caps: dict[str, int]
    total_cap: int
    presence: dict[str, float]  # présence exacte de chaque personne active


def _signature_key(fitted, counts: dict[str, int]) -> tuple:
    return (tuple(sorted(fitted)), tuple(sorted((t, k) for t, k in counts.items() if k)))


def enumerate_law(
    dec: Decisions,
    fitted: set[str],
    ptype: dict[str, str],
    caps: dict[str, int],
    total_cap: int,
    prune: float = 1e-12,
) -> Law:
    """Loi exacte (à l'élagage près) des signatures sous processus, engagements et plafonds.

    `fitted` : personnes à soutien estimé ; `ptype` : type (bloc, centre) des autres.
    Par bloc, les options (candidats à soutien estimé, effectifs par type) sont d'abord
    conditionnées au plafond du bloc (constante de normalisation Z_b, qui pondère les issues
    des processus), puis élaguées sous `prune` ; le plafond total s'applique au produit.
    """
    blocs = dec.blocs
    pools = dec.pools
    omegas = [
        dict(zip([p.id for p in pools], combo))
        for combo in itertools.product(*[p.members for p in pools])
    ]
    pool_people = {m for p in pools for m in p.members}
    forced_people = {c.person for c in dec.commitments}
    affected = {blocs[m] for m in pool_people | forced_people}
    all_blocs = sorted({blocs[p] for p in dec.independent} | {blocs[m] for m in pool_people})
    dropped = [0.0]

    def bloc_options(b: str, nominees: list[str], forced: set[str]):
        """Options normalisées sous le plafond du bloc et log Z_b (−inf si impossible)."""
        indep = [
            (p, q)
            for p, q in dec.independent.items()
            if blocs[p] == b and p not in forced and q > 0
        ]
        fit = [(p, q) for p, q in indep if p in fitted]
        by_type: dict[str, list[float]] = defaultdict(list)
        for p, q in indep:
            if p not in fitted:
                by_type[ptype[p]].append(q)
        certain = len(nominees) + sum(1 for _, q in indep if q >= 1)
        room = max(caps.get(b, 0), certain) - len(nominees)
        nom_fit = frozenset(m for m in nominees if m in fitted)
        nom_types: dict[str, int] = defaultdict(int)
        for m in nominees:
            if m not in fitted:
                nom_types[ptype[m]] += 1
        types = sorted(by_type)
        pmfs = [poisson_binomial(by_type[t], room) for t in types]
        raw = []
        for mask in itertools.product((False, True), repeat=len(fit)):
            pf, chosen = 1.0, []
            for (p, q), on in zip(fit, mask):
                pf *= q if on else 1 - q
                if on:
                    chosen.append(p)
            if pf == 0 or len(chosen) > room:
                continue
            for counts in itertools.product(range(room - len(chosen) + 1), repeat=len(types)):
                if len(chosen) + sum(counts) > room:
                    continue
                pc = pf * math.prod(pmfs[i][k] for i, k in enumerate(counts))
                if pc > 0:
                    raw.append((chosen, counts, pc))
        z = math.fsum(r[2] for r in raw)
        if z <= 0:
            return [], -math.inf
        options = []
        for chosen, counts, pc in raw:
            pc /= z
            if pc <= prune:
                dropped[0] += pc
                continue
            tc = dict(nom_types)
            for t, k in zip(types, counts):
                if k:
                    tc[t] = tc.get(t, 0) + k
            options.append(
                (nom_fit | frozenset(chosen), tc, pc,
                 len(nominees) + len(chosen) + sum(counts), certain)
            )
        return options, math.log(z)

    def combine(left, right):
        out = []
        for f1, t1, p1, n1, c1 in left:
            for f2, t2, p2, n2, c2 in right:
                p = p1 * p2
                if p <= prune:
                    dropped[0] += p
                    continue
                t = dict(t1)
                for k, v in t2.items():
                    t[k] = t.get(k, 0) + v
                out.append((f1 | f2, t, p, n1 + n2, c1 + c2))
        return out

    static = [(frozenset(), {}, 1.0, 0, 0)]
    log_static = 0.0
    log_z: dict[str, float] = {}
    for b in all_blocs:
        if b not in affected:
            opts, lz = bloc_options(b, [], set())
            static = combine(static, opts)
            log_static += lz
            log_z[b] = lz
    static_dropped = dropped[0]
    pool_log = sum(-math.log(len(p.members)) for p in pools)
    per_omega = []
    for wi, omega in enumerate(omegas):
        nominees = set(omega.values())
        forced = {c.person for c in dec.commitments if omega[c.process] in c.winners}
        if forced & nominees:
            continue  # issue impossible : désigné engagé à ne pas se présenter
        dropped[0] = 0.0
        current, lw = static, pool_log
        for b in all_blocs:
            if b in affected:
                opts, lz = bloc_options(b, sorted(m for m in nominees if blocs[m] == b), forced)
                current = combine(current, opts)
                lw += lz
        kept = [o for o in current if o[3] <= max(total_cap, o[4])]
        mass = math.fsum(o[2] for o in kept)
        if mass > 0 and lw > -math.inf:
            per_omega.append((wi, lw + math.log(mass), kept, mass, dropped[0] + static_dropped))
    if not per_omega:
        raise ValueError("aucune configuration admissible")
    top = max(x[1] for x in per_omega)
    weights = {wi: math.exp(lw - top) for wi, lw, *_ in per_omega}
    norm = math.fsum(weights.values())
    signatures: dict = {}
    dropped_total = 0.0
    presence: dict[str, float] = defaultdict(float)
    by_type: dict[str, list] = defaultdict(list)
    for p, q in dec.independent.items():
        if p not in fitted and q > 0:
            by_type[ptype[p]].append((p, q))
    inclusion: dict = {}
    for wi, lw, kept, mass, drop in per_omega:
        w = weights[wi] / norm
        dropped_total += w * drop / mass
        omega = omegas[wi]
        forced = frozenset(c.person for c in dec.commitments if omega[c.process] in c.winners)
        nominee_types: dict[str, int] = defaultdict(int)
        for m in omega.values():
            presence[m] += w
            if m not in fitted:
                nominee_types[ptype[m]] += 1
        counts: dict[tuple, float] = defaultdict(float)
        for f, t, p, n, certain in kept:
            q = w * p / mass
            key = _signature_key(f, t)
            entry = signatures.setdefault(key, {"mass": 0.0, "components": defaultdict(float)})
            entry["mass"] += q
            entry["components"][wi] += q
            for name in f:
                if name not in omega.values():
                    presence[name] += q
            for tid, k in t.items():
                counts[(tid, k - nominee_types.get(tid, 0))] += q
        for (tid, k), q in counts.items():
            people = [(p, r) for p, r in by_type.get(tid, []) if p not in forced]
            ckey = (tid, k, forced)
            if ckey not in inclusion:
                inclusion[ckey] = conditional_inclusion([r for _, r in people], k)
            for (p, _), inc in zip(people, inclusion[ckey]):
                presence[p] += q * inc
    for entry in signatures.values():
        entry["components"] = dict(entry["components"])
    # Probabilité (log10) que les décisions indépendantes respectent les plafonds.
    log10_admissible = (top + math.log(norm) + log_static) / math.log(10)
    return Law(signatures, omegas, dropped_total, log10_admissible, dict(caps), total_cap,
               dict(presence))


# --------------------------------------------------------------------------- soutien


def poll_medians(polls, results, election: int) -> dict[str, float]:
    """Médiane des parts (points) de chaque candidat, premier tour, toutes hypothèses."""
    import polars as pl

    ids = polls.filter(pl.col("election") == election)["poll_id"].implode()
    df = results.filter((pl.col("round") == 1) & pl.col("poll_id").is_in(ids))
    df = df.filter(pl.col("share").is_not_null())
    return dict(df.group_by("candidate_id").agg(pl.col("share").median()).iter_rows())


def prior_centre(
    person: str, current: dict[str, float], previous: dict[str, float], default: float
) -> tuple[float, str]:
    """Centre (fraction) du prior de soutien et son origine."""
    if person in current:
        return max(current[person], 1e-3) / 100, "cycle"
    if person in previous:
        return max(previous[person], 1e-3) / 100, "previous_cycle"
    return default, "default"


def large_candidates(draws, active_fitted: list[str], threshold: float) -> set[str]:
    """Gros candidats : part médiane a posteriori ≥ seuil parmi les actifs à soutien estimé."""
    from vvs.model.run import draws_log_shares

    if not active_fitted:
        return set()
    shares = np.exp(draws_log_shares(draws, active_fitted))
    med = np.median(shares, axis=0)
    return {c for c, m in zip(active_fitted, med) if m >= threshold}


# --------------------------------------------------------------------------- assemblage


def allocate(probabilities, n: int) -> np.ndarray:
    """Effectifs entiers de somme n, plus forts restes (copie de `simulate.allocate` du gel)."""
    p = np.asarray(probabilities, dtype=float)
    p = p / p.sum()
    raw = p * n
    k = np.floor(raw).astype(int)
    rest = n - k.sum()
    if rest > 0:
        k[np.argsort(-(raw - k), kind="stable")[:rest]] += 1
    return k


def assemble(
    law: Law, large: set[str], settings: Settings, n_draws: int, rng
) -> tuple[list[dict], dict]:
    """Signatures retenues (≤ plafond, couverture visée), puis strates du reste par état de
    présence des gros candidats ; tirages répartis par plus forts restes, puis tirés dans
    chaque strate proportionnellement aux masses (amendement 1).

    Chaque configuration rendue porte `count` tirages et `probability` = count / n_draws.
    """
    ordered = sorted(law.signatures.items(), key=lambda kv: (-kv[1]["mass"], kv[0]))
    top, covered = [], 0.0
    for key, entry in ordered:
        if covered >= settings.coverage or len(top) >= settings.max_configurations:
            break
        top.append((key, entry))
        covered += entry["mass"]
    strata: dict[tuple, list] = defaultdict(list)
    for key, entry in ordered[len(top):]:
        strata[tuple(sorted(set(key[0]) & large))].append((key, entry))
    patterns = sorted(strata)
    masses = [e["mass"] for _, e in top] + [
        math.fsum(e["mass"] for _, e in strata[p]) for p in patterns
    ]
    counts = allocate(masses, n_draws)
    configs = []

    def emit(key, entry, count, kind, pattern=None):
        configs.append(
            {"fitted": list(key[0]), "slots": dict(key[1]), "count": int(count),
             "probability": count / n_draws, "mass": entry["mass"], "kind": kind,
             "stratum": None if pattern is None else list(pattern),
             "components": sorted([w, v] for w, v in entry["components"].items())}
        )

    for (key, entry), count in zip(top, counts[: len(top)]):
        if count:
            emit(key, entry, count, "top")
    sampled_signatures = 0
    for pattern, count in zip(patterns, counts[len(top):]):
        if not count:
            continue
        members = strata[pattern]
        w = np.array([e["mass"] for _, e in members])
        draws = rng.choice(len(members), size=int(count), p=w / w.sum())
        idx, k = np.unique(draws, return_counts=True)
        sampled_signatures += len(idx)
        for i, c in zip(idx, k):
            emit(*members[i], c, "sampled", pattern)
    if sum(c["count"] for c in configs) != n_draws:
        raise ValueError("effectifs incohérents")
    info = {
        "n_signatures": len(law.signatures),
        "n_top": len(top),
        "top_coverage": covered,
        "n_strata": len(patterns),
        "stratified_mass": 1 - covered,
        "n_sampled_signatures": sampled_signatures,
        "n_configurations": len(configs),
    }
    return configs, info


def presence_from(configs: list[dict], dec: Decisions, fitted: set[str], ptype: dict,
                  omegas: list[dict]) -> dict[str, float]:
    """Présence exacte de chaque personne active sous une liste de signatures."""
    types = _type_members(dec, fitted, ptype)
    forced_of = [
        frozenset(c.person for c in dec.commitments if om[c.process] in c.winners)
        for om in omegas
    ]
    member_type = {m: (None if m in fitted else ptype[m]) for p in dec.pools for m in p.members}
    out: dict[str, float] = defaultdict(float)
    cache: dict = {}
    for cfg in configs:
        total = math.fsum(w for _, w in cfg["components"])
        scale = cfg["probability"] / total if "probability" in cfg else 1.0
        for wi, w in cfg["components"]:
            w *= scale
            om = omegas[wi]
            for f in cfg["fitted"]:
                if f not in om.values():
                    out[f] += w
            for m in om.values():
                out[m] += w
            for t, total in cfg["slots"].items():
                k = total - sum(1 for m in om.values() if member_type.get(m) == t)
                persons = [(p, q) for p, q in types.get(t, []) if p not in forced_of[wi]]
                key = (t, k, forced_of[wi])
                if key not in cache:
                    cache[key] = conditional_inclusion([q for _, q in persons], k)
                for (p, _), inc in zip(persons, cache[key]):
                    out[p] += w * inc
    return dict(out)


def _type_members(dec: Decisions, fitted: set[str], ptype: dict) -> dict[str, list]:
    types: dict[str, list] = defaultdict(list)
    for p, q in dec.independent.items():
        if p not in fitted and q > 0:
            types[ptype[p]].append((p, q))
    return types


def build_case(
    cr,
    registry,
    election: int,
    horizon: int,
    asof: dt.date,
    settings: Settings,
    draws,
    current_medians: dict[str, float],
    previous_medians: dict[str, float],
    n_draws: int,
    seed: int,
) -> dict:
    """Configurations et audit d'un cas avant liste officielle (sérialisable en JSON).

    Ne lit ni le statut final des personnes, ni la liste officielle, ni les champs futurs
    des processus : seulement `state_at`, les entrées publiées, les engagements publiés,
    les tirages figés (noms, soutien) et les sondages disponibles.
    """
    dec = decision_units(cr, asof, settings)
    fit_names = {str(c) for c in draws["candidates"]}
    active = sorted(dec.blocs)
    fitted = {p for p in active if p in fit_names}
    centres, ptype, types = {}, {}, {}
    for p in active:
        if p in fitted:
            continue
        c, source = prior_centre(p, current_medians, previous_medians, settings.prior_default)
        m14 = current_medians.get(p, 1.0)
        key = (dec.blocs[p], round(c, 8), round(m14, 8))
        if key not in types:
            types[key] = f"t{len(types)}"
        ptype[p] = types[key]
        centres[p] = (c, source, m14)
    caps, total_cap = bloc_caps(registry, election)
    law = enumerate_law(dec, fitted, ptype, caps, total_cap, settings.prune)
    large = large_candidates(draws, sorted(fitted), settings.large_share)
    configs, info = assemble(law, large, settings, n_draws, np.random.default_rng([seed, 3]))
    exact = law.presence
    represented = presence_from(configs, dec, fitted, ptype, law.omegas)
    type_table = {}
    members = _type_members(dec, fitted, ptype)
    for (bloc, c, m14), tid in types.items():
        type_table[tid] = {
            "bloc": bloc,
            "centre": c,
            "centre_mean14d": m14,
            "persons": [[p, q] for p, q in members.get(tid, [])],
        }
    unit = {p: "independent" for p in dec.independent}
    for pool in dec.pools:
        for m in pool.members:
            unit[m] = pool.id
    persons = []
    for p in sorted(dec.states):
        s = dec.states[p]
        is_active = p in dec.blocs
        row = {
            "person": p,
            "state": s.state,
            "since": str(s.since),
            "reason": s.reason,
            "returned": s.returned,
            "active": is_active,
            "bloc": dec.blocs.get(p),
            "party": dec.parties.get(p),
            "unit": unit.get(p),
            "p_decision": dec.independent.get(p) if unit.get(p) == "independent" else (
                1 / len(next(x for x in dec.pools if x.id == unit[p]).members) if is_active else 0.0
            ),
            "estimated_support": p in fitted,
            "large": p in large,
            "prior_centre": centres.get(p, (None,))[0],
            "prior_source": centres.get(p, (None, None))[1],
            "type": ptype.get(p),
            "presence_exact": exact.get(p, 0.0),
            "presence_represented": represented.get(p, 0.0),
        }
        persons.append(row)
    states_mass = defaultdict(lambda: {"n": 0, "mass_exact": 0.0, "mass_represented": 0.0})
    for row in persons:
        if row["active"]:
            key = row["state"] if row["unit"] == "independent" else f"{row['state']}+process"
            states_mass[key]["n"] += 1
            states_mass[key]["mass_exact"] += row["presence_exact"]
            states_mass[key]["mass_represented"] += row["presence_represented"]
    audit = {
        "caps": caps,
        "total_cap": total_cap,
        "pools": [asdict(p) for p in dec.pools],
        "commitments": [
            {"person": c.person, "process": c.process, "winners": sorted(c.winners)}
            for c in dec.commitments
        ],
        "pruned_mass_bound": law.pruned_bound,
        "log10_constraint_probability": law.admissible,
        "large_candidates": sorted(large),
        "polled_not_active": sorted(fit_names - set(active)),
        "without_estimated_support": sorted(p for p in active if p not in fitted),
        "mass_by_state": dict(states_mass),
        "max_presence_shift_from_sampling": max(
            (abs(represented.get(p, 0) - exact.get(p, 0)) for p in active), default=0.0
        ),
        "expected_candidates": math.fsum(exact.values()),
    } | info
    return {
        "election": election,
        "horizon": horizon,
        "asof": str(asof),
        "source": "decisions",
        "settings": asdict(settings),
        "configurations": configs,
        "omegas": law.omegas,
        "pools": {p.id: list(p.members) for p in dec.pools},
        "commitments": audit["commitments"],
        "types": type_table,
        "member_type": {
            m: (None if m in fitted else ptype[m]) for p in dec.pools for m in p.members
        },
        "persons": persons,
        "audit": audit,
    }


# --------------------------------------------------------------------------- simulation


def slot_name(election: int, horizon: int, variant: str, type_id: str, j: int) -> str:
    return f"{SLOT_PREFIX}-{election}-{horizon:03d}-{variant}-{type_id}-{j}"


def simulation_configurations(decision: dict, variant: str) -> tuple[list[dict], dict]:
    """Configurations à simuler (noms estimés + pseudo-noms de type) et table des priors."""
    e, h = decision["election"], decision["horizon"]
    configs, priors = [], {}
    for cfg in decision["configurations"]:
        names = list(cfg["fitted"])
        for t, k in sorted(cfg["slots"].items()):
            for j in range(k):
                name = slot_name(e, h, variant, t, j)
                info = decision["types"][t]
                priors[name] = {
                    "type": t,
                    "bloc": info["bloc"],
                    "centre": info["centre"],
                    "centre_mean14d": info["centre_mean14d"],
                }
                names.append(name)
        configs.append({"candidates": sorted(names), "probability": cfg["probability"]})
    return configs, priors


def augment_registry(registry, election: int, priors: dict) -> None:
    """Ajoute en mémoire les pseudo-noms de type au registre (bloc seulement)."""
    from vvs.registry.models import Affiliation, BoundaryCase, Candidate

    for name, info in priors.items():
        if name in registry.candidates:
            continue
        registry.candidates[name] = Candidate(
            name=name, first_name="type", last_name=name, parties=[Affiliation(party="div")]
        )
        registry.boundary_cases.append(
            BoundaryCase(
                id=f"decision-{name}",
                candidate_ids=[name],
                elections=[election],
                bloc=info["bloc"],
                rationale="pseudo-candidat du lot 2 (candidat sans soutien estimé)",
            )
        )


def prior_shares(centres, n: int, settings: Settings, rng) -> np.ndarray:
    """Parts (fractions, n × K) des candidats sans soutien estimé : log-normale déclarée."""
    z = rng.normal(np.log(centres), settings.prior_sigma, (n, len(centres)))
    s = np.clip(np.exp(z), settings.prior_low, settings.prior_high)
    total = s.sum(axis=1, keepdims=True)
    return np.where(total > settings.prior_total_max, s * settings.prior_total_max / total, s)


def mix_log_shares(cands, fitted, x_fitted, slots, s) -> np.ndarray:
    """Log-parts de la configuration : priors s, estimés × (1 − somme des priors)."""
    x = np.zeros((s.shape[0], len(cands)))
    if fitted:
        x[:, [cands.index(c) for c in fitted]] = x_fitted + np.log1p(-s.sum(axis=1, keepdims=True))
    x[:, [cands.index(c) for c in slots]] = np.log(s)
    return x


def simulate_chain(case, draws, error_model, model2, turnout, spec, polls, method="reference",
                   priors=None, settings: Settings | None = None):
    """`consolidation.simulate_chain` du gel, avec des candidats à prior de soutien.

    Sans prior (`priors` vide), le code et la consommation du générateur sont ceux du gel :
    même résultat au bit près (contrôlé au rejeu). Avec priors : log-parts des candidats à
    soutien estimé multipliées par (1 − somme des priors), tirés par un générateur séparé.
    """
    import polars as pl

    from vvs.eval.metrics import top2
    from vvs.model.run import draws_log_shares
    from vvs.model.simulate import allocate, sample_turnout
    from vvs.registry import default_registry

    priors = priors or {}
    settings = settings or Settings()
    reg = default_registry()
    rng = np.random.default_rng(spec["seed"])
    rng_prior = np.random.default_rng([spec["seed"], 1])
    n = spec["n_draws"]
    configs = case["configurations"]
    candidates = sorted({c for cfg in configs for c in cfg["candidates"]})
    index = {c: i for i, c in enumerate(candidates)}
    shares = np.zeros((n, len(candidates)))
    config_ids = np.repeat(
        np.arange(len(configs)), allocate(np.array([cfg["probability"] for cfg in configs]), n)
    )
    for k, cfg in enumerate(configs):
        rows = np.flatnonzero(config_ids == k)
        if not len(rows):
            continue
        cands = cfg["candidates"]
        slots = [c for c in cands if c in priors]
        if method == "reference":
            if not slots:
                logp = draws_log_shares(draws, cands)
                x = logp[rng.integers(0, len(logp), len(rows))]
            else:
                fitted = [c for c in cands if c not in priors]
                s = prior_shares([priors[c]["centre"] for c in slots], len(rows), settings,
                                 rng_prior)
                xf = None
                if fitted:
                    logp = draws_log_shares(draws, fitted)
                    xf = logp[rng.integers(0, len(logp), len(rows))]
                x = mix_log_shares(cands, fitted, xf, slots, s)
        else:
            recent = polls.filter(
                pl.col("fieldwork_end")
                > dt.date.fromisoformat(case["asof"]) - dt.timedelta(days=14)
            )
            if recent.is_empty():
                recent = polls.filter(pl.col("fieldwork_end") == polls["fieldwork_end"].max())
            means = recent.group_by("poll_id", "candidate_id").agg(pl.col("share").mean())
            means = dict(means.group_by("candidate_id").agg(pl.col("share").mean()).iter_rows())
            older = dict(polls.group_by("candidate_id").agg(pl.col("share").median()).iter_rows())
            p = np.array([
                max(priors[c]["centre_mean14d"], 0.01) if c in priors
                else max(means.get(c, older.get(c, 1.0)), 0.01)
                for c in cands
            ])
            x = np.broadcast_to(np.log(p / p.sum()), (len(rows), len(cands))).copy()
        p = np.exp(x)
        eta = error_model.sample_eta(
            case["horizon"], [reg.bloc_of(c, case["election"]) for c in cands], p, rng
        )
        x = x + eta
        p = np.exp(x - x.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        shares[np.ix_(rows, [index[c] for c in cands])] = p * 100
    qualified = top2(shares)
    pair_codes, pair_idx = np.unique(qualified, axis=0, return_inverse=True)
    pairs = [(candidates[a], candidates[b]) for a, b in pair_codes]
    model2.fit(case["election"], case["asof"], pairs)
    tau = sample_turnout(turnout, rng, n)
    winner = np.full(n, -1, dtype=int)
    share2 = np.zeros(n)
    details = []
    for k in np.unique(config_ids):
        for code in np.unique(pair_idx[config_ids == k]):
            rows = np.flatnonzero((config_ids == k) & (pair_idx == code))
            pair = pairs[code]
            fr = {c: shares[rows, index[c]] / 100 for c in configs[k]["candidates"]}
            fr["abstain"] = (1 - tau[rows]) / tau[rows]
            second = model2.sample(pair, fr, case["horizon"], rng, n=len(rows))
            share2[rows] = second.share
            winner[rows] = np.where(second.share > 50, pair_codes[code, 0], pair_codes[code, 1])
            details.append(
                {
                    "config": int(k),
                    "candidate_a": pair[0],
                    "candidate_b": pair[1],
                    "n": len(rows),
                    "ess_ratio": second.ess / len(rows),
                    "n_duel_polls": model2.pair_fit(pair).polls.n,
                }
            )
    return {
        "candidates": np.array(candidates),
        "shares": shares,
        "qualified": qualified,
        "winner": winner,
        "share_second": share2,
        "config": config_ids,
    }, details


def sample_conditional(persons: list[tuple[str, float]], k: int, n: int, rng) -> np.ndarray:
    """n tirages de k noms parmi des Bernoulli indépendantes conditionnées à k succès."""
    out = np.empty((n, k), dtype=object)
    if k == 0:
        return out
    ps = [q for _, q in persons]
    table = tail_table(ps)
    if table[0, k] <= 0:
        raise ValueError("effectif impossible pour ce type")
    r = np.full(n, k)
    for i, (name, q) in enumerate(persons):
        num = q * table[i + 1, np.maximum(r - 1, 0)]
        den = table[i, r]
        with np.errstate(divide="ignore", invalid="ignore"):
            prob = np.where((r > 0) & (den > 0), num / den, 0.0)
        take = rng.random(n) < prob
        rows = np.flatnonzero(take)
        out[rows, k - r[rows]] = name
        r = r - take
    if (r != 0).any():
        raise ValueError("tirage conditionnel incomplet")
    return out


def assign_names(decision: dict, variant: str, sim: dict, sim_configs: list[dict], rng) -> dict:
    """Renomme les pseudo-noms tirage par tirage (loi conditionnelle exacte de la signature).

    Rend un dictionnaire de tirages au format du gel (noms réels) dont `config` renvoie aux
    configurations développées (ensembles de noms réalisés), dans `expanded`.
    """
    e, h = decision["election"], decision["horizon"]
    n = len(sim["winner"])
    expected = [c["count"] for c in decision["configurations"]]
    if np.bincount(sim["config"], minlength=len(expected)).tolist() != expected:
        raise ValueError("répartition des tirages différente de la construction")
    omegas = decision["omegas"]
    member_type = decision["member_type"]
    commitments = decision["commitments"]
    old = [str(c) for c in sim["candidates"]]
    label = np.empty((n, len(old)), dtype=object)  # nom réel par tirage et colonne
    for j, c in enumerate(old):
        label[:, j] = c
    for k, cfg in enumerate(decision["configurations"]):
        rows = np.flatnonzero(sim["config"] == k)
        if not len(rows):
            continue
        comps = cfg["components"]
        w = np.array([c[1] for c in comps], dtype=float)
        choice = rng.choice(len(comps), size=len(rows), p=w / w.sum())
        for ci in np.unique(choice):
            sub = rows[choice == ci]
            om = omegas[comps[ci][0]]
            forced = {c["person"] for c in commitments if om[c["process"]] in c["winners"]}
            for t, total in sorted(cfg["slots"].items()):
                nominees = sorted(m for m in om.values() if member_type.get(m) == t)
                persons = [(p, q) for p, q in decision["types"][t]["persons"]
                           if p not in forced]
                drawn = sample_conditional(persons, total - len(nominees), len(sub), rng)
                for j in range(total):
                    col = old.index(slot_name(e, h, variant, t, j))
                    label[sub, col] = nominees[j] if j < len(nominees) else drawn[:, j - len(nominees)]
    present = sim["shares"] > 0
    names = sorted({str(x) for x in label[present]} | {c for c in old if not c.startswith(SLOT_PREFIX)})
    if any(x.startswith(SLOT_PREFIX) for x in names):
        raise ValueError("pseudo-nom non renommé")
    pos = {c: i for i, c in enumerate(names)}
    shares = np.zeros((n, len(names)))
    rows_all = np.arange(n)
    for j in range(len(old)):
        mask = present[:, j]
        if old[j].startswith(SLOT_PREFIX):
            cols = np.array([pos[str(x)] for x in label[mask, j]], dtype=int)
            shares[rows_all[mask], cols] += sim["shares"][mask, j]
        else:
            shares[mask, pos[old[j]]] += sim["shares"][mask, j]
    if not np.allclose(shares.sum(axis=1), sim["shares"].sum(axis=1), atol=1e-10, rtol=0):
        raise ValueError("deux pseudo-noms renommés vers la même personne")
    if ((shares > 0).sum(axis=1) != present.sum(axis=1)).any():
        raise ValueError("collision de noms dans un tirage")

    def remap(idx):
        return np.array([pos[str(label[i, j])] for i, j in enumerate(idx)], dtype=int)

    qualified = np.column_stack([remap(sim["qualified"][:, 0]), remap(sim["qualified"][:, 1])])
    winner = remap(sim["winner"])
    sets: dict[tuple, int] = {}
    config = np.empty(n, dtype=int)
    for i in range(n):
        key = tuple(np.flatnonzero(shares[i] > 0))
        config[i] = sets.setdefault(key, len(sets))
    counts = np.bincount(config, minlength=len(sets))
    expanded = [
        {"candidates": [names[j] for j in key], "probability": counts[i] / n}
        for key, i in sets.items()
    ]
    return {
        "candidates": np.array(names),
        "shares": shares,
        "qualified": qualified,
        "winner": winner,
        "share_second": sim["share_second"],
        "config": config,
        "signature": sim["config"],
    }, expanded


def presence(configurations: list[dict]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for cfg in configurations:
        for c in cfg["candidates"]:
            out[c] += cfg["probability"]
    return dict(sorted(out.items()))
