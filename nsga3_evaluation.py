"""Auswertung der NSGA-III-Demo: ein Lauf (NSGA-II ODER NSGA-III), Sweep über Divisions/Populationsgröße, und zwei
Experimente - Referenzpunkt-Abdeckung (NSGA-II vs. NSGA-III bei 4 Zielen, der eigene Mechanismus-Vorteil) und Abdeckung der
Brute-Force-Pareto-Front auf der kleinen Vergleichsinstanz (schließt nsga2-demos eigenes Experiment ab)."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations

import numpy as np

import nsga3_algorithm as A
import nsga3_constants as C
import nsga3_scenario as S

BRUTE_FORCE_MAX_N = 9


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = 0
    seed: int = C.DEFAULT_SEED
    pop: int = C.DEFAULT_POP
    gens: int = C.DEFAULT_GEN
    cx: float = C.DEFAULT_CX
    mut: float = C.DEFAULT_MUT
    k: int = C.DEFAULT_K
    divisions: int = C.DEFAULT_DIV
    run_seed: int = C.DEFAULT_RUN_SEED


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate_perm(n, cluster_share, seed)
    return inst, A.dist_matrix(inst.xy)


def objective_fn(settings):
    """(Zielfunktion, Knotenzahl) für `settings`. Immer 4 Ziele: Distanz, CO2, Fahrzeit, Risiko."""
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)

    def fn(pop):
        dist = A.tour_length_batch(pop, D)
        co2 = A.tour_edge_cost_batch(pop, D, inst.co2_factor_matrix)
        time_ = A.tour_edge_cost_batch(pop, D, inst.time_factor_matrix)
        risk = A.tour_edge_cost_batch(pop, D, inst.risk_factor_matrix)
        return np.stack([dist, co2, time_, risk], axis=1)
    return fn, inst.n_nodes


def run3(settings, keep_history=False):
    fn, n_nodes = objective_fn(settings)
    return A.run_nsga3(fn, n_nodes, settings.pop, settings.gens, settings.cx, settings.mut, settings.k, settings.divisions, settings.run_seed, keep_history=keep_history)


def run2(settings, keep_history=False):
    fn, n_nodes = objective_fn(settings)
    return A.run_nsga2(fn, n_nodes, settings.pop, settings.gens, settings.cx, settings.mut, settings.k, settings.run_seed, keep_history=keep_history)


@dataclass
class Analysis:
    settings: Settings
    result: object


def analyse3(settings, keep_history=True):
    return Analysis(settings, run3(settings, keep_history=keep_history))


# --- Referenzpunkt-Abdeckung: algorithmusunabhängige Kennzahl (post-hoc auf JEDE Front anwendbar) -------------------------------------------


def reference_coverage(front_objectives, divisions):
    """Wie viele der Referenzrichtungen (Das-Dennis, `divisions`) haben mindestens ein zugeordnetes Individuum dieser Front -
    unabhängig davon, ob die Front von NSGA-II oder NSGA-III stammt. Front wird mit ihrer EIGENEN Normalisierung (Idealpunkt,
    Extrempunkte, Achsenabschnitte) auf den Referenzraum abgebildet."""
    n_obj = front_objectives.shape[1]
    ref_points = A.generate_reference_points(n_obj, divisions)
    ideal = A.ideal_point(front_objectives)
    translated = front_objectives - ideal
    if len(front_objectives) < n_obj:
        return 0, len(ref_points)          # Extrempunktbestimmung braucht mindestens n_obj Individuen
    ext = A.extreme_points(translated, n_obj)
    interc = A.intercepts(ext, translated, n_obj)
    norm = A.normalize_objectives(front_objectives, ideal, interc)
    assoc, _ = A.associate(norm, ref_points)
    return len(set(assoc.tolist())), len(ref_points)


def niche_occupancy(front_objectives, divisions):
    """Wie viele Individuen jedem Referenzpunkt zugeordnet sind - (n_ref,) Array. Leer (alle Nullen) zurückgegeben, wenn die
    Front zu klein für eine Extrempunktbestimmung ist (< n_obj Individuen)."""
    n_obj = front_objectives.shape[1]
    ref_points = A.generate_reference_points(n_obj, divisions)
    if len(front_objectives) < n_obj:
        return np.zeros(len(ref_points), dtype=np.int64)
    ideal = A.ideal_point(front_objectives)
    translated = front_objectives - ideal
    ext = A.extreme_points(translated, n_obj)
    interc = A.intercepts(ext, translated, n_obj)
    norm = A.normalize_objectives(front_objectives, ideal, interc)
    assoc, _ = A.associate(norm, ref_points)
    return np.bincount(assoc, minlength=len(ref_points))


def shared_reference_coverage(front_a, front_b, divisions):
    """Wie `reference_coverage`, aber beide Fronten teilen sich EINE Normalisierung (Idealpunkt/Extrempunkte/Achsenabschnitte
    aus der Vereinigung beider Fronten) - fairer Vergleich zweier unterschiedlich geformter Fronten auf derselben Skala,
    statt jede Front an ihrer eigenen (unterschiedlichen) Spannweite zu messen."""
    n_obj = front_a.shape[1]
    ref_points = A.generate_reference_points(n_obj, divisions)
    combined = np.vstack([front_a, front_b])
    ideal = A.ideal_point(combined)
    translated = combined - ideal
    ext = A.extreme_points(translated, n_obj)
    interc = A.intercepts(ext, translated, n_obj)

    def covered(front):
        norm = A.normalize_objectives(front, ideal, interc)
        assoc, _ = A.associate(norm, ref_points)
        return len(set(assoc.tolist()))
    return covered(front_a), covered(front_b), len(ref_points)


def saturation_share(front1_history, pop_size):
    """Anteil der Generationen, in denen Front 1 bereits die ganze Population einnimmt (alle Individuen gegenseitig nicht
    dominiert) - Diagnose dafür, wie oft NSGA-IIIs Nischenbildung (wirkt nur beim Abschneiden einer ZU GROSSEN Front) und
    NSGA-IIs Crowding-Abschneiden überhaupt etwas zu tun bekommen."""
    return float(np.mean([len(f) >= pop_size for f in front1_history]))


# --- Brute-Force-Referenz (kleine Instanzen) -----------------------------------------------------------------------------------------------


def brute_force_front(n_nodes, fn):
    """Alle (n_nodes - 1)! Touren. Gibt (alle Objektive, eindeutige nicht-dominierte Zielwerte) zurück - dedupliziert vor der
    Nicht-Dominanz-Prüfung (siehe nsga2-demo: zwei Touren können denselben Zielwert erreichen)."""
    tours = np.array([(0,) + p for p in permutations(range(1, n_nodes))], dtype=np.int64)
    obj = fn(tours)
    unique_obj = np.unique(obj, axis=0)
    nd_mask = A.non_dominated_mask(unique_obj)
    return obj, unique_obj[nd_mask]


def front_coverage(true_front_obj, found_obj, tol=1e-6):
    reached = 0
    for point in true_front_obj:
        if np.any(np.all(np.abs(found_obj - point) < tol, axis=1)):
            reached += 1
    return reached, len(true_front_obj)


# --- Experiment 1: Referenzpunkt-Abdeckung, NSGA-II vs. NSGA-III (eigener Mechanismus-Vorteil) ----------------------------------------------


def coverage_experiment(n=None, seed=None, pop=None, gens=None, divisions=None, seeds=None):
    """NSGA-II gegen NSGA-III bei denselben vier Zielen, faire (geteilte) Referenzpunkt-Abdeckung über mehrere Läufe, plus
    die Sättigungs-Diagnose (Anteil der Generationen, in denen Front 1 schon die ganze Population einnimmt) je Algorithmus."""
    n = C.DEFAULT_N if n is None else n
    seed = C.DEFAULT_SEED if seed is None else seed
    pop = C.COVERAGE_POP if pop is None else pop
    gens = C.COVERAGE_GENS if gens is None else gens
    divisions = C.DEFAULT_DIV if divisions is None else divisions
    seeds = C.COVERAGE_SEEDS if seeds is None else seeds

    cov2, cov3, sat2, sat3, total = [], [], [], [], None
    for run_seed in seeds:
        s = Settings(n=n, seed=seed, pop=pop, gens=gens, divisions=divisions, run_seed=run_seed)
        r2 = run2(s, keep_history=False)
        r3 = run3(s, keep_history=False)
        c2, c3, total = shared_reference_coverage(r2.front1, r3.front1, divisions)
        cov2.append(c2)
        cov3.append(c3)
        sat2.append(saturation_share(r2.front1_history, pop))
        sat3.append(saturation_share(r3.front1_history, pop))
    return {
        "total": total,
        "nsga2": {"covered_median": float(np.median(cov2)), "covered_all": cov2, "saturation_median": float(np.median(sat2))},
        "nsga3": {"covered_median": float(np.median(cov3)), "covered_all": cov3, "saturation_median": float(np.median(sat3))},
    }


# --- Experiment 2: Abdeckung der Brute-Force-Front, NSGA-II vs. NSGA-III (schließt nsga2-demos Cliffhanger) --------------------------------


def comparison_experiment(n=None, seed=None, pop=None, gens=None, seeds=None):
    n = C.COMPARISON_N if n is None else n
    seed = C.COMPARISON_VEHICLE_SEED if seed is None else seed
    pop = C.COMPARISON_POP if pop is None else pop
    gens = C.COMPARISON_GENS if gens is None else gens
    seeds = C.COMPARISON_SEEDS if seeds is None else seeds

    s0 = Settings(n=n, seed=seed, pop=pop, gens=gens)
    fn, n_nodes = objective_fn(s0)
    _, true_front = brute_force_front(n_nodes, fn)

    rows = {}
    for label, run_fn in (("nsga2", run2), ("nsga3", run3)):
        reached_list = []
        for run_seed in seeds:
            r = run_fn(replace(s0, run_seed=run_seed), keep_history=False)
            reached, _ = front_coverage(true_front, r.front1)
            reached_list.append(reached)
        rows[label] = {"reached_median": float(np.median(reached_list)), "reached_all": reached_list}
    return {"front_size": len(true_front), **rows}


# --- Sweep: Divisions/Populationsgröße vs. Referenzpunkt-Abdeckung (NSGA-III) ----------------------------------------------------------------


def run_config(param, value, base, seeds=None):
    seeds = C.SWEEP_SEEDS if seeds is None else seeds
    s0 = replace(base, **{param: value})
    covered_list, total = [], None
    for run_seed in seeds:
        r = run3(replace(s0, run_seed=run_seed), keep_history=False)
        covered, total = reference_coverage(r.front1, s0.divisions)
        covered_list.append(covered)
    return {"covered": float(np.mean(covered_list)), "total": total, "share": float(np.mean(covered_list)) / total}


def sweep(param, base=None, values=None):
    base = Settings() if base is None else base
    values = C.SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(param, v, base)} for v in values]
