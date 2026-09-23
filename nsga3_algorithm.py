"""NSGA-III (Deb & Jain, 2014) plus ein wortgleich aus nsga2-demo/nsga2_algorithm.py kopierter NSGA-II-Kern - beide laufen
in dieser Demo auf demselben 4-Ziele-Vehikel, für den direkten Mechanismus-Vergleich (referenzpunktbasierte Nischenbildung
gegen Crowding-Distance).

NSGA-III ersetzt NSGA-IIs Crowding-Distance-Diversität durch: (1) eine Menge fester Referenzpunkte auf dem
Zielraum-Simplex (Das-Dennis-Verfahren), (2) Normalisierung der Population (Idealpunkt + Extrempunkte über eine
Achievement-Scalarizing-Function + Hyperebenen-Achsenabschnitte), (3) Zuordnung jedes Individuums zum nächsten
Referenzpunkt, (4) Nischenbildung: beim Auffüllen der letzten, nicht vollständig passenden Front werden
unterbesetzte Referenzpunkte bevorzugt. Mating-Selektion ist ein reines Rang-Turnier (kein Crowding/Niching dabei,
wie im Originalpapier)."""

import warnings
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


# ===============================================================================================================================
# Kopiert aus nsga2-demo/nsga2_algorithm.py (Distanz/Kantenkosten, Operatoren, nicht-dominierte Sortierung, NSGA-II-Kern) -
# NSGA-II läuft hier als direkte Vergleichsbasis zu NSGA-III auf demselben Vehikel.
# ===============================================================================================================================


def dist_matrix(xy):
    xy = np.asarray(xy, dtype=float)
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


def tour_length_batch(pop, D):
    nxt = np.roll(pop, -1, axis=1)
    return D[pop, nxt].sum(axis=1)


def tour_edge_cost_batch(pop, D, factor_matrix):
    nxt = np.roll(pop, -1, axis=1)
    return (D[pop, nxt] * factor_matrix[pop, nxt]).sum(axis=1)


def tour_edges(tour):
    t = np.asarray(tour)
    a, b = t, np.roll(t, -1)
    return {(int(min(x, y)), int(max(x, y))) for x, y in zip(a, b)}


def edge_share(tour, reference):
    return len(tour_edges(tour) & tour_edges(reference)) / len(tour)


def init_population(pop_size, n_nodes, rng):
    return np.array([rng.permutation(n_nodes) for _ in range(pop_size)], dtype=np.int64)


def order_crossover(p1, p2, rng):
    n = len(p1)
    i, j = sorted(rng.integers(0, n, size=2))
    child = -np.ones(n, dtype=np.int64)
    child[i:j + 1] = p1[i:j + 1]
    taken = set(child[i:j + 1].tolist())
    fill = [g for g in p2.tolist() if g not in taken]
    pos = [k for k in range(n) if not (i <= k <= j)]
    for k, g in zip(pos, fill):
        child[k] = g
    return child


def swap_mutation(ind, p_mut, rng):
    if rng.random() >= p_mut:
        return ind.copy()
    out = ind.copy()
    i, j = rng.integers(0, len(ind), size=2)
    out[i], out[j] = out[j], out[i]
    return out


def dominates(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def dominance_matrix(objectives):
    F = np.asarray(objectives, dtype=float)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt
    np.fill_diagonal(dom, False)
    return dom


def non_dominated_mask(objectives):
    """Boolmaske der nicht-dominierten Punkte ohne volle (N, N)-Matrix - O(N * F) statt O(N²), siehe nsga2-demo."""
    F = np.asarray(objectives, dtype=float)
    order = np.argsort(F[:, 0])
    frontier_idx = []
    frontier_obj = np.empty((0, F.shape[1]))
    for idx in order:
        p = F[idx]
        if len(frontier_obj) and np.any(np.all(frontier_obj <= p, axis=1) & np.any(frontier_obj < p, axis=1)):
            continue
        if len(frontier_obj):
            dominated_by_p = np.all(p <= frontier_obj, axis=1) & np.any(p < frontier_obj, axis=1)
            keep = ~dominated_by_p
            frontier_idx = [fi for fi, k in zip(frontier_idx, keep.tolist()) if k]
            frontier_obj = frontier_obj[keep]
        frontier_idx.append(int(idx))
        frontier_obj = np.vstack([frontier_obj, p])
    mask = np.zeros(len(F), dtype=bool)
    mask[frontier_idx] = True
    return mask


def fast_non_dominated_sort(objectives):
    n = len(objectives)
    dom = dominance_matrix(objectives)
    domination_count = dom.sum(axis=0).astype(np.int64)
    rank = np.full(n, -1, dtype=np.int64)
    current = np.where(domination_count == 0)[0]
    rank[current] = 0
    fronts = [current]
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            dominated_by_p = np.where(dom[p])[0]
            domination_count[dominated_by_p] -= 1
            newly_free = dominated_by_p[domination_count[dominated_by_p] == 0]
            rank[newly_free] = i + 1
            next_front.extend(newly_free.tolist())
        i += 1
        fronts.append(np.array(next_front, dtype=np.int64))
    fronts.pop()
    return fronts, rank


def crowding_distance(front_objectives):
    m, n_obj = front_objectives.shape
    if m <= 2:
        return np.full(m, np.inf)
    dist = np.zeros(m)
    for k in range(n_obj):
        order = np.argsort(front_objectives[:, k])
        vals = front_objectives[order, k]
        spread = vals[-1] - vals[0]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        if spread < EPS:
            continue
        dist[order[1:-1]] += (vals[2:] - vals[:-2]) / spread
    return dist / n_obj


def crowding_distance_all(objectives, fronts):
    n = len(objectives)
    dist = np.zeros(n)
    for front in fronts:
        if len(front) == 0:
            continue
        dist[front] = crowding_distance(objectives[front])
    return dist


def crowded_compare(rank_a, cd_a, rank_b, cd_b):
    if rank_a != rank_b:
        return rank_a < rank_b
    return cd_a > cd_b


def crowded_tournament_select(rank, crowding, k, n_select, rng):
    n = len(rank)
    out = np.empty(n_select, dtype=np.int64)
    for s in range(n_select):
        cand = rng.integers(0, n, size=k)
        best = cand[0]
        for c in cand[1:]:
            if crowded_compare(rank[c], crowding[c], rank[best], crowding[best]):
                best = c
        out[s] = best
    return out


def select_survivors_nsga2(objectives, pop_size):
    fronts, rank_all = fast_non_dominated_sort(objectives)
    survivors = []
    crowding_out = np.zeros(len(objectives))
    for front in fronts:
        if len(survivors) + len(front) <= pop_size:
            if len(front) > 0:
                cd = crowding_distance(objectives[front])
                crowding_out[front] = cd
            survivors.extend(front.tolist())
            if len(survivors) == pop_size:
                break
        else:
            remaining = pop_size - len(survivors)
            cd = crowding_distance(objectives[front])
            order = np.argsort(-cd)
            chosen = front[order[:remaining]]
            crowding_out[chosen] = cd[order[:remaining]]
            survivors.extend(chosen.tolist())
            break
    idx = np.array(survivors, dtype=np.int64)
    return idx, rank_all[idx], crowding_out[idx]


@dataclass
class Generation:
    population: np.ndarray
    objectives: np.ndarray
    rank: np.ndarray


@dataclass
class NSGA2Result:
    final_population: np.ndarray
    final_objectives: np.ndarray
    final_rank: np.ndarray
    front1_history: list = field(default_factory=list)
    generations: list = field(default_factory=list)

    @property
    def front1(self):
        return self.final_objectives[self.final_rank == 0]

    @property
    def front1_population(self):
        return self.final_population[self.final_rank == 0]


def run_nsga2(objective_fn, n_nodes, pop_size, generations, cx_prob, mut_prob, tournament_k, seed, keep_history=False):
    """objective_fn(population) -> (pop_size, n_obj), niedriger ist besser je Ziel. Vergleichsbasis für NSGA-III (unten)."""
    rng = np.random.default_rng(seed)
    pop = init_population(pop_size, n_nodes, rng)
    obj = objective_fn(pop)
    fronts, rank = fast_non_dominated_sort(obj)
    crowding = crowding_distance_all(obj, fronts)
    front1_history = [obj[rank == 0].copy()]
    gens = [Generation(pop.copy(), obj.copy(), rank.copy())] if keep_history else []

    for _ in range(generations):
        parents = crowded_tournament_select(rank, crowding, tournament_k, pop_size, rng)
        children = []
        for i in range(pop_size):
            p1 = pop[parents[i]]
            p2 = pop[parents[rng.integers(0, pop_size)]]
            child = order_crossover(p1, p2, rng) if rng.random() < cx_prob else p1.copy()
            child = swap_mutation(child, mut_prob, rng)
            children.append(child)
        offspring = np.array(children, dtype=np.int64)
        obj_offspring = objective_fn(offspring)
        combined_pop = np.concatenate([pop, offspring], axis=0)
        combined_obj = np.concatenate([obj, obj_offspring], axis=0)
        idx, rank, crowding = select_survivors_nsga2(combined_obj, pop_size)
        pop = combined_pop[idx]
        obj = combined_obj[idx]
        front1_history.append(obj[rank == 0].copy())
        if keep_history:
            gens.append(Generation(pop.copy(), obj.copy(), rank.copy()))

    return NSGA2Result(pop, obj, rank, front1_history, gens)


# ===============================================================================================================================
# NSGA-III (Deb & Jain, 2014) - neu
# ===============================================================================================================================


def generate_reference_points(n_obj, divisions):
    """Das-Dennis-Verfahren: Punkte auf dem (n_obj-1)-dimensionalen Einheitssimplex (Koordinatensumme 1), `divisions`
    Unterteilungen je Achse. Punktzahl = C(divisions + n_obj - 1, n_obj - 1)."""
    def recurse(n_left, remaining):
        if n_left == 1:
            yield (remaining,)
            return
        for i in range(remaining + 1):
            for rest in recurse(n_left - 1, remaining - i):
                yield (i,) + rest
    return np.array(list(recurse(n_obj, divisions)), dtype=float) / divisions


def ideal_point(objectives):
    return objectives.min(axis=0)


def extreme_points(translated, n_obj):
    """`translated` = objectives - ideal_point (>= 0). Extrempunkt je Achse j über die Achievement-Scalarizing-Function:
    das Individuum, das max_k(f_k / w_k) minimiert, mit w nahe dem Einheitsvektor auf Achse j (kleines Epsilon sonst)."""
    epsilon = 1e-6
    points = np.zeros((n_obj, n_obj))
    for j in range(n_obj):
        w = np.full(n_obj, epsilon)
        w[j] = 1.0
        asf = (translated / w).max(axis=1)
        points[j] = translated[np.argmin(asf)]
    return points


def intercepts(extreme_pts, translated, n_obj):
    """Achsenabschnitte der Hyperebene durch die Extrempunkte (sum(x_k / a_k) = 1). Rückfall auf den Nadir (Maximalwert je
    Achse) bei singulärem Gleichungssystem oder einem Achsenabschnitt <= 0 (Deb & Jains eigene Rückfalllogik aus dem
    Originalpapier 2014). Die pymoo-Bibliothek geht hier bewusst einen Schritt weiter (eigener Kommentar im pymoo-Quelltext:
    "NOTE: different to the proposed version in the paper") und deckelt den gelösten Achsenabschnitt zusätzlich am
    schlechtesten bisher beobachteten Wert der gesamten Population - deshalb weichen die beiden Implementierungen hier
    bewusst voneinander ab (Kreuzprobe unten prüft `ideal_point`/`extreme_points`/`associate` exakt gegen pymoo, `intercepts`
    nur per Handrechnung, siehe Tests)."""
    try:
        a = np.linalg.solve(extreme_pts, np.ones(n_obj))     # a_k = 1 / Achsenabschnitt_k
        with warnings.catch_warnings():                       # a_k nahe 0 -> erwartete Division, wird gleich abgefangen
            warnings.simplefilter("ignore", RuntimeWarning)
            intercept = 1.0 / a
        if np.any(intercept <= 1e-10) or np.any(~np.isfinite(intercept)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercept = translated.max(axis=0)
        intercept = np.where(intercept < 1e-10, 1.0, intercept)
    return intercept


def normalize_objectives(objectives, ideal, intercept):
    translated = objectives - ideal
    return translated / np.where(intercept < 1e-10, 1.0, intercept)


def associate(normalized_obj, ref_points):
    """Kürzester senkrechter Abstand jedes Individuums zu jeder Referenzlinie (Ursprung -> Referenzpunkt). Gibt
    (zugeordneter Referenzpunkt-Index, Abstand) je Individuum zurück."""
    ref_norm2 = (ref_points ** 2).sum(axis=1)
    proj_scale = (normalized_obj @ ref_points.T) / ref_norm2[None, :]
    proj = proj_scale[:, :, None] * ref_points[None, :, :]
    diff = normalized_obj[:, None, :] - proj
    dist = np.sqrt((diff ** 2).sum(axis=2))
    assoc = np.argmin(dist, axis=1)
    min_dist = dist[np.arange(len(normalized_obj)), assoc]
    return assoc, min_dist


def niching_select(fronts, ref_assoc, ref_dist, n_refs, pop_size, rng):
    """Fronten der Reihe nach auffüllen; die letzte, nicht vollständig passende Front wird über Referenzpunkt-Nischen
    aufgefüllt (Deb & Jain 2014, Algorithmus "Niching"): das Referenzpunkt mit der kleinsten bisherigen Besetzung (unter
    denen mit noch verfügbaren Kandidaten in der aktuellen Front) wird gewählt; bei Besetzung 0 der nächstgelegene
    Kandidat, sonst ein zufälliger."""
    survivors = []
    niche_count = np.zeros(n_refs, dtype=np.int64)
    front_idx = 0
    while front_idx < len(fronts) and len(survivors) + len(fronts[front_idx]) <= pop_size:
        front = fronts[front_idx]
        for idx in front.tolist():
            niche_count[ref_assoc[idx]] += 1
        survivors.extend(front.tolist())
        front_idx += 1
        if len(survivors) == pop_size:
            return np.array(survivors, dtype=np.int64)
    if front_idx >= len(fronts):
        return np.array(survivors, dtype=np.int64)
    last_front = list(fronts[front_idx].tolist())
    remaining = pop_size - len(survivors)
    for _ in range(remaining):
        candidate_refs = sorted({ref_assoc[i] for i in last_front})
        counts = niche_count[candidate_refs]
        min_count = counts.min()
        best_refs = [r for r, c in zip(candidate_refs, counts) if c == min_count]
        chosen_ref = best_refs[int(rng.integers(len(best_refs)))] if len(best_refs) > 1 else best_refs[0]
        candidates_for_ref = [i for i in last_front if ref_assoc[i] == chosen_ref]
        if niche_count[chosen_ref] == 0:
            dists = [ref_dist[i] for i in candidates_for_ref]
            pick = candidates_for_ref[int(np.argmin(dists))]
        else:
            pick = candidates_for_ref[int(rng.integers(len(candidates_for_ref)))]
        survivors.append(pick)
        last_front.remove(pick)
        niche_count[chosen_ref] += 1
    return np.array(survivors, dtype=np.int64)


def rank_tournament_select(rank, k, n_select, rng):
    """Turnierselektion nur nach Rang (kein Crowding/Niching bei der Paarung, wie im NSGA-III-Originalpapier)."""
    n = len(rank)
    out = np.empty(n_select, dtype=np.int64)
    for s in range(n_select):
        cand = rng.integers(0, n, size=k)
        out[s] = cand[int(np.argmin(rank[cand]))]
    return out


@dataclass
class NSGA3Result:
    final_population: np.ndarray
    final_objectives: np.ndarray
    final_rank: np.ndarray
    ref_points: np.ndarray
    front1_history: list = field(default_factory=list)
    generations: list = field(default_factory=list)

    @property
    def front1(self):
        return self.final_objectives[self.final_rank == 0]

    @property
    def front1_population(self):
        return self.final_population[self.final_rank == 0]


def run_nsga3(objective_fn, n_nodes, pop_size, generations, cx_prob, mut_prob, tournament_k, divisions, seed, keep_history=False):
    """objective_fn(population) -> (pop_size, n_obj), niedriger ist besser je Ziel. Normalisierung (Idealpunkt, Extrempunkte,
    Achsenabschnitte) wird jede Generation frisch aus der kombinierten Eltern+Nachkommen-Population berechnet (Standardpraxis,
    keine über Generationen laufend aktualisierte Fassung)."""
    rng = np.random.default_rng(seed)
    pop = init_population(pop_size, n_nodes, rng)
    obj = objective_fn(pop)
    n_obj = obj.shape[1]
    ref_points = generate_reference_points(n_obj, divisions)
    fronts, rank = fast_non_dominated_sort(obj)
    front1_history = [obj[rank == 0].copy()]
    gens = [Generation(pop.copy(), obj.copy(), rank.copy())] if keep_history else []

    for _ in range(generations):
        parents = rank_tournament_select(rank, tournament_k, pop_size, rng)
        children = []
        for i in range(pop_size):
            p1 = pop[parents[i]]
            p2 = pop[parents[rng.integers(0, pop_size)]]
            child = order_crossover(p1, p2, rng) if rng.random() < cx_prob else p1.copy()
            child = swap_mutation(child, mut_prob, rng)
            children.append(child)
        offspring = np.array(children, dtype=np.int64)
        obj_offspring = objective_fn(offspring)
        combined_pop = np.concatenate([pop, offspring], axis=0)
        combined_obj = np.concatenate([obj, obj_offspring], axis=0)

        fronts, rank_c = fast_non_dominated_sort(combined_obj)
        ideal = ideal_point(combined_obj)
        translated = combined_obj - ideal
        ext = extreme_points(translated, n_obj)
        interc = intercepts(ext, translated, n_obj)
        norm = normalize_objectives(combined_obj, ideal, interc)
        assoc, dist = associate(norm, ref_points)
        idx = niching_select(fronts, assoc, dist, len(ref_points), pop_size, rng)

        pop = combined_pop[idx]
        obj = combined_obj[idx]
        rank = rank_c[idx]
        front1_history.append(obj[rank == 0].copy())
        if keep_history:
            gens.append(Generation(pop.copy(), obj.copy(), rank.copy()))

    return NSGA3Result(pop, obj, rank, ref_points, front1_history, gens)
