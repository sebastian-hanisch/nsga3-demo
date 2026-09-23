"""Kern gegen Handrechnung + Bibliotheksgegenprobe (pymoo). Der kopierte NSGA-II-Kern wird knapper erneut geprüft (die volle
Kreuzprobe steht in nsga2-demo); der neue NSGA-III-Kern (Referenzpunkte, Normalisierung, Zuordnung, Nischenbildung) wird
vollständig geprüft - Referenzpunkte, Idealpunkt, Extrempunkte und Zuordnung stimmen exakt mit pymoo überein; die
Achsenabschnitte weichen bewusst von pymoos eigener, über das Originalpapier hinausgehender Deckelung ab (siehe
`nsga3_algorithm.intercepts`-Docstring) und werden deshalb nur per Handrechnung geprüft."""

from itertools import permutations

import numpy as np
import pytest
from pymoo.algorithms.moo.nsga3 import associate_to_niches
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance as pymoo_crowding_distance
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort as pymoo_sort
from pymoo.util.ref_dirs import get_reference_directions

import nsga3_algorithm as A

# --- Kopierter NSGA-II-Kern: knappe erneute Kreuzprobe (volle Prüfung steht in nsga2-demo) --------------------------------------------------

HAND_POINTS = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 1.0], [2.0, 5.0], [3.0, 3.0], [5.0, 2.0]])


def test_fast_non_dominated_sort_matches_hand_calculation():
    fronts, rank = A.fast_non_dominated_sort(HAND_POINTS)
    assert sorted(fronts[0].tolist()) == [0, 1, 2]
    assert sorted(fronts[1].tolist()) == [3, 4, 5]


@pytest.mark.parametrize("seed", range(15))
def test_fast_non_dominated_sort_matches_pymoo(seed):
    rng = np.random.default_rng(seed)
    n_obj = int(rng.integers(2, 5))
    F = rng.random((25, n_obj)) * 100
    fronts_mine, _ = A.fast_non_dominated_sort(F)
    fronts_pymoo = pymoo_sort(F)
    assert [sorted(f.tolist()) for f in fronts_mine] == [sorted(f) for f in fronts_pymoo]


@pytest.mark.parametrize("seed", range(10))
def test_crowding_distance_matches_pymoo(seed):
    rng = np.random.default_rng(seed)
    n_obj = int(rng.integers(2, 5))
    F = rng.random((25, n_obj)) * 100
    fronts, _ = A.fast_non_dominated_sort(F)
    for front in fronts:
        if len(front) < 3:
            continue
        np.testing.assert_allclose(np.sort(A.crowding_distance(F[front])), np.sort(pymoo_crowding_distance(F[front])), rtol=1e-9)


def test_order_crossover_matches_hand_calculation():
    class _FixedRNG:
        def integers(self, lo, hi, size=None):
            return np.array([3, 6])
    p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    p2 = np.array([5, 4, 6, 9, 2, 1, 7, 8, 3])
    assert A.order_crossover(p1, p2, _FixedRNG()).tolist() == [9, 2, 1, 4, 5, 6, 7, 8, 3]


def test_run_nsga2_smoke():
    def fn(pop):
        return ((pop[:, :2].astype(float) - 50.0) ** 2)
    r = A.run_nsga2(lambda pop: fn(pop), 10, pop_size=20, generations=15, cx_prob=0.9, mut_prob=0.2, tournament_k=2, seed=0)
    assert len(r.front1) >= 1


# --- NSGA-III: Referenzpunkte, Normalisierung, Zuordnung - exakte Kreuzprobe gegen pymoo -----------------------------------------------------


@pytest.mark.parametrize("n_obj,divisions", [(3, 4), (4, 5), (5, 3)])
def test_generate_reference_points_matches_pymoo(n_obj, divisions):
    mine = A.generate_reference_points(n_obj, divisions)
    pym = get_reference_directions("das-dennis", n_obj, n_partitions=divisions)
    assert mine.shape == pym.shape
    assert np.allclose(mine.sum(axis=1), 1.0)
    mine_set = {tuple(np.round(r, 6)) for r in mine}
    pym_set = {tuple(np.round(r, 6)) for r in pym}
    assert mine_set == pym_set


def test_generate_reference_points_count_formula():
    from math import comb
    for n_obj, divisions in [(3, 4), (4, 5), (2, 10)]:
        assert len(A.generate_reference_points(n_obj, divisions)) == comb(divisions + n_obj - 1, n_obj - 1)


@pytest.mark.parametrize("seed", range(10))
def test_ideal_and_extreme_points_match_pymoo(seed):
    from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization
    rng = np.random.default_rng(seed)
    n_obj = int(rng.integers(2, 5))
    F = rng.random((30, n_obj)) * 100

    ideal = A.ideal_point(F)
    translated = F - ideal
    ext = A.extreme_points(translated, n_obj)

    norm = HyperplaneNormalization(n_obj)
    dom = A.dominance_matrix(F)
    nds = np.where(~dom.any(axis=0))[0]
    norm.update(F, nds)

    np.testing.assert_allclose(ideal, norm.ideal_point)
    np.testing.assert_allclose(ext, norm.extreme_points - norm.ideal_point)


@pytest.mark.parametrize("seed", range(10))
def test_associate_matches_pymoo(seed):
    """Isoliert von der Normalisierung: dieselbe (bereits 'normalisierte') Matrix an beide geben, ideal=0/nadir=1 bei pymoo
    macht dessen interne Normalisierung zum No-op - reine Zuordnungslogik wird verglichen."""
    rng = np.random.default_rng(seed)
    n_obj = int(rng.integers(2, 5))
    norm = rng.random((25, n_obj))
    ref = A.generate_reference_points(n_obj, 5)

    assoc_mine, dist_mine = A.associate(norm, ref)
    niche_pymoo, dist_pymoo, _ = associate_to_niches(norm, ref, np.zeros(n_obj), np.ones(n_obj))

    assert np.array_equal(assoc_mine, niche_pymoo)
    np.testing.assert_allclose(dist_mine, dist_pymoo, rtol=1e-9)


# --- Achsenabschnitte: nur Handrechnung (siehe Docstring für die bewusste Abweichung von pymoo) ---------------------------------------------


def test_intercepts_matches_hand_calculation_diagonal_case():
    # Extrempunkte exakt auf den Achsen -> Achsenabschnitte sind direkt ablesbar
    ext = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 5.0]])
    interc = A.intercepts(ext, translated=np.array([[5.0, 5.0, 5.0]]), n_obj=3)
    assert interc == pytest.approx([10.0, 20.0, 5.0])


def test_intercepts_falls_back_to_nadir_on_singular_system():
    ext = np.array([[10.0, 0.0], [10.0, 0.0]])       # entartet: beide Extrempunkte identisch, singuläres System
    translated = np.array([[3.0, 7.0], [10.0, 2.0]])
    interc = A.intercepts(ext, translated, n_obj=2)
    assert interc == pytest.approx(translated.max(axis=0))


# --- Nischenbildung -----------------------------------------------------------------------------------------------------------------------


def test_niching_select_fills_fronts_and_prefers_empty_niches():
    # 2 Referenzpunkte; Front 0 hat 1 Punkt (Niche 0 besetzt), Front 1 hat 2 Punkte (je einer pro Niche) - bei pop_size=2
    # muss der Niche-1-Kandidat aus Front 1 gewählt werden (Niche 0 ist schon besetzt, Niche 1 noch nicht).
    fronts = [np.array([0]), np.array([1, 2])]
    ref_assoc = {0: 0, 1: 0, 2: 1}
    ref_dist = {0: 0.1, 1: 0.5, 2: 0.3}
    rng = np.random.default_rng(0)
    idx = A.niching_select(fronts, ref_assoc, ref_dist, n_refs=2, pop_size=2, rng=rng)
    assert sorted(idx.tolist()) == [0, 2]


def test_niching_select_keeps_exact_pop_size_across_many_sizes():
    rng = np.random.default_rng(1)
    objectives = rng.random((40, 4)) * 100
    ref = A.generate_reference_points(4, 4)
    ideal = A.ideal_point(objectives)
    translated = objectives - ideal
    ext = A.extreme_points(translated, 4)
    interc = A.intercepts(ext, translated, 4)
    norm = A.normalize_objectives(objectives, ideal, interc)
    assoc, dist = A.associate(norm, ref)
    fronts, _ = A.fast_non_dominated_sort(objectives)
    for pop_size in (1, 5, 20, 40):
        idx = A.niching_select(fronts, assoc, dist, len(ref), pop_size, rng)
        assert len(idx) == pop_size == len(set(idx.tolist()))


def test_rank_tournament_prefers_lower_rank_far_more_than_chance():
    rank = np.array([1, 1, 0, 1, 1])
    rng = np.random.default_rng(0)
    selected = A.rank_tournament_select(rank, k=len(rank), n_select=2000, rng=rng)
    assert (selected == 2).mean() > 0.5


# --- NSGA-III-Lauf auf einer sehr kleinen Instanz: nachweislich die Mehrheit der Brute-Force-Front (Mehrheit, nicht jeder Seed) -------------


def _brute_force_front(n_nodes, fn):
    tours = np.array([(0,) + p for p in permutations(range(1, n_nodes))])
    obj = fn(tours)
    unique_obj = np.unique(obj, axis=0)
    return unique_obj[A.non_dominated_mask(unique_obj)]


def test_nsga3_finds_most_of_the_brute_force_front_on_a_tiny_instance():
    rng = np.random.default_rng(42)
    n_nodes = 6
    xy = rng.random((n_nodes, 2)) * 100.0
    D = A.dist_matrix(xy)
    factors = []
    for _ in range(2):
        raw = rng.uniform(0.6, 3.4, size=(n_nodes, n_nodes))
        f = (raw + raw.T) / 2.0
        np.fill_diagonal(f, 0.0)
        factors.append(f)

    def fn(pop):
        dist = A.tour_length_batch(pop, D)
        others = [A.tour_edge_cost_batch(pop, D, f) for f in factors]
        return np.stack([dist] + others, axis=1)

    true_front = _brute_force_front(n_nodes, fn)
    hits = []
    for seed in range(10):
        r = A.run_nsga3(fn, n_nodes, pop_size=30, generations=60, cx_prob=0.9, mut_prob=0.2, tournament_k=2, divisions=4, seed=seed)
        found = r.front1
        reached = sum(np.any(np.all(np.abs(found - p) < 1e-6, axis=1)) for p in true_front)
        hits.append(reached / len(true_front))
    assert np.median(hits) >= 0.6
