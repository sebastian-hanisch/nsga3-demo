"""Vehikel (Reproduzierbarkeit, bitidentisch zu nsga2-demo für xy/CO2/Fahrzeit) und Auswertung (Objektive, Brute-Force-Front,
Referenzpunkt-Abdeckung, Sättigung, Sweep, beide Experimente) - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import nsga3_algorithm as A
import nsga3_constants as C
import nsga3_evaluation as E
import nsga3_scenario as S


def test_generate_perm_is_reproducible_and_shaped():
    a = S.generate_perm(20, cluster_share=30, seed=7)
    b = S.generate_perm(20, cluster_share=30, seed=7)
    assert np.array_equal(a.xy, b.xy) and np.array_equal(a.co2_factor_matrix, b.co2_factor_matrix) and np.array_equal(a.risk_factor_matrix, b.risk_factor_matrix)
    assert a.xy.shape == (21, 2) and a.risk_factor_matrix.shape == (21, 21)
    assert not np.array_equal(a.co2_factor_matrix, a.risk_factor_matrix)
    assert not np.array_equal(a.time_factor_matrix, a.risk_factor_matrix)


def test_risk_factor_matrix_is_symmetric_zero_diagonal_in_range():
    inst = S.generate_perm(30, 0, seed=1)
    m = inst.risk_factor_matrix
    assert np.array_equal(m, m.T)
    assert np.all(np.diag(m) == 0.0)
    off_diag = m[~np.eye(len(m), dtype=bool)]
    assert off_diag.min() >= C.RISK_FACTOR_LO and off_diag.max() <= C.RISK_FACTOR_HI


def test_generate_perm_matches_nsga2_demo_bit_for_bit_on_the_comparison_instance():
    """Die kleine Vergleichsinstanz (n=8, Seed 19) muss xy/CO2/Fahrzeit bitidentisch zu nsga2-demo liefern - reproduziert
    nsga2_scenario.generate_perm hier lokal (kein Cross-Repo-Import, wie überall im Portfolio)."""
    def nsga2_generate_perm(n, cluster_share, seed):
        rng = np.random.default_rng(seed)
        n_grouped = int(round(n * cluster_share / 100))
        uniform = rng.random((n - n_grouped, 2)) * C.AREA
        centres = C.CLUSTER_MARGIN + rng.random((C.N_CLUSTERS, 2)) * (C.AREA - 2 * C.CLUSTER_MARGIN)
        which = rng.integers(0, C.N_CLUSTERS, size=n_grouped)
        grouped = np.clip(centres[which] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n_grouped, 2)), 0.0, C.AREA)
        depot = np.array([[C.AREA / 2, C.AREA / 2]])
        xy = np.vstack([depot, uniform, grouped])
        n_nodes = n + 1
        raw_co2 = rng.uniform(C.CO2_FACTOR_LO, C.CO2_FACTOR_HI, size=(n_nodes, n_nodes))
        co2 = (raw_co2 + raw_co2.T) / 2.0
        np.fill_diagonal(co2, 0.0)
        raw_time = rng.uniform(C.TIME_FACTOR_LO, C.TIME_FACTOR_HI, size=(n_nodes, n_nodes))
        time_ = (raw_time + raw_time.T) / 2.0
        np.fill_diagonal(time_, 0.0)
        return xy, co2, time_

    xy_ref, co2_ref, time_ref = nsga2_generate_perm(C.COMPARISON_N, 0, C.COMPARISON_VEHICLE_SEED)
    inst = S.generate_perm(C.COMPARISON_N, 0, C.COMPARISON_VEHICLE_SEED)
    assert np.array_equal(inst.xy, xy_ref)
    assert np.array_equal(inst.co2_factor_matrix, co2_ref)
    assert np.array_equal(inst.time_factor_matrix, time_ref)


def test_objective_fn_has_four_objectives_and_matches_manual_computation():
    s = E.Settings(n=5, seed=1, pop=10, gens=5)
    fn, n_nodes = E.objective_fn(s)
    pop = np.array([np.arange(n_nodes)])
    obj = fn(pop)
    assert obj.shape == (1, 4)
    inst, D = E.instance(s.n, s.cluster_share, s.seed)
    dist = A.tour_length_batch(pop, D)
    co2 = A.tour_edge_cost_batch(pop, D, inst.co2_factor_matrix)
    assert obj[0, 0] == pytest.approx(dist[0]) and obj[0, 1] == pytest.approx(co2[0])


def test_brute_force_front_is_non_dominated_and_deduplicated_at_small_n():
    s = E.Settings(n=6, seed=3)
    fn, n_nodes = E.objective_fn(s)
    all_obj, front = E.brute_force_front(n_nodes, fn)
    assert len(all_obj) == 720
    assert len(np.unique(front, axis=0)) == len(front)
    assert not A.dominance_matrix(front).any()


def test_comparison_instance_front_size_is_reasonable():
    s = E.Settings(n=C.COMPARISON_N, seed=C.COMPARISON_VEHICLE_SEED)
    fn, n_nodes = E.objective_fn(s)
    _, front = E.brute_force_front(n_nodes, fn)
    assert 1 <= len(front) <= 8 * 7 * 6 * 5   # grobe Plausibilitätsgrenze, keine feste Zahl (4 Ziele -> viel größere Front als bei 2)


def test_reference_coverage_is_bounded():
    s = E.Settings(n=10, seed=1, pop=20, gens=20)
    r = E.run3(s, keep_history=False)
    covered, total = E.reference_coverage(r.front1, s.divisions)
    assert 0 <= covered <= total


def test_niche_occupancy_sums_to_front_size():
    s = E.Settings(n=10, seed=1, pop=20, gens=20)
    r = E.run3(s, keep_history=False)
    counts = E.niche_occupancy(r.front1, s.divisions)
    assert counts.sum() == len(r.front1)


def test_saturation_share_is_a_fraction():
    s = E.Settings(n=10, seed=1, pop=20, gens=30)
    r = E.run3(s, keep_history=True)
    share = E.saturation_share(r.front1_history, s.pop)
    assert 0.0 <= share <= 1.0


def test_run_config_and_sweep_smoke():
    rows = E.sweep("pop", base=E.Settings(n=10, gens=20), values=(20, 30))
    assert len(rows) == 2
    assert all(0 <= r["covered"] <= r["total"] for r in rows)


def test_coverage_experiment_smoke_small():
    rows = E.coverage_experiment(n=10, pop=20, gens=20, seeds=(1, 2))
    assert rows["total"] > 0
    for label in ("nsga2", "nsga3"):
        assert 0 <= rows[label]["covered_median"] <= rows["total"]
        assert 0.0 <= rows[label]["saturation_median"] <= 1.0


def test_comparison_experiment_smoke_small():
    report = E.comparison_experiment(n=6, seed=3, pop=20, gens=30, seeds=(1, 2))
    assert report["front_size"] >= 1
    for label in ("nsga2", "nsga3"):
        assert 0 <= report[label]["reached_median"] <= report["front_size"]
