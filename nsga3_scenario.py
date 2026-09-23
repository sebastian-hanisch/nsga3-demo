"""Vehikel der NSGA-III-Demo: dieselbe Lieferroute wie nsga2-demo (Depot + n Kundenstopps, 100 x 100-km-Gebiet, euklidische
Entfernungen) - `xy`-, CO2- und Fahrzeit-Faktor-Erzeugung wortgleich aus `nsga2_scenario.generate_perm` kopiert, damit die
kleine Vergleichsinstanz (n=8, Seed 19) weiterhin bitidentisch zu genetic-algorithm-demo/nsga2-demo ist. Neu: ein viertes,
unabhängiges Kantenmerkmal Risiko (derselbe Konstruktionstrick - manche Straßenabschnitte sind unfallträchtiger als andere,
unabhängig von Distanz, CO2 und Fahrzeit). Vier Ziele insgesamt - die eigentliche Motivation von NSGA-III braucht echt viele
Ziele, nicht nur zwei oder drei."""

from dataclasses import dataclass

import numpy as np

import nsga3_constants as C


@dataclass(frozen=True)
class PermInstance:
    xy: np.ndarray                    # (n + 1, 2); Zeile 0 = Depot
    co2_factor_matrix: np.ndarray     # (n + 1, n + 1); symmetrisch
    time_factor_matrix: np.ndarray    # (n + 1, n + 1); symmetrisch, unabhängig vom CO2-Faktor
    risk_factor_matrix: np.ndarray    # (n + 1, n + 1); symmetrisch, unabhängig von CO2/Fahrzeit
    n: int
    cluster_share: int
    seed: int

    @property
    def n_nodes(self):
        return self.n + 1


def generate_perm(n, cluster_share=0, seed=0):
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
    co2_factor_matrix = (raw_co2 + raw_co2.T) / 2.0
    np.fill_diagonal(co2_factor_matrix, 0.0)
    raw_time = rng.uniform(C.TIME_FACTOR_LO, C.TIME_FACTOR_HI, size=(n_nodes, n_nodes))
    time_factor_matrix = (raw_time + raw_time.T) / 2.0
    np.fill_diagonal(time_factor_matrix, 0.0)
    raw_risk = rng.uniform(C.RISK_FACTOR_LO, C.RISK_FACTOR_HI, size=(n_nodes, n_nodes))
    risk_factor_matrix = (raw_risk + raw_risk.T) / 2.0
    np.fill_diagonal(risk_factor_matrix, 0.0)
    return PermInstance(xy, co2_factor_matrix, time_factor_matrix, risk_factor_matrix, n, int(cluster_share), int(seed))
