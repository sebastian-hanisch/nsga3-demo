"""Konstanten der NSGA-III-Demo: Vehikel (wie nsga2-demo, plus viertes Ziel Risiko), NSGA-II- und NSGA-III-Regler, Presets
(Presets folgen nach den Messungen)."""

# --- Vehikel: Lieferroute (wie nsga2-demo/genetic-algorithm-demo) --------------------------------------------------------------------------

AREA = 100.0
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0
CLUSTER_MARGIN = 12.0
N_MIN, N_MAX, DEFAULT_N, N_STEP = 8, 100, 30, 2

# CO2-, Fahrzeit- und Risiko-Faktor je Straßenabschnitt (Kante), unabhängig voneinander und von der Distanz gezogen.
# CO2-/Fahrzeit-Bereich bewusst identisch zu nsga2-demo, damit die kleine Vergleichsinstanz (n=8, Seed 19) bitidentisch
# bleibt (xy + beide Faktoren werden vor dem neuen vierten Faktor gezogen, siehe nsga3_scenario.generate_perm).
CO2_FACTOR_LO, CO2_FACTOR_HI = 0.6, 3.4
TIME_FACTOR_LO, TIME_FACTOR_HI = 0.6, 3.4
RISK_FACTOR_LO, RISK_FACTOR_HI = 0.6, 3.4

N_OBJ = 4                          # diese Demo zeigt bewusst immer alle vier Ziele - keine 2/3-Umschaltung wie bei nsga2-demo
OBJECTIVE_LABELS = ("Distanz", "CO2", "Fahrzeit", "Risiko")

# --- Gemeinsame GA-Regler (NSGA-II- und NSGA-III-Lauf teilen sich Population/Generationen/Operatorraten) ------------------------------------

POP_MIN, POP_MAX, DEFAULT_POP, POP_STEP = 20, 200, 60, 10
GEN_MIN, GEN_MAX, DEFAULT_GEN, GEN_STEP = 10, 400, 150, 10
CX_MIN, CX_MAX, DEFAULT_CX, CX_STEP = 0.0, 1.0, 0.9, 0.05
MUT_MIN, MUT_MAX, DEFAULT_MUT, MUT_STEP = 0.0, 1.0, 0.2, 0.05
K_MIN, K_MAX, DEFAULT_K, K_STEP = 2, 8, 2, 1
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_RUN_SEED = 7

DIV_MIN, DIV_MAX, DEFAULT_DIV, DIV_STEP = 3, 8, 5, 1    # Divisions H der Referenzpunkte (bei n_obj=4, H=5: 56 Punkte)

# --- Kleine Vergleichsinstanz: identisch zur Pareto-Front-Instanz der genetic-algorithm-demo / nsga2-demo -----------------------------------

COMPARISON_N = 8
COMPARISON_VEHICLE_SEED = 19
COMPARISON_SEEDS = tuple(range(800000, 800020))
COMPARISON_POP, COMPARISON_GENS = 60, 150

# --- Referenzpunkt-Abdeckungs-Experiment (Mechanismus-Vergleich: Crowding-Distance vs. Nischenbildung) -----------------------------------------

COVERAGE_SEEDS = tuple(range(900000, 900020))
COVERAGE_POP, COVERAGE_GENS = 60, 150

SWEEP_SEEDS = tuple(range(700000, 700005))
SWEEP_VALUES = {"pop": (20, 40, 60, 100, 150), "divisions": (3, 4, 5, 6, 7)}
SWEEP_LABELS = {"pop": "Populationsgröße", "divisions": "Referenzpunkt-Divisions H"}


def _preset(n=DEFAULT_N, pop=DEFAULT_POP, gens=DEFAULT_GEN, cx=DEFAULT_CX, mut=DEFAULT_MUT, k=DEFAULT_K, divisions=DEFAULT_DIV, seed=DEFAULT_SEED, run_seed=DEFAULT_RUN_SEED):
    # kein "ballung"-Regler in dieser Demo (anders als nsga2-demo) - cluster_share bleibt fest bei 0, siehe app.py
    return {"n": n, "pop": pop, "gens": gens, "cx": cx, "mut": mut, "k": k, "divisions": divisions, "seed": seed, "run_seed": run_seed}


PRESETS = {
    "Standardfall": _preset(),
    "Kleine Population": _preset(pop=20),
    "Große Population": _preset(pop=150),
    "Wenige Referenzpunkte": _preset(divisions=3),
    "Kleine Instanz (Vergleich mit Brute-Force)": _preset(n=COMPARISON_N, seed=COMPARISON_VEHICLE_SEED, pop=COMPARISON_POP, gens=COMPARISON_GENS),
}
# Gemessen mit dem jeweiligen Preset-Seed (siehe PRESETS) - ein einzelner Lauf, kein Mittel über mehrere Seeds
# (Ausnahme: der Verweis auf die Experimente weiter unten in der App, die selbst über mehrere Seeds mitteln).
PRESET_HELP = {
    "Standardfall": "30 Stopps, Populationsgröße 60, 150 Generationen, Divisions 5 (56 Referenzpunkte): Front 1 wächst auf alle 60 Individuen, deckt davon 19 von 56 Referenzpunkten ab. In 69 % der Generationen ist Front 1 schon die ganze Population - dort greift Nischenbildung nicht mehr.",
    "Kleine Population": "Nur 20 Individuen: Front 1 wird ebenfalls vollständig (20 von 20), deckt aber nur 11 von 56 Referenzpunkten ab - eine kleinere Population hat schlicht weniger Individuen, um Referenzpunkte zu besetzen.",
    "Große Population": "150 statt 60 Individuen: Front 1 wird auch hier vollständig (150 von 150), deckt in diesem einzelnen Lauf 10 von 56 Referenzpunkten ab - kein verlässlicher Trend aus nur einem Lauf, das robuste Bild über viele Läufe liefert das Experiment weiter unten.",
    "Wenige Referenzpunkte": "Divisions 3 statt 5 (nur 20 statt 56 Referenzpunkte): Front 1 deckt 9 von 20 ab (45 %) - ein gröberes Gitter ist leichter vollständig zu erreichen, unterscheidet aber auch weniger fein zwischen Kompromissen.",
    "Kleine Instanz (Vergleich mit Brute-Force)": "Dieselbe kleine Instanz (8 Stopps) wie das Experiment „Schließt sich nsga2-demos Cliffhanger...“ weiter unten. Hier sättigt Front 1 in 93 % der Generationen (gegenüber 69 % im Standardfall mit 30 Stopps) - bei so wenigen Stopps wird die ganze Population fast von Anfang an gegenseitig nicht-dominiert.",
}
