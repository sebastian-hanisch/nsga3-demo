"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test.

Einzelne 150-Generationen-Läufe sind chaotisch empfindlich gegenüber winziger Fließkomma-Rundung (welche Route bei einem
Gleichstand gewinnt, kann über viele Generationen kaskadieren) - dieselbe Ursache wie jede andere CI-Linux-vs-Windows-
Abweichung in diesem Portfolio (siehe feedback_ci_platform_robust_tests.md, und die eigene Erfahrung aus nsga2-demo).
Zahlen aus einem EINZELNEN Lauf (Presets) bekommen deshalb großzügige Bänder; Zahlen, die über mehrere Seeds mitteln
(Experimente), sind von Natur aus robuster und behalten engere Bänder."""

import warnings

import pytest

import nsga3_constants as C
import nsga3_evaluation as E

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _preset_analysis(name):
    p = C.PRESETS[name]
    s = E.Settings(n=p["n"], seed=p["seed"], pop=p["pop"], gens=p["gens"], cx=p["cx"], mut=p["mut"], k=p["k"], divisions=p["divisions"], run_seed=p["run_seed"])
    return E.analyse3(s, keep_history=True)


def test_standardfall_preset_claims():
    a = _preset_analysis("Standardfall")
    r = a.result
    assert len(r.front1) >= 55           # praktisch die ganze Population, Einzellauf: nicht zwingend exakt 60
    covered, total = E.reference_coverage(r.front1, C.PRESETS["Standardfall"]["divisions"])
    assert total == 56
    assert covered == pytest.approx(19, abs=6)
    sat = E.saturation_share(r.front1_history, C.PRESETS["Standardfall"]["pop"])
    assert sat == pytest.approx(0.69, abs=0.2)


def test_kleine_population_preset_claims():
    a = _preset_analysis("Kleine Population")
    r = a.result
    assert len(r.front1) >= 18
    covered, total = E.reference_coverage(r.front1, C.PRESETS["Kleine Population"]["divisions"])
    assert covered == pytest.approx(11, abs=6)


def test_grosse_population_preset_claims():
    a = _preset_analysis("Große Population")
    r = a.result
    assert len(r.front1) >= 140
    covered, total = E.reference_coverage(r.front1, C.PRESETS["Große Population"]["divisions"])
    assert covered == pytest.approx(10, abs=7)


def test_wenige_referenzpunkte_preset_claims():
    a = _preset_analysis("Wenige Referenzpunkte")
    r = a.result
    covered, total = E.reference_coverage(r.front1, C.PRESETS["Wenige Referenzpunkte"]["divisions"])
    assert total == 20
    assert covered == pytest.approx(9, abs=5)


def test_kleine_instanz_preset_claims():
    a = _preset_analysis("Kleine Instanz (Vergleich mit Brute-Force)")
    p = C.PRESETS["Kleine Instanz (Vergleich mit Brute-Force)"]
    assert p["n"] == C.COMPARISON_N and p["seed"] == C.COMPARISON_VEHICLE_SEED
    sat = E.saturation_share(a.result.front1_history, p["pop"])
    assert sat == pytest.approx(0.93, abs=0.1)


# --- Headlinezahlen der beiden Experimente (mitteln über 20 Seeds, robuster) ------------------------------------------------------------


def test_coverage_experiment_headline_claims():
    rows = E.coverage_experiment()
    assert rows["total"] == 56
    assert rows["nsga2"]["covered_median"] == pytest.approx(14, abs=6)
    assert rows["nsga3"]["covered_median"] == pytest.approx(8, abs=6)
    # Kernbefund: NSGA-II deckt hier robust MEHR ab als NSGA-III (nicht nur in einem Einzellauf)
    assert rows["nsga2"]["covered_median"] > rows["nsga3"]["covered_median"]


def test_comparison_experiment_headline_claims():
    report = E.comparison_experiment()
    assert report["front_size"] == 23
    assert report["nsga2"]["reached_median"] == pytest.approx(22, abs=3)
    assert report["nsga3"]["reached_median"] == pytest.approx(17, abs=4)
    assert report["nsga2"]["reached_median"] > report["nsga3"]["reached_median"]
