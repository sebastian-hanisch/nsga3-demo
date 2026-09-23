"""Presets: Vollständigkeit, gültige Werte, Grenzen/Schrittweiten - reine Datenprüfungen ohne Streamlit-Session
(Permalink-Klammern und Preset-Knöpfe werden über AppTest in test_app.py geprüft, wie im Rest des Portfolios üblich)."""

import nsga3_constants as C
import nsga3_evaluation as E
import nsga3_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX
        assert C.POP_MIN <= p["pop"] <= C.POP_MAX and C.GEN_MIN <= p["gens"] <= C.GEN_MAX
        assert C.DIV_MIN <= p["divisions"] <= C.DIV_MAX
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall"]
    s = E.Settings(n=p["n"], seed=p["seed"], pop=p["pop"], gens=p["gens"], cx=p["cx"], mut=p["mut"], k=p["k"], divisions=p["divisions"], run_seed=p["run_seed"])
    assert s == E.Settings()


def test_bounds_and_steps_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert set(P.STEPS) == {"n_slider", "pop_slider", "gens_slider", "cx_slider", "mut_slider", "k_slider", "div_slider"}


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_comparison_preset_uses_the_shared_small_instance():
    p = C.PRESETS["Kleine Instanz (Vergleich mit Brute-Force)"]
    assert p["n"] == C.COMPARISON_N and p["seed"] == C.COMPARISON_VEHICLE_SEED
