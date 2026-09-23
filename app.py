"""NSGA-III - referenzpunktbasierte Nischenbildung statt Crowding-Distance - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Populations-Metaheuristiken-Linie der "Konzepte"-Reihe, Fortsetzung von nsga2-demo: dessen eigenes
Experiment zeigte, dass NSGA-IIs Crowding-Distance bei mehr Zielen nur unter knappem Budget messbar schwächelt. NSGA-III
(Deb & Jain, 2014) ersetzt Crowding-Distance durch feste Referenzpunkte auf dem Zielraum-Simplex und Nischenbildung. Diese
Demo zeigt am selben Vehikel wie nsga2-demo, jetzt mit einem VIERTEN Ziel (Risiko), den direkten Mechanismus-Vergleich - und
berichtet ehrlich, was dabei tatsächlich gemessen wurde, nicht was die Lehrbuch-Erwartung vorgibt.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import nsga3_constants as C
from nsga3_evaluation import Settings, analyse3, comparison_experiment, coverage_experiment, niche_occupancy, reference_coverage, sweep
from nsga3_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_run_seed, randomize_seed, sync_query_params
from nsga3_visualization import build_bruteforce_comparison, build_coverage_comparison, build_front1_size_curve, build_niche_occupancy, build_parallel_coordinates, build_sweep

st.set_page_config(page_title="NSGA-III – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse3(settings, keep_history=True)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _coverage():
    return coverage_experiment()


@st.cache_data(show_spinner=False)
def _comparison():
    return comparison_experiment()


st.title("🧬 NSGA-III – Referenzpunkte statt Crowding-Distance")
st.markdown(
    """
NSGA-II hält eine Front breit gefächert über die **Crowding-Distance** - wie dicht die Nachbarn eines Individuums im
Zielraum liegen. Bei **vielen Zielen** wird dieses Maß unschärfer: fast jede Lösung hat "weite" Nachbarn, die Distanz
verliert an Trennkraft. **NSGA-III** (Deb & Jain, 2014) ersetzt sie durch ein festes Gitter aus **Referenzpunkten** auf dem
Zielraum-Simplex - jedes Individuum wird dem nächsten Referenzpunkt zugeordnet, und beim Auffüllen der Population werden
gezielt unterbesetzte Referenzpunkte bevorzugt (**Nischenbildung**).
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - "
    "drittes Stück der Populations-Metaheuristiken-Linie der \"Konzepte\"-Reihe, Fortsetzung von "
    "[nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/) - **ein** Verfahren an einem wachsenden Beispiel, direkt "
    "gegen seinen Vorgänger gemessen. Vehikel ist dieselbe Lieferroute, jetzt mit vier Zielen: Distanz, CO2, Fahrzeit und Risiko."
)

with st.expander("So funktioniert NSGA-III", expanded=True):
    st.markdown(
        """
1. **Referenzpunkte.** Ein festes Gitter von Punkten auf dem Zielraum-Simplex (Das-Dennis-Verfahren) - bei vier Zielen und
   Divisions 5 sind das 56 Punkte, unabhängig von der Population.
2. **Normalisierung.** Idealpunkt (bester Wert je Ziel), Extrempunkte (über eine Achievement-Scalarizing-Function) und die
   Achsenabschnitte der Hyperebene durch sie - bildet die Population auf denselben Referenzraum ab.
3. **Zuordnung.** Jedes Individuum wird dem Referenzpunkt mit dem kürzesten senkrechten Abstand zugeordnet.
4. **Turnier + Crossover + Mutation.** Eltern werden nur nach **Rang** gewählt (kein Crowding/Niching bei der Paarung, wie im
   Originalpapier) - Nachkommen entstehen wie beim GA/NSGA-II (Order Crossover, Tausch-Mutation).
5. **Nischenbildung.** Eltern und Nachkommen (2N) werden nicht-dominiert sortiert; Fronten werden der Reihe nach übernommen.
   Passt die letzte Front nicht mehr vollständig: der am wenigsten besetzte Referenzpunkt bekommt Vorrang (bei Besetzung 0
   der nächstgelegene Kandidat, sonst ein zufälliger) - bis die Population voll ist.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider("Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Kundenstopps (das Depot kommt dazu).")
    st.markdown("**NSGA-III**")
    pop_size = st.slider("Populationsgröße", *bounds("pop_slider"), key="pop_slider", step=C.POP_STEP)
    generations = st.slider("Generationen", *bounds("gens_slider"), key="gens_slider", step=C.GEN_STEP)
    cx_prob = st.slider("Crossover-Rate", *bounds("cx_slider"), key="cx_slider", step=C.CX_STEP, format="%.2f")
    mut_prob = st.slider("Mutationsrate", *bounds("mut_slider"), key="mut_slider", step=C.MUT_STEP, format="%.2f")
    tournament_k = st.slider("Turniergröße k", *bounds("k_slider"), key="k_slider", step=C.K_STEP, help="NSGA-III vergleicht Turnierteilnehmer nur nach Rang - kein Crowding/Niching bei der Paarung.")
    divisions = st.slider("Referenzpunkt-Divisions H", *bounds("div_slider"), key="div_slider", step=C.DIV_STEP, help="Feinheit des Referenzpunkt-Gitters. Bei 4 Zielen: H=3 → 20 Punkte, H=5 → 56, H=7 → 120.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    run_seed = st.number_input("Zufalls-Seed des NSGA-III-Laufs", *bounds("run_seed_input"), key="run_seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_run_seed)

sync_query_params({
    "n_slider": int(n_stops), "pop_slider": int(pop_size), "gens_slider": int(generations), "cx_slider": float(cx_prob),
    "mut_slider": float(mut_prob), "k_slider": int(tournament_k), "div_slider": int(divisions), "seed_input": int(seed), "run_seed_input": int(run_seed),
})

settings = Settings(int(n_stops), 0, int(seed), int(pop_size), int(generations), float(cx_prob), float(mut_prob), int(tournament_k), int(divisions), int(run_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
result = a.result
n_gens_run = len(result.generations) - 1
data_key = settings

# --- NSGA-III in Aktion --------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 NSGA-III in Aktion")
if "nsga3_gen" not in st.session_state or st.session_state.get("nsga3_gen_owner") != data_key:
    st.session_state["nsga3_gen"] = n_gens_run
    st.session_state["nsga3_gen_owner"] = data_key
gen_col, play_col = st.columns([5, 2])
with gen_col:
    gen = st.slider("Generation", 0, n_gens_run, key="nsga3_gen", help="0 = Startpopulation.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_gens_run == 0:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_gens_run, min(n_gens_run + 1, 40))})


def _render(g):
    gd = result.generations[g]
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Generation {g} von {n_gens_run} – Front 1: {int((gd.rank == 0).sum())} von {settings.pop} Individuen**")
        c1.plotly_chart(build_parallel_coordinates(gd.objectives, gd.rank, C.OBJECTIVE_LABELS), width="stretch", key=f"g_pc_{g}")
        c2.markdown("**Größe von Front 1**")
        c2.plotly_chart(build_front1_size_curve(result.front1_history[:g + 1], settings.pop), width="stretch", key=f"g_curve_{g}")


if auto_play:
    for f in _frames():
        _render(f)
        time.sleep(0.15)
else:
    _render(gen)

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was NSGA-III gefunden hat")
covered, total_ref = reference_coverage(result.front1, settings.divisions)
m1, m2, m3 = st.columns(3)
m1.metric("Größe von Front 1", f"{len(result.front1)}", help="Wie viele Individuen der Endpopulation nicht dominiert sind.")
m2.metric("Abgedeckte Referenzpunkte", f"{covered} von {total_ref}", help="Wie viele der Referenzrichtungen mindestens ein Individuum aus Front 1 zugeordnet bekommen haben.")
m3.metric("Generationen × Population", f"{settings.gens} × {settings.pop}")

st.markdown("---")

# --- Sweep -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt die Referenzpunkt-Abdeckung von Population und Divisions ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(C.SWEEP_LABELS), format_func=lambda k: C.SWEEP_LABELS[k], key="sweep_select")
base_sweep = Settings(n=settings.n, cx=settings.cx, mut=settings.mut, k=settings.k, divisions=settings.divisions if sweep_param != "divisions" else settings.divisions, pop=settings.pop)
if st.button("Sweep über 5 feste Vehikel berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, C.SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")

st.markdown("---")

# --- Experiment 1: Mechanismus-Vergleich (ehrlicher Befund) ---------------------------------------------------------------------------------

st.subheader("🔬 Deckt NSGA-III den Zielraum bei vier Zielen besser ab als NSGA-II?")
st.caption("NSGA-II (aus nsga2-demo kopiert) gegen NSGA-III, beide bei denselben vier Zielen, faire (geteilte) Referenzpunkt-Abdeckung über 20 Läufe.")
if st.button("NSGA-II gegen NSGA-III rechnen (dauert etwa 20 Sekunden)", key="coverage_start"):
    st.session_state["coverage_on"] = True
if st.session_state.get("coverage_on"):
    with st.spinner("Rechne 20 NSGA-II- und 20 NSGA-III-Läufe..."):
        rows_cov = _coverage()
    st.plotly_chart(build_coverage_comparison(rows_cov), width="stretch", key="coverage_chart")
    cc1, cc2 = st.columns(2)
    cc1.metric("NSGA-II, Median", f"{rows_cov['nsga2']['covered_median']:.0f} von {rows_cov['total']}")
    cc2.metric("NSGA-III, Median", f"{rows_cov['nsga3']['covered_median']:.0f} von {rows_cov['total']}")
    st.warning(
        f"⚠️ **Ehrlicher Befund, nicht die Lehrbuch-Erwartung**: NSGA-II deckt hier MEHR Referenzpunkte ab als NSGA-III. Der Grund liegt in der Paarung, nicht in der Nischenbildung selbst: "
        f"in {rows_cov['nsga2']['saturation_median']:.0%} (NSGA-II) bzw. {rows_cov['nsga3']['saturation_median']:.0%} (NSGA-III) der Generationen ist Front 1 schon die ganze Population - "
        "Nischenbildung greift nur, wenn eine Front abgeschnitten werden muss, und das passiert bei vier weitgehend unabhängigen Zielen schnell nicht mehr. NSGA-IIIs Eltern-Turnier "
        "wählt danach rein nach Rang (keine Diversität mehr), während NSGA-IIs Crowding-Distance auch bei voller Front-1-Sättigung weiter eine schwache Streuungs-Richtung vorgibt. "
        "NSGA-IIIs eigentlicher Vorteil braucht deutlich mehr Ziele (die Originalarbeit testet bis 15) als diese Demo mit vier Zielen zeigen kann."
    )

st.markdown("---")

# --- Experiment 2: schließt nsga2-demos Cliffhanger (ebenfalls ehrlich) --------------------------------------------------------------------

st.subheader("🔬 Schließt sich nsga2-demos Cliffhanger jetzt zugunsten von NSGA-III?")
st.caption(f"Dieselbe kleine Instanz wie nsga2-demos Experimente ({C.COMPARISON_N} Stopps, Vehikel-Seed {C.COMPARISON_VEHICLE_SEED}), jetzt mit allen vier Zielen - Abdeckung der Brute-Force-Pareto-Front über 20 Läufe.")
if st.button("Brute-Force-Front gegen beide Verfahren rechnen (dauert etwa 20 Sekunden)", key="comparison_start"):
    st.session_state["comparison_on"] = True
if st.session_state.get("comparison_on"):
    with st.spinner("Rechne die Brute-Force-Front und je 20 NSGA-II-/NSGA-III-Läufe..."):
        report = _comparison()
    st.plotly_chart(build_bruteforce_comparison(report), width="stretch", key="comparison_chart")
    bc1, bc2 = st.columns(2)
    bc1.metric("NSGA-II, Median", f"{report['nsga2']['reached_median']:.0f} von {report['front_size']}")
    bc2.metric("NSGA-III, Median", f"{report['nsga3']['reached_median']:.0f} von {report['front_size']}")
    st.caption("Auch hier liegt NSGA-II gemessen vorn - dieselbe Erklärung wie oben. Kein Widerspruch zur Literatur: NSGA-IIIs Vorteil ist für echt viele Ziele belegt, nicht für vier auf einer kleinen Instanz.")

st.markdown("---")

# --- Wie Nischenbildung wirklich wirkt (didaktisch) -----------------------------------------------------------------------------------------

st.subheader("🔬 Wie sieht die Nischenbildung selbst aus?")
st.caption("Besetzung der Referenzpunkte durch Front 1 des aktuellen Laufs (rechts in der Seitenleiste einstellbar) - zeigt die Mechanik direkt, unabhängig vom Vergleich oben.")
_counts = niche_occupancy(result.front1, settings.divisions)
if len(result.front1) >= 4:
    st.plotly_chart(build_niche_occupancy(_counts), width="stretch", key="niche_chart")
    st.caption(f"{int((_counts > 0).sum())} von {len(_counts)} Referenzpunkten haben mindestens ein zugeordnetes Individuum; {int((_counts == 0).sum())} sind leer.")
else:
    st.caption("Front 1 hat zu wenige Individuen für diese Ansicht (mindestens 4 nötig).")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nischenbildung wird oft genug gebraucht** | Sättigt die Population (alle gegenseitig nicht-dominiert), greift Nischenbildung nicht mehr - beide Verfahren dieser Demo zeigen das bei nur vier Zielen schon nach wenigen Dutzend Generationen. | Mehr Ziele (die Originalarbeit testet bis 15) oder ein Vehikel mit einer echt größeren Pareto-Front |
| **Rein rangbasierte Paarung reicht** | Sobald Rang nicht mehr unterscheidet, hat NSGA-III keine Richtungskraft mehr bei der Elternwahl (anders als NSGA-II mit Crowding-Distance). | Eigene Varianten mit diversitätsbewusster Paarung (außerhalb dieser Linie) |
| **Ein einziges Verfahren für jede Zielzahl** | Ab sehr vielen Zielen wird selbst Referenzpunkt-basierte Nischenbildung schwerer zu parametrieren (Divisions wachsen kombinatorisch). | außerhalb dieser Linie |
| **Pareto-Dominanz ist die richtige Vergleichsregel** | Ein anderer Mechanismus - Zerlegung in Skalarisierungs-Unterprobleme statt Dominanz-Sortierung - löst dieselbe Aufgabe anders. | **MOEA/D** (Kontrast, kein Fix, direkter Nachfolger von NSGA-II) |
"""
)
st.caption(
    "Nachfolger von NSGA-III in der Populations-Linie: keiner geplant (NSGA-III ist ein Blatt der DAG). Geschwister: **MOEA/D** "
    "(Kontrast zu NSGA-II). Vorgänger: [nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Referenzpunkte.** Das-Dennis-Verfahren: alle $(n_1, \dots, n_M)$ mit $\sum_i n_i = H$, $n_i \ge 0$ ganzzahlig, geteilt durch
$H$ - Punkte auf dem $(M-1)$-dimensionalen Einheitssimplex. Anzahl $\binom{H+M-1}{M-1}$.

**Idealpunkt.** $z^{\min}_i = \min_x f_i(x)$ über die aktuelle Population.

**Extrempunkte.** Für jede Achse $j$: das Individuum, das $\max_k \frac{f_k(x) - z^{\min}_k}{w_k}$ minimiert, mit
$w = (\epsilon, \dots, 1_j, \dots, \epsilon)$ (Achievement-Scalarizing-Function).

**Achsenabschnitte.** Löse $\sum_k x_k / a_k = 1$ für die $M$ Extrempunkte; Rückfall auf den Nadir (Maximalwert je Achse)
bei singulärem System.

**Normalisierung.** $f'_i(x) = \frac{f_i(x) - z^{\min}_i}{a_i - z^{\min}_i}$.

**Zuordnung.** Kürzester senkrechter Abstand von $f'(x)$ zu jeder Referenzlinie (Ursprung durch Referenzpunkt $r$); das
Individuum gehört zur nächsten.

**Nischenbildung.** Fronten der Reihe nach in die neue Population übernehmen; für die letzte, nicht vollständig passende
Front: den am wenigsten besetzten Referenzpunkt wählen (unter denen mit verfügbaren Kandidaten), bei Besetzung 0 den
nächstgelegenen Kandidaten nehmen, sonst einen zufälligen - bis die Population voll ist.

Implementiert in `nsga3_algorithm.py` (Referenzpunkte, Normalisierung, Zuordnung, Nischenbildung, Hauptschleife; NSGA-II
als Vergleichsbasis wortgleich aus nsga2-demo kopiert), `nsga3_scenario.py` (Vehikel), `nsga3_evaluation.py` (Kennzahlen,
Sweep, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
