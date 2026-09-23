# 🧬 NSGA-III – Referenzpunkte statt Crowding-Distance

Drittes Stück der **Populations-Metaheuristiken-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) –
Operations Research und Machine Learning. Fortsetzung von [nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/):
NSGA-III (Deb & Jain, 2014) ersetzt NSGA-IIs Crowding-Distance durch ein festes Gitter aus **Referenzpunkten** auf dem
Zielraum-Simplex plus **Nischenbildung** - gedacht als Fix für NSGA-IIs bekannte Schwäche bei vielen Zielen. Diese Demo
erweitert das Vehikel auf **vier** Ziele (Distanz, CO2, Fahrzeit, Risiko) und misst NSGA-II gegen NSGA-III direkt.

## Warum dieses Problem

NSGA-IIs Crowding-Distance misst, wie dicht die Nachbarn einer Lösung im Zielraum liegen. Mit mehr Zielen wird dieses Maß
stumpfer - fast jede Lösung hat "weite" Nachbarn, die Distanz verliert an Trennkraft (ein in der Literatur gut
dokumentiertes Problem, Deb & Jain testen ihre Alternative an bis zu 15 Zielen). NSGA-III ersetzt die Distanz-basierte
Diversität durch ein explizites, festes Referenzpunkt-Gitter: jede Lösung wird dem nächsten Referenzpunkt zugeordnet,
unterbesetzte Punkte werden beim Auffüllen der Population bevorzugt.

## Ehrlicher Befund (Kernaussage dieser Demo)

**Bei vier Zielen auf dieser Instanzgröße deckt NSGA-II den Zielraum robust BESSER ab als NSGA-III** - nicht nur in einem
Einzellauf, sondern über 20 Läufe gemittelt, sowohl bei einer fairen Referenzpunkt-Abdeckung (Median 14 von 56 gegen 8 von
56) als auch bei der Abdeckung der Brute-Force-Pareto-Front auf einer kleinen Instanz (Median 22 von 23 gegen 17 von 23).
Das ist die **Gegenmessung** zur ursprünglichen Erwartung, nicht das erhoffte Ergebnis - und wird hier absichtlich nicht
versteckt.

**Die Erklärung ist mechanistisch, nicht mysteriös.** Bei vier weitgehend unabhängigen Zielen wird die Population schnell
gegenseitig nicht-dominiert (Front 1 füllt in 56–93 % der Generationen die gesamte Population, je nach Instanzgröße - siehe
Tests). Sobald das passiert, hat NSGA-IIIs Nischenbildung nichts mehr zu tun (sie greift nur beim Abschneiden einer zu
großen Front), und die Eltern-Turnierselektion wählt rein nach Rang - **keine Diversitätskraft mehr**. NSGA-IIs
Crowding-Distance-Turnier bleibt dagegen auch bei voller Sättigung aktiv und gibt weiter eine schwache Streuungsrichtung
vor. NSGA-IIIs eigentlicher Vorteil ist für echt viele Ziele (die Originalarbeit: bis 15) belegt - vier Ziele auf einer
Instanz dieser Größe reichen nicht, um ihn zu zeigen. Kein Widerspruch zur Literatur, nur eine ehrliche Grenze dieser Demo.

## Modell

Dieselbe Lieferroute wie nsga2-demo: ein Depot in der Mitte und *n* Kundenstopps in einem 100 × 100-km-Gebiet, euklidische
Entfernungen. Vier voneinander unabhängige Kantenmerkmale (nicht an der Distanz hängend): CO2-, Fahrzeit- und Risiko-Faktor
je Straßenabschnitt (0,6–3,4), identisch konstruiert. **xy sowie CO2- und Fahrzeit-Faktor der kleinen Vergleichsinstanz
(8 Stopps, Vehikel-Seed 19) sind bitidentisch zu genetic-algorithm-demo/nsga2-demo** - `nsga3_scenario.generate_perm`
reproduziert deren Erzeugung wortgleich, bevor das vierte Merkmal (Risiko) dazukommt.

## Methodik

NSGA-III-Kern (`nsga3_algorithm.py`): Referenzpunkte (Das-Dennis-Verfahren), Normalisierung (Idealpunkt, Extrempunkte über
eine Achievement-Scalarizing-Function, Hyperebenen-Achsenabschnitte), Zuordnung (kürzester senkrechter Abstand),
Nischenbildung bei der Überlebensauswahl. Mating-Selektion ist reines Rang-Turnier (wie im Originalpapier). Der komplette
NSGA-II-Kern ist wortgleich aus nsga2-demo kopiert und läuft im selben Repo als direkte Vergleichsbasis.

## Befunde / Korrekturen

Beim ersten Live-Test lieferte das Referenzpunkt-Abdeckungs-Experiment das Gegenteil der Erwartung (NSGA-II vorn statt
NSGA-III). Statt die Metrik zu verwerfen, wurde die Ursache gesucht: die anfängliche Fassung normierte jede Front an
ihrer EIGENEN Spannweite, was unterschiedlich geformte Fronten nicht fair vergleichbar machte - behoben durch eine geteilte
Normierung (Idealpunkt/Extrempunkte/Achsenabschnitte aus der Vereinigung beider Fronten). Der Befund blieb auch danach
bestehen (robust über mehrere Budgets, Populationsgrößen und die separate Brute-Force-Instanz geprüft) - siehe "Ehrlicher
Befund" oben.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Deckt NSGA-III den Zielraum bei vier Zielen besser ab? | NSGA-II Median 14 von 56 Referenzpunkten, NSGA-III Median **8 von 56** - NSGA-II liegt vorn | `test_coverage_experiment_headline_claims` |
| Schließt sich nsga2-demos Cliffhanger zugunsten von NSGA-III? | Auf derselben 8-Stopp-Instanz: NSGA-II Median 22 von 23 Frontpunkten, NSGA-III Median **17 von 23** - auch hier NSGA-II vorn | `test_comparison_experiment_headline_claims` |
| Wie oft sättigt die Population (Front 1 = ganze Population)? | Standardfall (30 Stopps) 69 %, kleine Instanz (8 Stopps) 93 % der Generationen | `test_standardfall_preset_claims`, `test_kleine_instanz_preset_claims` |
| Stimmen Referenzpunkte/Normalisierung/Zuordnung mit der Literatur überein? | Exakte Übereinstimmung mit `pymoo` über 15–30 Zufallsinstanzen (2–4 Ziele) | `test_generate_reference_points_matches_pymoo`, `test_ideal_and_extreme_points_match_pymoo`, `test_associate_matches_pymoo` |

## Ehrliche Grenzen

- **Nischenbildung wird oft genug gebraucht** - sättigt die Population, greift sie nicht mehr; bei vier Zielen auf dieser
  Instanzgröße passiert das schon nach wenigen Dutzend Generationen (siehe "Ehrlicher Befund" oben).
- **Rein rangbasierte Paarung** - sobald Rang nicht mehr unterscheidet, hat NSGA-III keine Richtungskraft mehr bei der
  Elternwahl, anders als NSGA-II mit Crowding-Distance.
- **Achsenabschnitte weichen bewusst von der pymoo-Bibliothek ab** - diese Demo folgt Deb & Jains Originalpapier (2014);
  pymoo deckelt den gelösten Achsenabschnitt zusätzlich am schlechtesten bisher beobachteten Populationswert (eigener
  Kommentar im pymoo-Quelltext: "different to the proposed version in the paper"). Referenzpunkte, Idealpunkt, Extrempunkte
  und Zuordnung stimmen exakt überein; nur die Achsenabschnitt-Rückfalllogik nicht (siehe `nsga3_algorithm.intercepts`).
- **O(N²) nicht-dominierte Sortierung** in der Populationsgröße - wie bei NSGA-II.

## Tests

103 Tests (`pytest tests/ -v`): Referenzpunkte/Idealpunkt/Extrempunkte/Zuordnung exakt gegen `pymoo` geprüft,
Achsenabschnitte per Handrechnung (diagonaler Fall + singulärer Rückfall), Nischenbildung gegen konstruierte Beispiele,
NSGA-III findet auf einer sehr kleinen Instanz nachweislich die Mehrheit der Brute-Force-Front, der kopierte NSGA-II-Kern
erneut knapp kreuzgeprüft, AppTest-Rauchtests (jedes Preset, Generation-Slider inkl. Abspielen, Permalink-Grenzen, alle drei
Experimente + Sweep auf Abruf) und `test_claims.py` (jede Zahl aus diesem README, mit CI-robusten Bändern für
Einzellauf-Kennzahlen - siehe `feedback_ci_platform_robust_tests.md`).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `nsga3_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Presets |
| `nsga3_presets.py` | Permalink/Presets-Mechanik |
| `nsga3_scenario.py` | Vehikel-Erzeuger (Lieferroute, Distanz/CO2/Fahrzeit/Risiko) |
| `nsga3_algorithm.py` | NSGA-III-Kern + kopierter NSGA-II-Kern (Vergleichsbasis) |
| `nsga3_evaluation.py` | Kennzahlen, Brute-Force-Referenz, Referenzpunkt-Abdeckung, Sweep, beide Experimente |
| `nsga3_visualization.py` | Plotly-Abbildungen (Parallelkoordinaten, Vergleiche) |

## Bewusst nicht umgesetzt

- Mehr als vier Ziele - genau die Größenordnung, bei der NSGA-IIIs Vorteil laut Literatur beginnt, aber diese Demo nicht
  mehr zeigen kann.
- Diversitätsbewusste Paarung für NSGA-III (außerhalb des Originalpapiers).
- Ein PDF-Export - wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
