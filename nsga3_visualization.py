"""Plotly-Abbildungen der NSGA-III-Demo: Parallelkoordinaten für die wachsende Population (vier Ziele auf einmal),
Vergleichsabbildungen NSGA-II gegen NSGA-III, Sweep. Achsen sind gesperrt (fixedrange), wo sinnvoll."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FRONT1_COLOR = "#54a24b"
OTHER_COLOR = "#9ecae9"
NSGA2_COLOR = "#4c78a8"
NSGA3_COLOR = "#f58518"
REF_COLOR = "#7f7f7f"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_parallel_coordinates(objectives, rank, labels):
    """Vier Achsen (Distanz, CO2, Fahrzeit, Risiko), eine Linie je Individuum, Front 1 grün hervorgehoben, Rest blass-blau."""
    is_front1 = (rank == 0).astype(int)
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=is_front1, colorscale=[[0, OTHER_COLOR], [1, FRONT1_COLOR]], showscale=False),
        dimensions=[dict(label=lab, values=objectives[:, i]) for i, lab in enumerate(labels)],
    ))
    fig.update_layout(height=340, margin=dict(l=40, r=40, t=30, b=10))
    return fig


def build_front1_size_curve(front1_history, pop_size):
    xs = list(range(len(front1_history)))
    ys = [len(f) for f in front1_history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=NSGA3_COLOR, width=2.5), name="Front 1"))
    fig.add_hline(y=pop_size, line=dict(color=REF_COLOR, dash="dot"), annotation_text="ganze Population", annotation_position="bottom right")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Größe von Front 1")
    fig.update_layout(showlegend=False)
    return _base(fig, 260)


def build_coverage_comparison(rows):
    """Referenzpunkt-Abdeckung NSGA-II gegen NSGA-III über mehrere Läufe (Box), plus die Sättigungs-Anteile als Text im Titel."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=rows["nsga2"]["covered_all"], name="NSGA-II", marker_color=NSGA2_COLOR, boxpoints="all"))
    fig.add_trace(go.Box(y=rows["nsga3"]["covered_all"], name="NSGA-III", marker_color=NSGA3_COLOR, boxpoints="all"))
    fig.add_hline(y=rows["total"], line=dict(color=REF_COLOR, dash="dot"), annotation_text="alle Referenzpunkte", annotation_position="top right")
    fig.update_yaxes(title_text="Abgedeckte Referenzpunkte")
    fig.update_layout(showlegend=False)
    return _base(fig, 340)


def build_bruteforce_comparison(report):
    fig = go.Figure()
    fig.add_trace(go.Box(y=report["nsga2"]["reached_all"], name="NSGA-II", marker_color=NSGA2_COLOR, boxpoints="all"))
    fig.add_trace(go.Box(y=report["nsga3"]["reached_all"], name="NSGA-III", marker_color=NSGA3_COLOR, boxpoints="all"))
    fig.add_hline(y=report["front_size"], line=dict(color=REF_COLOR, dash="dot"), annotation_text="volle Front", annotation_position="top right")
    fig.update_yaxes(title_text="Getroffene Frontpunkte")
    fig.update_layout(showlegend=False)
    return _base(fig, 340)


def build_niche_occupancy(counts, title=None):
    """Balkendiagramm: wie viele Individuen je Referenzpunkt zugeordnet sind (sortiert, damit Lücken sichtbar sind)."""
    fig = go.Figure()
    order = np.argsort(-counts)
    fig.add_trace(go.Bar(x=list(range(len(counts))), y=counts[order], marker_color=NSGA3_COLOR, showlegend=False))
    fig.update_xaxes(title_text="Referenzpunkt (sortiert nach Besetzung)")
    fig.update_yaxes(title_text="Zugeordnete Individuen")
    return _base(fig, 260)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Abgedeckte Referenzpunkte", "Anteil der Referenzpunkte"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=xs, y=[r["covered"] for r in rows], mode="lines+markers", line=dict(color=NSGA3_COLOR, width=2.5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["share"] for r in rows], mode="lines+markers", line=dict(color=NSGA3_COLOR, width=2.5), showlegend=False), row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_xaxes(title_text=param_label)
    return _base(fig, 300)
