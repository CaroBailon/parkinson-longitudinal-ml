"""Utilidades de visualización con Plotly.

Toda la interactividad usa controles nativos de Plotly (menús desplegables), así funciona igual en
VS Code, Jupyter en el navegador y en los HTML exportados a reports/, sin depender de widgets.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

pio.templates.default = "plotly_white"


def guardar(fig, nombre: str, alto: int = 550) -> None:
    """Guarda la figura en reports/ como HTML interactivo y, si es posible, como PNG.

    - HTML: se abre en el navegador y mantiene la interactividad.
    - PNG: imagen estática que GitHub sí muestra (requiere el paquete 'kaleido').
    """
    REPORTS_DIR.mkdir(exist_ok=True)
    fig.write_html(REPORTS_DIR / f"{nombre}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(REPORTS_DIR / f"{nombre}.png", width=1100, height=alto, scale=2)
    except Exception as e:  # kaleido no instalado o sin navegador disponible
        print(f"(PNG no generado para '{nombre}': {type(e).__name__}. El HTML sí se guardó.)")


# ---------------------------------------------------------------------------
# Tarjetas de indicadores (KPI)
# ---------------------------------------------------------------------------
def tarjetas_kpi(indicadores: list[tuple[str, float, str]]) -> go.Figure:
    """Fila de tarjetas. Cada indicador es (titulo, valor, formato), p. ej. ("Pacientes", 42, ",.0f").

    Para porcentajes usar el formato ".0%" con el valor entre 0 y 1.
    """
    n = len(indicadores)
    fig = go.Figure()
    for i, (titulo, valor, formato) in enumerate(indicadores):
        fig.add_trace(go.Indicator(
            mode="number",
            value=valor,
            number={"valueformat": formato, "font": {"size": 40}},
            title={"text": titulo, "font": {"size": 14}},
            domain={"x": [i / n + 0.01, (i + 1) / n - 0.01], "y": [0, 1]},
        ))
    fig.update_layout(height=170, margin={"t": 40, "b": 10, "l": 10, "r": 10})
    return fig


# ---------------------------------------------------------------------------
# Explorador de trayectorias por paciente
# ---------------------------------------------------------------------------
def explorador_paciente(df: pd.DataFrame, id_col: str, tiempo: str, objetivos: list[str]) -> go.Figure:
    """Gráfico con un menú desplegable para elegir el paciente y ver la evolución de los objetivos."""
    pacientes = sorted(df[id_col].unique())
    n_obj = len(objetivos)
    fig = go.Figure()

    for k, pid in enumerate(pacientes):
        d = df[df[id_col] == pid].sort_values(tiempo)
        for obj in objetivos:
            fig.add_trace(go.Scatter(x=d[tiempo], y=d[obj], mode="lines+markers", name=obj,
                                     marker={"size": 4}, visible=(k == 0)))

    def titulo(pid):
        d = df[df[id_col] == pid]
        return f"Paciente {pid} · {d['age'].iloc[0]} años · {len(d)} registros"

    botones = []
    for k, pid in enumerate(pacientes):
        visible = [False] * (len(pacientes) * n_obj)
        visible[k * n_obj:(k + 1) * n_obj] = [True] * n_obj
        botones.append({"label": f"Paciente {pid}", "method": "update",
                        "args": [{"visible": visible}, {"title.text": titulo(pid)}]})

    fig.update_layout(
        title=titulo(pacientes[0]),
        xaxis_title="Días desde el inicio", yaxis_title="UPDRS",
        updatemenus=[{"buttons": botones, "direction": "down", "x": 1.0, "xanchor": "right",
                      "y": 1.18, "yanchor": "top"}],
        height=480,
    )
    return fig


# ---------------------------------------------------------------------------
# Explorador de dispersión: global vs. dentro del paciente
# ---------------------------------------------------------------------------
def explorador_dispersion(df: pd.DataFrame, centrados: pd.DataFrame, variables: list[str],
                          objetivo: str, correlaciones: pd.DataFrame, id_col: str) -> go.Figure:
    """Dos paneles lado a lado (global | dentro del paciente) y un menú para elegir la medida de voz."""
    def subtitulos(var):
        return (f"Global · Spearman {correlaciones.loc[var, 'global']:+.2f}",
                f"Dentro del paciente · Spearman {correlaciones.loc[var, 'dentro_paciente']:+.2f}")

    v0 = variables[0]
    fig = make_subplots(rows=1, cols=2, subplot_titles=subtitulos(v0))
    color = df[id_col]
    for col, datos in [(1, df), (2, centrados)]:
        fig.add_trace(go.Scattergl(x=datos[v0], y=datos[objetivo], mode="markers",
                                   marker={"size": 4, "opacity": 0.35, "color": color,
                                           "colorscale": "Turbo"},
                                   showlegend=False, hovertemplate="x=%{x:.4f}<br>y=%{y:.1f}"),
                      row=1, col=col)

    botones = []
    for var in variables:
        s1, s2 = subtitulos(var)
        botones.append({
            "label": var, "method": "update",
            "args": [{"x": [df[var].tolist(), centrados[var].tolist()]},
                     {"annotations[0].text": s1, "annotations[1].text": s2,
                      "xaxis.title.text": var, "xaxis2.title.text": f"{var} (centrada por paciente)"}],
        })

    fig.update_layout(
        title=f"Medidas de voz vs. {objetivo} (color = paciente)",
        updatemenus=[{"buttons": botones, "direction": "down", "x": 1.0, "xanchor": "right",
                      "y": 1.22, "yanchor": "top"}],
        height=500,
    )
    fig.update_xaxes(title_text=v0, row=1, col=1)
    fig.update_xaxes(title_text=f"{v0} (centrada por paciente)", row=1, col=2)
    fig.update_yaxes(title_text=objetivo, row=1, col=1)
    fig.update_yaxes(title_text=f"{objetivo} (centrada)", row=1, col=2)
    return fig


# ---------------------------------------------------------------------------
# Diagnóstico de un modelo mixto
# ---------------------------------------------------------------------------
def _qq(valores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cuantiles teóricos (normal) vs. observados, y la recta de referencia."""
    from scipy import stats
    (teoricos, observados), (pendiente, ordenada, _) = stats.probplot(np.asarray(valores), dist="norm")
    return teoricos, observados, pendiente * teoricos + ordenada


def panel_diagnostico(residuos: pd.Series, ajustados: pd.Series, interceptos: pd.Series,
                      residuo_anterior: pd.Series) -> go.Figure:
    """Cuatro gráficos para revisar los supuestos de un modelo mixto."""
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.1, subplot_titles=(
        "QQ-plot de residuos", "Residuos vs. valores ajustados",
        "QQ-plot de interceptos aleatorios (1 por paciente)", "Residuo vs. residuo de la semana anterior"))
    gris = {"color": "gray", "dash": "dash"}

    for (fila, col), valores in [((1, 1), residuos), ((2, 1), interceptos)]:
        x, y, recta = _qq(valores)
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker={"size": 4, "opacity": 0.5}), row=fila, col=col)
        fig.add_trace(go.Scatter(x=x, y=recta, mode="lines", line=gris), row=fila, col=col)
        fig.update_xaxes(title_text="Cuantil teórico (normal)", row=fila, col=col)
        fig.update_yaxes(title_text="Cuantil observado", row=fila, col=col)

    fig.add_trace(go.Scatter(x=ajustados, y=residuos, mode="markers", marker={"size": 4, "opacity": 0.4}),
                  row=1, col=2)
    fig.add_hline(y=0, line=gris, row=1, col=2)
    fig.update_xaxes(title_text="Valor ajustado (UPDRS)", row=1, col=2)
    fig.update_yaxes(title_text="Residuo", row=1, col=2)

    fig.add_trace(go.Scatter(x=residuo_anterior, y=residuos, mode="markers", marker={"size": 4, "opacity": 0.4}),
                  row=2, col=2)
    fig.update_xaxes(title_text="Residuo semana anterior", row=2, col=2)
    fig.update_yaxes(title_text="Residuo", row=2, col=2)

    fig.update_layout(title="Diagnóstico del modelo mixto", showlegend=False, height=800)
    return fig
