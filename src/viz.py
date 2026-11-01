"""Utilidades de visualización: estilo común y guardado de figuras Plotly."""
from pathlib import Path

import plotly.io as pio

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

pio.templates.default = "plotly_white"


def guardar(fig, nombre: str) -> None:
    """Guarda la figura en reports/ como HTML interactivo y, si es posible, como PNG.

    - HTML: se abre en el navegador y mantiene la interactividad.
    - PNG: imagen estática que GitHub sí muestra (requiere el paquete 'kaleido').
    """
    REPORTS_DIR.mkdir(exist_ok=True)
    fig.write_html(REPORTS_DIR / f"{nombre}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(REPORTS_DIR / f"{nombre}.png", width=1100, height=550, scale=2)
    except Exception as e:  # kaleido no instalado o sin navegador disponible
        print(f"(PNG no generado para '{nombre}': {type(e).__name__}. El HTML sí se guardó.)")
