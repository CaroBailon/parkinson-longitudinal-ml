"""Modelos mixtos para medidas repetidas (statsmodels).

Idea central: separar cada predictor en dos partes (descomposición "entre / dentro", o de Mundlak):
- componente ENTRE pacientes: el promedio de esa variable para el paciente
  -> responde "¿los pacientes con voz más ronca tienen, en general, más gravedad?"
- componente DENTRO del paciente: el desvío de cada semana respecto de su propio promedio
  -> responde "¿cuando la voz de un paciente empeora, empeora su gravedad?" (el predictor longitudinal)

El modelo mixto agrega un intercepto y una pendiente temporal ALEATORIOS por paciente:
cada paciente tiene su propio nivel inicial y su propio ritmo de progresión.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.analysis import ID

# Una medida por familia, para evitar la redundancia detectada en el notebook 01
VOZ_SELECCION = {
    "Jitter(Abs)": "jitter_abs",
    "Shimmer:APQ11": "shimmer_apq11",
    "HNR": "hnr",
    "RPDE": "rpde",
    "DFA": "dfa",
    "PPE": "ppe",
}


def preparar_datos_mixto(semanal: pd.DataFrame, objetivo: str = "total_UPDRS") -> pd.DataFrame:
    """Arma la tabla para el modelo mixto a partir de los datos semanales.

    - Renombra las variables de voz a nombres válidos para fórmulas.
    - Estandariza (z-score) cada medida para que los coeficientes sean comparables:
      'cambio en puntos de UPDRS por cada desvío estándar de la variable'.
    - Separa cada medida en su componente entre pacientes y dentro del paciente.
    """
    d = semanal[[ID, "semana", "age", "sex", objetivo] + list(VOZ_SELECCION)].rename(columns=VOZ_SELECCION)
    d = d.rename(columns={objetivo: "y"})
    d["semanas"] = d["semana"] - d.groupby(ID)["semana"].transform("min")  # 0 = primera semana del paciente
    d["edad_z"] = (d["age"] - d["age"].mean()) / d["age"].std()
    for v in VOZ_SELECCION.values():
        z = (d[v] - d[v].mean()) / d[v].std()
        d[f"{v}_entre"] = z.groupby(d[ID]).transform("mean")
        d[f"{v}_dentro"] = z - d[f"{v}_entre"]
    return d


def formula_mixto() -> str:
    terminos = ["semanas", "edad_z", "sex"]
    for v in VOZ_SELECCION.values():
        terminos += [f"{v}_entre", f"{v}_dentro"]
    return "y ~ " + " + ".join(terminos)


def ajustar_mixto(d: pd.DataFrame):
    """Modelo mixto con intercepto y pendiente temporal aleatorios por paciente."""
    modelo = smf.mixedlm(formula_mixto(), d, groups=d[ID], re_formula="~semanas")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # avisos de convergencia habituales en mixedlm
        return modelo.fit(method="lbfgs", reml=True)


def tabla_coeficientes(resultado) -> pd.DataFrame:
    """Efectos fijos con intervalo de confianza del 95% y p-valor, en formato largo para graficar."""
    ic = resultado.conf_int()
    tabla = pd.DataFrame({
        "coef": resultado.fe_params,
        "ic_inf": ic.loc[resultado.fe_params.index, 0],
        "ic_sup": ic.loc[resultado.fe_params.index, 1],
        "p_valor": resultado.pvalues.loc[resultado.fe_params.index],
    })
    tabla.index.name = "termino"
    tabla = tabla.reset_index()
    nombres = {v: k for k, v in VOZ_SELECCION.items()}

    def describir(termino):
        for sufijo, nivel in [("_entre", "Entre pacientes"), ("_dentro", "Dentro del paciente")]:
            if termino.endswith(sufijo):
                return nombres[termino[: -len(sufijo)]], nivel
        return termino, "Otros"

    tabla[["variable", "nivel"]] = tabla["termino"].apply(lambda t: pd.Series(describir(t)))
    tabla["significativo"] = tabla["p_valor"] < 0.05
    return tabla


def componentes_varianza(resultado) -> dict:
    """Varianzas del modelo: intercepto por paciente, pendiente por paciente y residuo."""
    cov = resultado.cov_re
    var_intercepto = float(cov.iloc[0, 0])
    var_pendiente = float(cov.iloc[1, 1])
    var_residuo = float(resultado.scale)
    return {
        "var_intercepto": var_intercepto,
        "sd_pendiente": np.sqrt(var_pendiente),
        "var_residuo": var_residuo,
        "icc_inicio": var_intercepto / (var_intercepto + var_residuo),
    }
