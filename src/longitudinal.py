"""Modelos mixtos para medidas repetidas (statsmodels).

Idea central: separar cada predictor en dos partes (descomposición "entre / dentro", o de Mundlak):
- componente ENTRE pacientes: el promedio de esa variable para el paciente
  -> responde "¿los pacientes con voz más ronca tienen, en general, más gravedad?"
- componente DENTRO del paciente: el desvío de cada semana respecto de su propio promedio
  -> responde "¿cuando la voz de un paciente empeora, empeora su gravedad?" (el predictor longitudinal)

El modelo mixto agrega un intercepto y una pendiente temporal ALEATORIOS por paciente:
cada paciente tiene su propio nivel inicial y su propio ritmo de progresión.

El tiempo entra en MESES (no en semanas): con semanas, la varianza de la pendiente es tan chica
comparada con la del intercepto que el optimizador no converge, y el resultado no convergido
subestima la varianza entre pacientes (lo que achica los intervalos de confianza).
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

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


DIAS_POR_MES = 30.4375


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
    d["meses"] = d["semanas"] * 7 / DIAS_POR_MES                             # el tiempo que usa el modelo
    d["edad_z"] = (d["age"] - d["age"].mean()) / d["age"].std()
    for v in VOZ_SELECCION.values():
        z = (d[v] - d[v].mean()) / d[v].std()
        d[f"{v}_entre"] = z.groupby(d[ID]).transform("mean")
        d[f"{v}_dentro"] = z - d[f"{v}_entre"]
    return d


def formula_mixto() -> str:
    terminos = ["meses", "edad_z", "sex"]
    for v in VOZ_SELECCION.values():
        terminos += [f"{v}_entre", f"{v}_dentro"]
    return "y ~ " + " + ".join(terminos)


def ajustar_mixto(d: pd.DataFrame):
    """Modelo mixto con intercepto y pendiente temporal aleatorios por paciente.

    No se silencian los avisos: si el optimizador no converge, los errores estándar no son confiables
    y conviene enterarse. Por eso además se verifica `converged` explícitamente.
    """
    modelo = smf.mixedlm(formula_mixto(), d, groups=d[ID], re_formula="~meses")
    resultado = modelo.fit(method="lbfgs", reml=True)
    if not resultado.converged:
        warnings.warn("El modelo mixto NO convergió: los p-valores e intervalos no son confiables.")
    return resultado


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


# ---------------------------------------------------------------------------
# Diagnóstico del modelo: ¿se cumplen los supuestos?
# ---------------------------------------------------------------------------
def tabla_diagnostico(d: pd.DataFrame, resultado) -> pd.DataFrame:
    """Chequea los supuestos de los que dependen los p-valores del modelo mixto.

    - Convergencia del optimizador.
    - Normalidad de los efectos aleatorios (42 valores, uno por paciente: Shapiro-Wilk es adecuado).
    - Residuos: forma (curtosis) y varianza constante. No se usa Shapiro con ~1.000 residuos:
      con tantos datos rechaza la normalidad por desvíos mínimos; para eso está el QQ-plot.
    - Independencia: autocorrelación de los residuos de una semana con la siguiente, dentro del paciente.
    """
    efectos = pd.DataFrame(resultado.random_effects).T
    efectos.columns = ["intercepto", "pendiente"]
    r = d.assign(residuo=resultado.resid, ajustado=resultado.fittedvalues).sort_values([ID, "semanas"])
    autocorr = r["residuo"].corr(r.groupby(ID)["residuo"].shift())
    filas = [
        ("Convergencia del optimizador", "Sí" if resultado.converged else "No",
         "Necesaria para confiar en errores estándar y p-valores"),
        ("Efectos aleatorios · intercepto (Shapiro p)", f"{stats.shapiro(efectos['intercepto']).pvalue:.2f}",
         "p > 0,05: compatible con normalidad"),
        ("Efectos aleatorios · pendiente (Shapiro p)", f"{stats.shapiro(efectos['pendiente']).pvalue:.2f}",
         "p > 0,05: compatible con normalidad"),
        ("Residuos · asimetría", f"{stats.skew(r['residuo']):+.2f}", "Cerca de 0: simétricos"),
        ("Residuos · exceso de curtosis", f"{stats.kurtosis(r['residuo']):.1f}",
         "> 0: colas más pesadas que la normal"),
        ("Residuos · |residuo| vs. ajustado (Spearman)",
         f"{stats.spearmanr(r['residuo'].abs(), r['ajustado']).statistic:+.2f}", "Cerca de 0: varianza constante"),
        ("Residuos · autocorrelación semana a semana", f"{autocorr:+.2f}",
         "Cerca de 0: independientes dentro del paciente"),
    ]
    return pd.DataFrame(filas, columns=["chequeo", "valor", "cómo leerlo"]).set_index("chequeo")


def chequeos_robustez(d: pd.DataFrame) -> pd.DataFrame:
    """Repite las dos conclusiones principales con métodos que NO dependen de los supuestos del modelo mixto.

    1. Progresión: se ajusta una recta por paciente y se testea si la pendiente promedio es distinta de 0
       (t de Student sobre 42 pendientes). La autocorrelación dentro del paciente no afecta este test.
    2. Efectos entre pacientes: regresión con una fila por paciente (promedios), sin estructura longitudinal.
    """
    pendientes = d.groupby(ID)[["meses", "y"]].apply(lambda g: np.polyfit(g["meses"], g["y"], 1)[0])
    t = stats.ttest_1samp(pendientes, 0)
    filas = [{"conclusión": "Progresión promedio (UPDRS por mes)", "método": "Recta por paciente + t de Student",
              "estimado": pendientes.mean(), "p_valor": t.pvalue}]

    entre = [f"{v}_entre" for v in VOZ_SELECCION.values()]
    pacientes = d.groupby(ID)[["y", "edad_z", "sex"] + entre].mean()
    ols = smf.ols("y ~ edad_z + sex + " + " + ".join(entre), pacientes).fit()
    nombres = {f"{v}_entre": k for k, v in VOZ_SELECCION.items()}
    for termino in entre:
        filas.append({"conclusión": f"Efecto entre pacientes · {nombres[termino]}",
                      "método": "Regresión con 1 fila por paciente",
                      "estimado": ols.params[termino], "p_valor": ols.pvalues[termino]})
    return pd.DataFrame(filas).set_index("conclusión")
