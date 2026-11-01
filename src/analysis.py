"""Funciones de análisis para datos longitudinales (una fila por medición, varias por paciente).

Se usan desde los notebooks para que las celdas queden cortas y los cálculos sean reutilizables.
"""
import pandas as pd

ID = "subject_id"
TIEMPO = "test_time"
OBJETIVOS = ["motor_UPDRS", "total_UPDRS"]
NO_VOZ = [ID, "age", "sex", TIEMPO] + OBJETIVOS


def columnas_voz(df: pd.DataFrame) -> list[str]:
    """Devuelve las 16 medidas de voz (todo lo que no es identificación, tiempo u objetivo)."""
    numericas = df.select_dtypes("number").columns
    return [c for c in numericas if c not in NO_VOZ]


# ---------------------------------------------------------------------------
# Calidad de datos
# ---------------------------------------------------------------------------
def calidad_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de control de calidad: tipo, faltantes, valores únicos, mínimo y máximo por columna."""
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "faltantes": df.isna().sum(),
        "unicos": df.nunique(),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
    })


# ---------------------------------------------------------------------------
# Diseño longitudinal
# ---------------------------------------------------------------------------
def resumen_por_paciente(df: pd.DataFrame, objetivo: str = "total_UPDRS") -> pd.DataFrame:
    """Pasa de 'una fila por medición' a 'una fila por paciente'."""
    g = df.sort_values([ID, TIEMPO]).groupby(ID)
    res = g.agg(
        n_registros=(TIEMPO, "size"),
        dia_min=(TIEMPO, "min"),
        dia_max=(TIEMPO, "max"),
        edad=("age", "first"),
        sexo=("sex", "first"),
        updrs_inicial=(objetivo, "first"),
        updrs_final=(objetivo, "last"),
    )
    res["seguimiento_dias"] = res["dia_max"] - res["dia_min"]
    res["cambio_updrs"] = res["updrs_final"] - res["updrs_inicial"]
    res["sexo"] = res["sexo"].map({0: "Hombre", 1: "Mujer"})
    return res


# ---------------------------------------------------------------------------
# Varianza entre / dentro de pacientes (aproximación al ICC)
# ---------------------------------------------------------------------------
def varianza_entre_pacientes(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Para cada columna: qué proporción de su varianza se explica por QUIÉN es el paciente.

    Cerca de 1  -> la variable casi no cambia dentro del paciente (es 'una firma' del paciente).
    Cerca de 0  -> la variable cambia mucho de una medición a otra del mismo paciente.
    """
    filas = []
    for col in columnas:
        var_total = df[col].var()
        var_entre = df.groupby(ID)[col].transform("mean").var()
        prop = var_entre / var_total
        filas.append({"variable": col, "entre_pacientes": prop, "dentro_paciente": 1 - prop})
    return pd.DataFrame(filas).set_index("variable").sort_values("entre_pacientes", ascending=False)


# ---------------------------------------------------------------------------
# Correlaciones
# ---------------------------------------------------------------------------
def centrar_por_paciente(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Resta a cada valor el promedio de su paciente: deja solo la variación DENTRO del paciente."""
    return df[columnas] - df.groupby(ID)[columnas].transform("mean")


def correlaciones_con_objetivo(df: pd.DataFrame, objetivo: str = "total_UPDRS") -> pd.DataFrame:
    """Correlación de Spearman de cada medida de voz con el objetivo, en dos niveles:

    - global: mezcla diferencias entre pacientes y cambios en el tiempo.
    - dentro_paciente: solo los cambios de cada paciente respecto de su propio promedio.
      Es la pregunta longitudinal: "cuando la voz de ESTE paciente empeora, ¿empeora su UPDRS?"
    """
    voz = columnas_voz(df)
    glob = df[voz].corrwith(df[objetivo], method="spearman")
    cent = centrar_por_paciente(df, voz + [objetivo])
    dentro = cent[voz].corrwith(cent[objetivo], method="spearman")
    res = pd.DataFrame({"global": glob, "dentro_paciente": dentro})
    return res.reindex(res["global"].abs().sort_values(ascending=False).index)


def pares_redundantes(df: pd.DataFrame, umbral: float = 0.9) -> pd.DataFrame:
    """Pares de medidas de voz con |correlación| >= umbral (multicolinealidad)."""
    corr = df[columnas_voz(df)].corr(method="spearman")
    pares = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= umbral:
                pares.append({"var_1": cols[i], "var_2": cols[j], "r": r})
    if not pares:
        return pd.DataFrame(columns=["var_1", "var_2", "r"])
    return pd.DataFrame(pares).sort_values("r", key=abs, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Resumen ejecutivo
# ---------------------------------------------------------------------------
def metricas_clave(df: pd.DataFrame, objetivo: str = "total_UPDRS") -> dict:
    """Calcula los números que alimentan el resumen ejecutivo."""
    pac = resumen_por_paciente(df, objetivo)
    icc = varianza_entre_pacientes(df, [objetivo]).loc[objetivo, "entre_pacientes"]
    corr = correlaciones_con_objetivo(df, objetivo)
    top = corr.index[0]
    return {
        "n_pacientes": df[ID].nunique(),
        "n_registros": len(df),
        "registros_mediana": pac["n_registros"].median(),
        "seguimiento_mediana": pac["seguimiento_dias"].median(),
        "registros_tiempo_negativo": int((df[TIEMPO] < 0).sum()),
        "icc_objetivo": icc,
        "cambio_mediana": pac["cambio_updrs"].median(),
        "pct_empeoran": (pac["cambio_updrs"] > 0).mean(),
        "top_variable": top,
        "top_r_global": corr.loc[top, "global"],
        "top_r_dentro": corr.loc[top, "dentro_paciente"],
        "max_r_dentro": corr["dentro_paciente"].abs().max(),
        "n_pares_redundantes": len(pares_redundantes(df)),
        "n_voz": len(columnas_voz(df)),
    }


def hallazgos_clave(m: dict, objetivo: str = "total_UPDRS") -> list[str]:
    """Convierte las métricas en 4 hallazgos redactados (el texto se adapta a los números)."""
    h1 = (f"**Diseño longitudinal denso:** {m['n_pacientes']} pacientes y {m['n_registros']:,} registros "
          f"(mediana de {m['registros_mediana']:.0f} por paciente, seguidos ~{m['seguimiento_mediana']:.0f} días). "
          + ("Hay suficientes mediciones por paciente para modelos de secuencias."
             if m["registros_mediana"] >= 20 else
             "Pocas mediciones por paciente: los modelos de secuencias tienen poco margen."))

    if m["icc_objetivo"] >= 0.8:
        h2 = (f"**La gravedad es casi una 'firma' del paciente:** el {m['icc_objetivo']:.0%} de la varianza de "
              f"`{objetivo}` se explica por quién es el paciente. Dividir train/test al azar inflaría las métricas: "
              f"la partición debe ser **por paciente**.")
    else:
        h2 = (f"**Variación mixta:** el {m['icc_objetivo']:.0%} de la varianza de `{objetivo}` es entre pacientes "
              f"y el {1 - m['icc_objetivo']:.0%} dentro de cada paciente. Igual conviene particionar por paciente.")

    h3 = (f"**Progresión lenta en 6 meses:** el cambio mediano de `{objetivo}` fue de {m['cambio_mediana']:+.1f} puntos "
          f"y el {m['pct_empeoran']:.0%} de los pacientes empeoró. Predecir el *cambio* es más difícil que predecir el *nivel*.")

    if abs(m["top_r_global"]) < 0.3 and m["max_r_dentro"] < 0.3:
        matiz = ("y ninguna medida de voz por sí sola muestra una asociación fuerte, ni global ni dentro del "
                 "paciente: habrá que evaluar combinaciones de variables en los modelos.")
    elif abs(m["top_r_dentro"]) < abs(m["top_r_global"]) / 2:
        matiz = ("pero dentro de cada paciente la asociación casi desaparece: la voz distingue "
                 "*entre* pacientes más que seguir la evolución *de* cada uno.")
    else:
        matiz = "y la asociación se mantiene dentro de cada paciente, lo que sugiere valor para seguir la evolución."
    fuerza = "moderada" if abs(m["top_r_global"]) >= 0.3 else "débil"
    h4 = (f"**Asociación {fuerza} entre voz y gravedad:** la variable más correlacionada es `{m['top_variable']}` "
          f"(Spearman global {m['top_r_global']:+.2f}), {matiz} Además hay {m['n_pares_redundantes']} pares de "
          f"medidas de voz casi redundantes (|r| ≥ 0.9).")
    return [h1, h2, h3, h4]
