"""Experimentos de modelado para datos longitudinales.

Regla de oro: en datos con varias mediciones por paciente, la validación tiene que simular el uso real.
Si el modelo se va a usar con pacientes NUEVOS, ningún paciente puede estar a la vez en train y en test.

Métricas: se calculan sobre las predicciones "fuera de muestra" de todos los folds juntos
(cross_val_predict), que es más estable que promediar el R² de cada fold cuando los folds son
grupos de pacientes muy distintos entre sí.
"""
import warnings

import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis import ID, TIEMPO, columnas_voz, varianza_entre_pacientes

SEMILLA = 42
N_FOLDS = 5
DEMOGRAFICAS = ["age", "sex"]


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def random_forest():
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEMILLA)


def gradient_boosting():
    """XGBoost si está instalado; si no, el equivalente de scikit-learn."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=SEMILLA, n_jobs=-1)
    except ImportError:
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                             random_state=SEMILLA)


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def evaluar(modelo, X: pd.DataFrame, y: pd.Series, grupos: pd.Series, cv) -> dict:
    """Validación cruzada: devuelve MAE, R² y las predicciones fuera de muestra."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KFold avisa que ignora 'groups': es esperado
        pred = cross_val_predict(modelo, X, y, cv=cv, groups=grupos)
    return {"MAE": mean_absolute_error(y, pred), "R2": r2_score(y, pred), "pred": pred}


def particion_aleatoria():
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)


def particion_por_paciente():
    return GroupKFold(n_splits=N_FOLDS)


# ---------------------------------------------------------------------------
# Experimento 1: fuga de información
# ---------------------------------------------------------------------------
def experimento_fuga(df: pd.DataFrame, objetivo: str = "total_UPDRS"):
    """Mismo modelo y mismos datos; solo cambia cómo se dividen train y test."""
    X = df[columnas_voz(df) + DEMOGRAFICAS]
    y, grupos = df[objetivo], df[ID]
    filas, preds = [], []
    for nombre, cv in [("Aleatoria (registros mezclados)", particion_aleatoria()),
                       ("Por paciente (GroupKFold)", particion_por_paciente())]:
        r = evaluar(random_forest(), X, y, grupos, cv)
        filas.append({"particion": nombre, "MAE": r["MAE"], "R2": r["R2"]})
        preds.append(pd.DataFrame({"real": y.values, "predicho": r["pred"],
                                   "particion": nombre, ID: grupos.values}))
    return pd.DataFrame(filas), pd.concat(preds, ignore_index=True)


# ---------------------------------------------------------------------------
# Experimento 2: predecir el nivel de gravedad de un paciente nuevo
# ---------------------------------------------------------------------------
def agregar_semanal(df: pd.DataFrame) -> pd.DataFrame:
    """Promedia las grabaciones de cada paciente por semana para reducir el ruido de la voz."""
    voz = columnas_voz(df)
    d = df.assign(semana=(df[TIEMPO] // 7).astype(int))
    agg = {c: "mean" for c in voz + ["motor_UPDRS", "total_UPDRS", TIEMPO]}
    agg.update({"age": "first", "sex": "first"})
    return d.groupby([ID, "semana"]).agg(agg).reset_index()


def experimento_nivel(df: pd.DataFrame, objetivo: str = "total_UPDRS") -> pd.DataFrame:
    voz = columnas_voz(df)
    modelos = [
        ("Media (sin información)", DummyRegressor(), DEMOGRAFICAS),
        ("Edad + sexo · Ridge", ridge(), DEMOGRAFICAS),
        ("Voz + edad + sexo · Ridge", ridge(), voz + DEMOGRAFICAS),
        ("Voz + edad + sexo · Random Forest", random_forest(), voz + DEMOGRAFICAS),
        ("Voz + edad + sexo · Gradient Boosting", gradient_boosting(), voz + DEMOGRAFICAS),
    ]
    filas = []
    for etiqueta_datos, datos in [("Cada grabación", df), ("Promedio semanal", agregar_semanal(df))]:
        for nombre, modelo, cols in modelos:
            r = evaluar(modelo, datos[cols], datos[objetivo], datos[ID], particion_por_paciente())
            filas.append({"modelo": nombre, "datos": etiqueta_datos, "MAE": r["MAE"], "R2": r["R2"]})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Experimento 3: predecir el CAMBIO respecto de la línea de base del propio paciente
# ---------------------------------------------------------------------------
def preparar_cambio(df: pd.DataFrame, objetivo: str = "total_UPDRS", dias_base: int = 14):
    """Usa los primeros `dias_base` días de cada paciente como su línea de base.

    - Objetivo: cambio en la UPDRS respecto de su promedio en la línea de base.
    - Predictores: cambio en cada medida de voz respecto de su línea de base + días transcurridos.
    Simula el uso clínico: conozco al paciente al inicio y quiero seguir su evolución.
    """
    voz = columnas_voz(df)
    d = df.copy()
    d["dias"] = d[TIEMPO] - d.groupby(ID)[TIEMPO].transform("min")
    es_base = d["dias"] <= dias_base
    base = d[es_base].groupby(ID)[voz + [objetivo]].mean()

    seguimiento = d[~es_base].copy()
    seguimiento["delta_objetivo"] = seguimiento[objetivo] - seguimiento[ID].map(base[objetivo])
    for c in voz:
        seguimiento[f"delta_{c}"] = seguimiento[c] - seguimiento[ID].map(base[c])
    deltas_voz = [f"delta_{c}" for c in voz]
    return seguimiento, deltas_voz


def experimento_cambio(df: pd.DataFrame, objetivo: str = "total_UPDRS") -> pd.DataFrame:
    d, deltas_voz = preparar_cambio(df, objetivo)
    modelos = [
        ("Sin cambio (Δ = 0)", DummyRegressor(strategy="constant", constant=0), ["dias"]),
        ("Cambio promedio", DummyRegressor(), ["dias"]),
        ("Tendencia temporal · Lineal", LinearRegression(), ["dias"]),
        ("Δ voz + días · Ridge", ridge(), deltas_voz + ["dias"]),
        ("Δ voz + días + edad + sexo · Random Forest", random_forest(), deltas_voz + ["dias"] + DEMOGRAFICAS),
    ]
    filas = []
    for nombre, modelo, cols in modelos:
        r = evaluar(modelo, d[cols], d["delta_objetivo"], d[ID], particion_por_paciente())
        filas.append({"modelo": nombre, "MAE": r["MAE"], "R2": r["R2"]})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Experimento 4: ¿qué aprende el modelo con partición aleatoria?
# ---------------------------------------------------------------------------
def importancia_vs_firma(df: pd.DataFrame, objetivo: str = "total_UPDRS") -> pd.DataFrame:
    """Importancia por permutación de un Random Forest evaluado con partición aleatoria,
    junto a qué tan 'propia del paciente' es cada variable (proporción de varianza entre pacientes).
    """
    cols = columnas_voz(df) + DEMOGRAFICAS
    X, y = df[cols], df[objetivo]
    train = X.sample(frac=0.8, random_state=SEMILLA).index
    test = X.index.difference(train)
    modelo = random_forest().fit(X.loc[train], y.loc[train])
    # n_jobs=1: el Random Forest ya predice en paralelo. Paralelizar también la permutación
    # copia el modelo en cada proceso y en Windows puede agotar la memoria (TerminatedWorkerError).
    imp = permutation_importance(modelo, X.loc[test], y.loc[test], n_repeats=5,
                                 random_state=SEMILLA, n_jobs=1)
    firma = varianza_entre_pacientes(df, cols)["entre_pacientes"]
    res = pd.DataFrame({"importancia": imp.importances_mean, "varianza_entre_pacientes": firma.reindex(cols).values},
                       index=pd.Index(cols, name="variable"))
    return res.sort_values("importancia", ascending=False)


# ---------------------------------------------------------------------------
# Todo junto + resumen ejecutivo
# ---------------------------------------------------------------------------
def correr_experimentos(df: pd.DataFrame, objetivo: str = "total_UPDRS") -> dict:
    print("1/4 Fuga de información...")
    fuga, preds = experimento_fuga(df, objetivo)
    print("2/4 Nivel de gravedad en pacientes nuevos...")
    nivel = experimento_nivel(df, objetivo)
    print("3/4 Cambio respecto de la línea de base...")
    cambio = experimento_cambio(df, objetivo)
    print("4/4 Importancia de variables...")
    importancia = importancia_vs_firma(df, objetivo)
    print("Listo.")
    return {"fuga": fuga, "predicciones_fuga": preds, "nivel": nivel, "cambio": cambio,
            "importancia": importancia}


def metricas_clave_02(res: dict) -> dict:
    fuga = res["fuga"].set_index("particion")
    nivel, cambio, imp = res["nivel"], res["cambio"], res["importancia"]
    referencia = nivel[(nivel["modelo"] == "Media (sin información)") & (nivel["datos"] == "Cada grabación")].iloc[0]
    mejor_nivel = nivel.loc[nivel["MAE"].idxmin()]
    mejor_cambio = cambio.loc[cambio["MAE"].idxmin()]
    sin_cambio = cambio[cambio["modelo"] == "Sin cambio (Δ = 0)"].iloc[0]
    con_voz = cambio["modelo"].str.startswith("Δ voz")
    voz_cambio = cambio[con_voz]["MAE"].min()
    simple_cambio = cambio[~con_voz]["MAE"].min()
    rho = spearmanr(imp["importancia"], imp["varianza_entre_pacientes"]).statistic
    return {
        "r2_aleatoria": fuga.iloc[0]["R2"], "r2_paciente": fuga.iloc[1]["R2"],
        "mae_aleatoria": fuga.iloc[0]["MAE"], "mae_paciente": fuga.iloc[1]["MAE"],
        "mae_referencia": referencia["MAE"],
        "mejor_nivel_modelo": f"{mejor_nivel['modelo']} ({mejor_nivel['datos'].lower()})",
        "mejor_nivel_mae": mejor_nivel["MAE"], "mejor_nivel_r2": mejor_nivel["R2"],
        "mejor_cambio_modelo": mejor_cambio["modelo"], "mejor_cambio_mae": mejor_cambio["MAE"],
        "sin_cambio_mae": sin_cambio["MAE"], "voz_cambio_mae": voz_cambio, "simple_cambio_mae": simple_cambio,
        "top_variable": imp.index[0], "top_importancia_pct": imp["importancia"].iloc[0] / imp["importancia"].clip(lower=0).sum(),
        "rho_importancia_firma": rho,
    }


def hallazgos_clave_02(m: dict) -> list[str]:
    h1 = (f"**La partición aleatoria engaña:** el mismo Random Forest obtiene R² = {m['r2_aleatoria']:.2f} "
          f"(MAE {m['mae_aleatoria']:.1f} puntos) con registros mezclados, pero R² = {m['r2_paciente']:.2f} "
          f"(MAE {m['mae_paciente']:.1f}) cuando se evalúa en pacientes que no vio. La diferencia es pura fuga de información.")

    mejora = 1 - m["mejor_nivel_mae"] / m["mae_referencia"]
    if mejora < 0.05:
        if m["mejor_nivel_modelo"].startswith("Media"):
            mejor_txt = f"el menor error ({m['mejor_nivel_mae']:.1f}) lo obtiene justamente predecir el promedio"
        else:
            mejor_txt = f"el mejor, {m['mejor_nivel_modelo']}, obtiene {m['mejor_nivel_mae']:.1f}"
        h2 = (f"**Para un paciente nuevo, ningún modelo supera de forma relevante a predecir el promedio** "
              f"(MAE de referencia {m['mae_referencia']:.1f} puntos; {mejor_txt}). "
              f"Con 42 pacientes, la voz no alcanza para estimar el nivel de gravedad de alguien desconocido.")
    else:
        h2 = (f"**Hay señal para pacientes nuevos:** el mejor modelo ({m['mejor_nivel_modelo']}) baja el MAE de "
              f"{m['mae_referencia']:.1f} a {m['mejor_nivel_mae']:.1f} puntos ({mejora:.0%} de mejora sobre predecir el promedio).")

    h3 = (f"**Con partición aleatoria el modelo 'memoriza pacientes':** la variable más importante es `{m['top_variable']}` "
          f"({m['top_importancia_pct']:.0%} de la importancia total), y la importancia de cada variable se correlaciona "
          f"con qué tan propia del paciente es (Spearman {m['rho_importancia_firma']:+.2f}). Aprende a reconocer quién es, no qué predice la gravedad.")

    reduccion = 1 - m["mejor_cambio_mae"] / m["mae_referencia"]
    if m["voz_cambio_mae"] > m["simple_cambio_mae"] * 0.95:
        voz_txt = (f"Los cambios en la voz no mejoran a una regla simple sin voz "
                   f"(MAE {m['voz_cambio_mae']:.1f} vs. {m['simple_cambio_mae']:.1f}).")
    else:
        voz_txt = (f"Los cambios en la voz sí aportan: MAE {m['voz_cambio_mae']:.1f} vs. "
                   f"{m['simple_cambio_mae']:.1f} de la mejor regla simple sin voz.")
    h4 = (f"**Conocer la línea de base del paciente es lo que más ayuda:** prediciendo el cambio respecto de sus primeras "
          f"2 semanas, el error baja a {m['mejor_cambio_mae']:.1f} puntos ({reduccion:.0%} menos que el modelo de referencia "
          f"para pacientes nuevos). {voz_txt}")
    return [h1, h2, h3, h4]
