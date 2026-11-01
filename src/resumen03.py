"""Resumen ejecutivo y conclusiones del notebook 03 (se redactan a partir de los resultados)."""
import pandas as pd


def _lista(tabla: pd.DataFrame) -> str:
    partes = [f"`{r.variable}` ({'+' if r.coef > 0 else '−'}{abs(r.coef):.1f})" for r in tabla.itertuples()]
    return ", ".join(partes)


def metricas_clave_03(coefs: pd.DataFrame, varianzas: dict, pronostico: pd.DataFrame) -> dict:
    voz = coefs[coefs["nivel"] != "Otros"]
    semanas = coefs.set_index("termino").loc["semanas"]
    mae = pronostico.set_index("modelo")["MAE"]
    return {
        "icc_inicio": varianzas["icc_inicio"],
        "pendiente_media": semanas["coef"],
        "pendiente_ic": (semanas["ic_inf"], semanas["ic_sup"]),
        "sd_pendiente": varianzas["sd_pendiente"],
        "sig_entre": voz[(voz["nivel"] == "Entre pacientes") & voz["significativo"]],
        "sig_dentro": voz[(voz["nivel"] == "Dentro del paciente") & voz["significativo"]],
        "edad": coefs.set_index("termino").loc["edad_z"],
        "mejor_modelo": mae.idxmin(), "mejor_mae": mae.min(),
        "mae_persistencia": mae["Persistencia (no cambia)"],
        "mae_lstm": mae["LSTM · historial UPDRS + voz"],
        "mae_ridge_updrs": mae["Ridge · historial UPDRS"],
        "mae_ridge_voz": mae["Ridge · historial UPDRS + voz"],
    }


def hallazgos_clave_03(m: dict) -> list[str]:
    h1 = (f"**Cada paciente tiene su propio nivel y su propio ritmo.** Al inicio, el {m['icc_inicio']:.0%} de la "
          f"varianza no explicada por los predictores es entre pacientes. La pendiente promedio es de "
          f"{m['pendiente_media']:+.2f} puntos de UPDRS por semana (IC 95%: {m['pendiente_ic'][0]:+.2f} a "
          f"{m['pendiente_ic'][1]:+.2f}), pero varía entre pacientes con un desvío de {m['sd_pendiente']:.2f} puntos/semana.")

    if len(m["sig_entre"]):
        h2 = (f"**Predictores entre pacientes** (p < 0,05; puntos de UPDRS por cada desvío estándar): "
              f"{_lista(m['sig_entre'])}. Estos rasgos distinguen a pacientes más y menos graves.")
    else:
        h2 = "**Ninguna medida de voz distingue de forma significativa a pacientes más y menos graves** una vez ajustado por edad y sexo."

    if len(m["sig_dentro"]):
        h3 = (f"**Predictores longitudinales (dentro del paciente):** {_lista(m['sig_dentro'])}. "
              f"Ojo con la magnitud: son cambios chicos y la UPDRS intermedia está interpolada.")
    else:
        h3 = ("**Ninguna medida de voz funciona como predictor longitudinal:** cuando la voz de un paciente cambia "
              "respecto de su propio promedio, su UPDRS no cambia de forma significativa.")

    titulo = ("**La LSTM es el mejor modelo de pronóstico:**" if m["mejor_modelo"].startswith("LSTM")
              else "**El deep learning no supera a una regla simple:**")
    h4 = (f"{titulo} pronosticando 4 semanas hacia adelante en pacientes nuevos, "
          f"el mejor modelo es *{m['mejor_modelo']}* (MAE {m['mejor_mae']:.2f}). La LSTM obtiene {m['mae_lstm']:.2f}, "
          f"y agregar voz a Ridge pasa de {m['mae_ridge_updrs']:.2f} a {m['mae_ridge_voz']:.2f}. "
          f"Con 42 pacientes y una UPDRS interpolada, la estructura de los datos favorece a la extrapolación lineal.")
    return [h1, h2, h3, h4]
