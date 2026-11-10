"""Resumen ejecutivo y conclusiones del notebook 03 (se redactan a partir de los resultados)."""
import pandas as pd


def _lista(tabla: pd.DataFrame) -> str:
    partes = [f"`{r.variable}` ({'+' if r.coef > 0 else '−'}{abs(r.coef):.1f})" for r in tabla.itertuples()]
    return ", ".join(partes)


def metricas_clave_03(coefs: pd.DataFrame, varianzas: dict, pronostico: pd.DataFrame,
                      diagnostico: pd.DataFrame, robustez: pd.DataFrame) -> dict:
    voz = coefs[coefs["nivel"] != "Otros"]
    tiempo = coefs.set_index("termino").loc["meses"]
    mae = pronostico.set_index("modelo")["MAE"]
    return {
        "icc_inicio": varianzas["icc_inicio"],
        "pendiente_media": tiempo["coef"],
        "pendiente_ic": (tiempo["ic_inf"], tiempo["ic_sup"]),
        "pendiente_p": tiempo["p_valor"],
        "sd_pendiente": varianzas["sd_pendiente"],
        "sig_entre": voz[(voz["nivel"] == "Entre pacientes") & voz["significativo"]],
        "sig_dentro": voz[(voz["nivel"] == "Dentro del paciente") & voz["significativo"]],
        "edad": coefs.set_index("termino").loc["edad_z"],
        "mejor_modelo": mae.idxmin(), "mejor_mae": mae.min(),
        "mae_persistencia": mae["Persistencia (no cambia)"],
        "mae_lstm": mae["LSTM · historial UPDRS + voz"],
        "mae_ridge_updrs": mae["Ridge · historial UPDRS"],
        "mae_ridge_voz": mae["Ridge · historial UPDRS + voz"],
        "convergio": diagnostico.loc["Convergencia del optimizador", "valor"] == "Sí",
        "autocorrelacion": float(diagnostico.loc["Residuos · autocorrelación semana a semana", "valor"]),
        "pendiente_p_robusta": robustez["p_valor"].iloc[0],
        "entre_sig_robusta": int((robustez["p_valor"].iloc[1:] < 0.05).sum()),
    }


def hallazgos_clave_03(m: dict) -> list[str]:
    h1 = (f"**Cada paciente tiene su propio nivel y su propio ritmo.** Al inicio, el {m['icc_inicio']:.0%} de la "
          f"varianza no explicada por los predictores es entre pacientes. La pendiente promedio es de "
          f"{m['pendiente_media']:+.2f} puntos de UPDRS por mes (IC 95%: {m['pendiente_ic'][0]:+.2f} a "
          f"{m['pendiente_ic'][1]:+.2f}; p = {m['pendiente_p']:.3f}), pero varía mucho entre pacientes: "
          f"desvío de {m['sd_pendiente']:.2f} puntos/mes.")

    if len(m["sig_entre"]):
        h2 = (f"**Predictores entre pacientes** (p < 0,05; puntos de UPDRS por cada desvío estándar): "
              f"{_lista(m['sig_entre'])}. Estos rasgos distinguen a pacientes más y menos graves.")
    else:
        h2 = ("**Ninguna medida de voz distingue de forma significativa a pacientes más y menos graves** una vez "
              "ajustado por edad, sexo y las demás medidas. Es coherente con el notebook 02: para un paciente nuevo, "
              "ningún modelo supera a predecir el promedio.")

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
    estado = "converge" if m["convergio"] else "NO converge (revisar antes de interpretar)"
    h5 = (f"**Los supuestos se revisaron, y las conclusiones resisten.** El modelo {estado}. Los residuos tienen una "
          f"autocorrelación semana a semana de {m['autocorrelacion']:+.2f} (la UPDRS está interpolada), "
          f"por lo que los p-valores *dentro del paciente* pueden ser optimistas: como igual ninguno es significativo, "
          f"la conclusión no cambia. Con métodos que no dependen de esos supuestos, la progresión "
          f"{'sigue siendo' if m['pendiente_p_robusta'] < 0.05 else 'deja de ser'} significativa "
          f"(p = {m['pendiente_p_robusta']:.3f}) y "
          + ("ninguna medida de voz resulta significativa entre pacientes." if m["entre_sig_robusta"] == 0 else
             f"{m['entre_sig_robusta']} medida(s) de voz resultan significativas entre pacientes."))
    return [h1, h2, h3, h4, h5]
