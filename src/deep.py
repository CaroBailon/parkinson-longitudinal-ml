"""Pronóstico a varias semanas con ventanas temporales: baselines, Ridge y LSTM (PyTorch).

Tarea: con las últimas `LARGO` semanas de un paciente (voz + UPDRS), predecir su UPDRS
`HORIZONTE` semanas después. Todos los modelos predicen el CAMBIO respecto de la última semana
observada y se evalúan con GroupKFold (pacientes nuevos en test).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis import ID
from src.longitudinal import VOZ_SELECCION

LARGO = 4       # semanas de historia que ve el modelo
HORIZONTE = 4   # semanas hacia adelante que predice
SEMILLA = 42
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Ventanas
# ---------------------------------------------------------------------------
def crear_ventanas(semanal: pd.DataFrame, objetivo: str = "total_UPDRS",
                   largo: int = LARGO, horizonte: int = HORIZONTE) -> dict:
    """Recorre la serie semanal de cada paciente y arma ventanas (historia -> valor futuro).

    Devuelve un diccionario con:
      X: array (n_ventanas, largo, n_variables) con [UPDRS, voz...] de cada semana de la historia
      y: UPDRS real `horizonte` semanas después de la última semana de la historia
      ultimo: UPDRS de la última semana de la historia (punto de partida de la predicción)
      grupos: paciente de cada ventana

    Algunos pacientes tienen semanas sin grabaciones. Para que "4 semanas de historia" y
    "4 semanas hacia adelante" signifiquen exactamente eso, solo se arman ventanas cuya historia
    son semanas consecutivas y cuyo objetivo está a exactamente `horizonte` semanas de la última.
    (Si se usara el orden de las filas, un hueco convertiría el horizonte de 4 en 5 a 9 semanas.)
    """
    columnas = [objetivo] + list(VOZ_SELECCION)
    X, y, ultimo, grupos = [], [], [], []
    for pid, g in semanal.sort_values([ID, "semana"]).groupby(ID):
        valores = g[columnas].to_numpy(dtype=float)
        posicion = {s: i for i, s in enumerate(g["semana"])}   # semana -> fila
        for s0 in g["semana"]:
            historia = [posicion.get(s0 + k) for k in range(largo)]
            futuro = posicion.get(s0 + largo - 1 + horizonte)
            if None in historia or futuro is None:
                continue
            X.append(valores[historia])
            ultimo.append(valores[historia[-1], 0])
            y.append(valores[futuro, 0])
            grupos.append(pid)
    return {"X": np.array(X), "y": np.array(y), "ultimo": np.array(ultimo),
            "grupos": np.array(grupos), "columnas": columnas}


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def predecir_persistencia(X_tr, d_tr, X_te, horizonte=HORIZONTE):
    """'Nada cambia': el cambio predicho es 0."""
    return np.zeros(len(X_te))


def predecir_extrapolacion(X_tr, d_tr, X_te, horizonte=HORIZONTE):
    """Prolonga la pendiente de la UPDRS dentro de la ventana."""
    t = np.arange(X_te.shape[1])
    pendientes = np.polyfit(t, X_te[:, :, 0].T, 1)[0]
    return pendientes * horizonte


def _ridge(X_tr, d_tr, X_te, columnas_idx):
    modelo = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    plano = lambda X: X[:, :, columnas_idx].reshape(len(X), -1)
    modelo.fit(plano(X_tr), d_tr)
    return modelo.predict(plano(X_te))


def predecir_ridge_updrs(X_tr, d_tr, X_te, horizonte=HORIZONTE):
    return _ridge(X_tr, d_tr, X_te, [0])


def predecir_ridge_voz(X_tr, d_tr, X_te, horizonte=HORIZONTE):
    return _ridge(X_tr, d_tr, X_te, list(range(X_tr.shape[2])))


def predecir_lstm(X_tr, d_tr, X_te, horizonte=HORIZONTE, epocas=80, oculto=16):
    """LSTM chica: lee la secuencia semana a semana y predice el cambio de UPDRS."""
    import torch
    from torch import nn

    torch.manual_seed(SEMILLA)
    # Estandarizar con estadísticas SOLO del entrenamiento (evita fuga de información)
    media = X_tr.reshape(-1, X_tr.shape[2]).mean(axis=0)
    desvio = X_tr.reshape(-1, X_tr.shape[2]).std(axis=0) + 1e-8
    escala_y = d_tr.std() + 1e-8
    a_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    Xtr, Xte = a_tensor((X_tr - media) / desvio), a_tensor((X_te - media) / desvio)
    ytr = a_tensor(d_tr / escala_y).unsqueeze(1)

    class RedLSTM(nn.Module):
        def __init__(self, n_variables):
            super().__init__()
            self.lstm = nn.LSTM(n_variables, oculto, batch_first=True)
            self.salida = nn.Linear(oculto, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)          # h: estado final después de leer toda la secuencia
            return self.salida(h[-1])

    red = RedLSTM(X_tr.shape[2])
    optimizador = torch.optim.Adam(red.parameters(), lr=5e-3, weight_decay=1e-4)
    perdida = nn.L1Loss()                     # optimiza directamente el error absoluto (MAE)
    generador = torch.Generator().manual_seed(SEMILLA)

    red.train()
    for _ in range(epocas):
        orden = torch.randperm(len(Xtr), generator=generador)
        for i in range(0, len(Xtr), 64):
            lote = orden[i:i + 64]
            optimizador.zero_grad()
            loss = perdida(red(Xtr[lote]), ytr[lote])
            loss.backward()
            optimizador.step()

    red.eval()
    with torch.no_grad():
        return red(Xte).squeeze(1).numpy() * escala_y


MODELOS = {
    "Persistencia (no cambia)": predecir_persistencia,
    "Extrapolación lineal": predecir_extrapolacion,
    "Ridge · historial UPDRS": predecir_ridge_updrs,
    "Ridge · historial UPDRS + voz": predecir_ridge_voz,
    "LSTM · historial UPDRS + voz": predecir_lstm,
}


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def evaluar_pronostico(ventanas: dict, modelos: dict = MODELOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GroupKFold por paciente. Devuelve (métricas, predicciones fuera de muestra)."""
    X, y, ultimo, grupos = ventanas["X"], ventanas["y"], ventanas["ultimo"], ventanas["grupos"]
    delta = y - ultimo
    cv = GroupKFold(n_splits=N_FOLDS)
    preds = {nombre: np.zeros(len(y)) for nombre in modelos}

    for k, (tr, te) in enumerate(cv.split(X, delta, groups=grupos), 1):
        print(f"  fold {k}/{N_FOLDS}")
        for nombre, funcion in modelos.items():
            preds[nombre][te] = ultimo[te] + funcion(X[tr], delta[tr], X[te])

    metricas = pd.DataFrame([
        {"modelo": nombre, "MAE": mean_absolute_error(y, p),
         "MAE_cambio_real_grande": mean_absolute_error(y[np.abs(delta) >= 2], p[np.abs(delta) >= 2])}
        for nombre, p in preds.items()
    ])
    predicciones = pd.DataFrame({"paciente": grupos, "real": y, "ultimo": ultimo, **preds})
    return metricas, predicciones
