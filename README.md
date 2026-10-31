# Predictores longitudinales en Parkinson con Machine Learning

Proyecto de práctica orientado a identificar predictores de progresión en la enfermedad de Parkinson a partir de datos clínicos longitudinales (mediciones repetidas por paciente).

## Datos
**Parkinson's Telemonitoring (UCI)** — 42 pacientes con Parkinson temprano, ~200 registros por paciente durante 6 meses.
Tsanas, A. & Little, M. (2009). UCI Machine Learning Repository. https://doi.org/10.24432/C5ZS3N — Licencia CC BY 4.0.

## Enfoque metodológico
- Partición de datos **por paciente** (evitar fuga de información entre train y test)
- Ingeniería de variables temporales (baseline, pendientes, cambios entre visitas)
- Modelos mixtos para medidas repetidas
- Machine Learning (Random Forest, XGBoost) e interpretabilidad con SHAP
- Deep Learning para secuencias (LSTM) como comparación

## Estructura
- `data/` — datos (no versionados, se descargan con código)
- `notebooks/` — análisis paso a paso
- `src/` — funciones reutilizables
- `reports/` — figuras y conclusiones

## Autora
Carolina Bailon Trujillo — Estudiante de Ciencia de Datos en Organizaciones (UNLP)
