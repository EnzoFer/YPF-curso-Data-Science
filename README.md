# VeredaSegura — patrones de daño observado en veredas y arbolado urbano

Proyecto integrador del curso Data Science / EnergIA Digital 2026. Analizamos el censo de arbolado urbano de NYC para explorar qué características de árboles y entorno se asocian con **daño observado** en la vereda, entrenar un clasificador educativo y explorar perfiles de arbolado sin supervisión.

> **Alcance responsable:** el relevamiento es una fotografía de 2015 y la etiqueta describe el estado de la vereda registrado en ese relevamiento. El modelo **no predice daño futuro**, no establece causalidad ni debe decidir automáticamente dónde intervenir. Sólo podría servir como demostración para priorizar una revisión humana, previa validación con datos locales actuales.

## Entregables del proyecto

- [`Final/README.md`](Final/README.md): problema, fuente, metodología, resultados, sesgos, instrucciones y diseño de API.
- [`Final/Final.ipynb`](Final/Final.ipynb): limpieza, EDA, feature engineering, clasificación, validación, GridSearchCV, métricas y explicabilidad.
- [`Final/Clustering.ipynb`](Final/Clustering.ipynb): K-Means, silhouette, perfiles exploratorios y contraste opcional con DBSCAN.
- [`Final/model.py`](Final/model.py), [`Final/train.py`](Final/train.py), [`Final/app.py`](Final/app.py): ingeniería de variables, entrenamiento reproducible y API FastAPI.
- [`Final/download_dataset.py`](Final/download_dataset.py): descarga una muestra reproducible de NYC Open Data a `Final/data/`.
- [`Presentaciones/Proyecto_Final.html`](Presentaciones/Proyecto_Final.html): presentación de 15 minutos, navegable con teclado.
- `requirements.txt`: entorno reproducible para notebooks, entrenamiento y API.

## Inicio rápido

Requiere Python 3.10 o posterior y acceso a internet para obtener los datos. Desde la raíz del repositorio:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python Final/download_dataset.py --rows 100000
python Final/train.py
uvicorn Final.app:app --reload
```

La API queda en `http://127.0.0.1:8000`; documentación interactiva en `/docs` y estado del modelo en `/health`. Para usar Optuna en vez de GridSearchCV: `python Final/train.py --tuning optuna --trials 6`.

## Orden sugerido para reproducir

1. Ejecutar `Final/download_dataset.py` para descargar `Final/data/nyc_tree_census_sample.csv` (muestra aleatoria reproducible de hasta 100.000 registros elegibles, semilla 42).
2. Abrir y ejecutar `Final/Final.ipynb` para revisar el EDA y los resultados supervisados originales.
3. Abrir `Final/Clustering.ipynb` para ejecutar la cuarta pre-entrega no supervisada.
4. Ejecutar `python Final/train.py` para producir `Final/artifacts/sidewalk_damage_pipeline.joblib` y `metrics.json` usados por la API.
5. Levantar FastAPI y probar `/docs` con un ejemplo de árbol (la respuesta advierte el alcance observacional).

El CSV de muestra se incluye en `Final/data/` para que el repositorio tenga el dataset de trabajo; `download_dataset.py` permite regenerarlo. Los artefactos del modelo se generan localmente y están ignorados por Git. No subir credenciales ni información personal.

## Estado de resultados

Entrenamiento modular ejecutado sobre la muestra incluida (semilla 42, test estratificado de 20.000 registros): F1 `Damage` 0,548; recall 0,615; precisión 0,494; accuracy 0,708; ROC-AUC 0,741. La línea base de clase mayoritaria obtuvo accuracy 0,712 y F1 `Damage` 0. Estos números están guardados en `Final/artifacts/metrics.json`; comparar accuracy con la línea base sin tener en cuenta el recall/F1 ocultaría el desempeño nulo de ésta en la clase minoritaria. La notebook supervisada fue re-ejecutada sobre la misma muestra.

## Contexto de cursada y equipo

El proyecto integra EDA, tratamiento de faltantes/outliers, feature engineering, clasificación, métricas, validación cruzada, búsqueda de hiperparámetros, interpretabilidad, clustering y puesta en servicio con FastAPI. El trabajo se organiza en equipo conforme a la cursada; sus integrantes están identificados en `Final/README.md`.
