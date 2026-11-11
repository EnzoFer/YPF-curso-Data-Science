# VeredaSegura: análisis de daño observado en veredas junto a árboles (NYC)

**Proyecto final — Data Science / EnergIA Digital 2026.** Equipo: Enzo Fernández (`enzo21fermon@gmail.com`), Juan Ignacio Olivares (`juaniolivares95@gmail.com`) y Gabriel Quispe (`gquispe34@gmail.com`). El objetivo es estudiar si características registradas de un árbol y su entorno se asocian con que la vereda contigua haya sido registrada como dañada. La motivación social es aportar evidencia exploratoria para una eventual priorización de inspecciones municipales; el sistema no prescribe intervenciones.

## Problema, variable objetivo y alcance

Se plantea una clasificación binaria a partir de `sidewalk`: `Damage` (1) o `NoDamage` (0). Se excluye `sidewalk` de las variables predictoras. La etiqueta es el estado observado durante el relevamiento de 2015; por lo tanto **el proyecto clasifica daño observado y no pronostica cuándo se romperá una vereda**. No es causal ni reemplaza una inspección. Antes de una aplicación real habría que validar la utilidad con autoridades, residentes y datos recientes del lugar.

## Dataset y procedencia

Fuente: [NYC Open Data — 2015 Street Tree Census, Tree Data](https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh). El censo original contiene aproximadamente 683.788 registros y 45 columnas. `download_dataset.py` consulta la API pública de Socrata, conserva árboles vivos con `sidewalk` conocido y diámetro entre 1 y 60 pulgadas, y selecciona de manera aleatoria reproducible hasta 100.000 registros (semilla 42). El archivo incluido en `data/nyc_tree_census_sample.csv` tiene 100.000 filas: 71.223 `NoDamage` y 28.777 `Damage`. El script permite regenerarlo. La atribución y URL quedan documentadas aquí y en el propio script; revisar los términos del portal antes de redistribuir.

```bash
python download_dataset.py --rows 100000
```

## EDA, limpieza y transformaciones

El análisis completo se encuentra en [`Final.ipynb`](Final.ipynb): inspección de dimensiones y tipos, faltantes, proporción de clases (71,22% sin daño y 28,78% con daño en la muestra), filtros de árboles vivos con etiqueta conocida, revisión del diámetro y gráficos de relaciones. Se imputan categorías ausentes con categorías explícitas cuando el contexto lo permite; diámetros 0 o superiores a 60 se excluyen (el descargador ya aplica ese filtro a la muestra). Se crean tres puntajes (conflictos de raíces, tronco y ramas). Para gráficos exploratorios, se muestran las 20 especies más frecuentes más `Other`; el clasificador usa la especie original y aprende OneHotEncoder con agrupación de categorías infrecuentes dentro del pipeline (`min_frequency=10`), sin usar la frecuencia del conjunto test. El EDA muestra asociaciones descriptivas, no efectos causales.

## Modelo supervisado y evaluación

La notebook compara Dummy, regresión logística, árbol y Random Forest mediante `StratifiedKFold` (5 folds), optimiza el bosque con GridSearchCV y reporta matriz de confusión, F1, recall, precisión, ROC-AUC e importancia por permutación/dependencia parcial. La métrica principal es F1 de `Damage`; también se informa recall porque omitir un daño observado podría ser relevante para una inspección. `train.py` fue ejecutado sobre 100.000 filas, split estratificado 80/20 y GridSearchCV. En test (20.000 filas): F1 `Damage` 0,5480; recall 0,6149; precisión 0,4942; accuracy 0,7081 y ROC-AUC 0,7414. La matriz de confusión (orden [NoDamage, Damage]) es `[[10623, 3622], [2216, 3539]]`. El Dummy mayoritario obtuvo F1 `Damage` 0 y accuracy 0,7123. La CV F1 del mejor ajuste fue 0,5441 (`max_depth=16`, `min_samples_leaf=10`, `n_estimators=100`). Estos son resultados de la muestra y el split con semilla 42; están guardados en `artifacts/metrics.json`.

## Aprendizaje no supervisado

[`Clustering.ipynb`](Clustering.ipynb) fue ejecutada sobre una submuestra reproducible de 10.000 árboles, excluyendo la etiqueta de daño y `sidewalk`. Silhouette fue máximo para K-Means con k=3 (0,2815; k=2: 0,2774), con tamaños 670, 7.004 y 2.326; medianas de diámetro 11, 8 y 16 pulgadas, y de conflictos radiculares 0, 0 y 1. La salud modal fue `Good` en los tres grupos. DBSCAN (eps=2,5; min_samples=20) encontró 5 grupos, 1,2% de ruido y silhouette 0,3333 en puntos no-ruido. Estos resultados son exploratorios: describen similitudes bajo esta representación y parámetros, no “tipos de riesgo” validados ni grupos naturales. La separación silhouette es modesta y las variables categóricas codificadas pueden dominar las distancias.

## Entrenamiento y API FastAPI

`model.py` centraliza ingeniería de variables y pipeline. `train.py` genera el modelo y las métricas. Por defecto hace una búsqueda GridSearchCV alineada con el análisis; alternativamente puede usarse Optuna TPE:

```bash
python -m pip install -r ../requirements.txt
python download_dataset.py --rows 100000
python train.py                         # GridSearchCV
python train.py --tuning optuna --trials 6
uvicorn app:app --reload
```

La API ofrece `GET /health` y `POST /predict`; Swagger en `/docs`. El cuerpo de `POST /predict` recibe `tree_dbh`, `borough`, `curb_loc`, `health`, `spc_latin`, `steward`, `guards` y las nueve variables de conflicto (`root_stone`, `root_grate`, `root_other`, `trunk_wire`, `trnk_light`, `trnk_other`, `brch_light`, `brch_shoe`, `brch_other`). Ejemplo mínimo (los conflictos omitidos toman valor `No`):

```json
{
  "tree_dbh": 15,
  "borough": "Brooklyn",
  "curb_loc": "OnCurb",
  "health": "Good",
  "spc_latin": "Acer rubrum",
  "steward": "None",
  "guards": "None",
  "root_stone": "Yes"
}
```

La respuesta contiene una clase y una probabilidad del modelo, con advertencia de alcance. La probabilidad no está calibrada como riesgo operativo y el umbral 0,5 es demostrativo. No usar la salida para decisiones automáticas.

## Sesgos, límites y uso responsable

El censo es una fotografía histórica de NYC, no representativa de otras ciudades ni del presente. Fue relevado por voluntarios y puede contener errores, faltantes y cobertura desigual. La etiqueta `sidewalk` y parte de las variables se capturaron en el mismo relevamiento; no hay orden temporal que permita inferir causalidad o pronóstico futuro. `borough`, especie y condiciones del entorno pueden actuar como proxies de desigualdades en mantenimiento y cobertura. Priorizar recursos únicamente con este clasificador podría reforzar esas desigualdades, ignorar zonas subrepresentadas y generar falsas alarmas u omisiones. La predicción requiere inspección humana, validación por zona y subgrupo, datos actualizados, consulta comunitaria, auditorías periódicas y posibilidad de impugnar decisiones. No usar para sanciones, seguros o evaluación de residentes.

## Reproducibilidad y estructura

Requiere Python 3.10+; instalar desde la raíz con `python -m pip install -r requirements.txt`. Ejecutar `Final.ipynb` y `Clustering.ipynb` con kernel limpio tras descargar los datos. `train.py` usa semilla 42, partición estratificada y guarda artefactos bajo `artifacts/`. La API no funciona hasta que el entrenamiento genere el modelo. No se afirma que esté desplegada en la nube; una publicación pública requeriría controles de seguridad, monitoreo de drift, pruebas y datos autorizados.

## Presentación y registro de IA

La presentación oral de 15 minutos está en [`../Presentaciones/Proyecto_Final.html`](../Presentaciones/Proyecto_Final.html). Actualizar las diapositivas con los resultados reales obtenidos por el equipo. Si se utilizó una IA copiloto, declarar herramienta, finalidad, partes asistidas y cómo se verificó el resultado; la decisión analítica y revisión final corresponden al equipo.
