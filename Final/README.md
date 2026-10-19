# Predicción de Daño en Veredas por Arbolado Urbano (NYC)

**Proyecto Final – Curso de Data Science / IA – Fundación YPF**

| | |
|---|---|
| **Grupo N°** | _completar_ |
| **Integrantes** | _Nombre Apellido 1_, _Nombre Apellido 2_, _Nombre Apellido 3_ |
| **Notebook principal** | [`Final.ipynb`](Final.ipynb) |

---

## 1. Descripción del problema

Las raíces de los árboles de vereda levantan y rompen las baldosas, generando riesgo de caídas para los peatones, reclamos legales y costos de reparación para la ciudad. Hoy las inspecciones se hacen de forma reactiva (cuando alguien reclama).

**Objetivo de negocio:** dar a los departamentos de obras públicas una herramienta para **priorizar inspecciones** de veredas según las características de cada árbol.

**Pregunta que responde el modelo:**
> Dadas las características físicas de un árbol vivo (diámetro, especie, salud), su entorno (barrio, ubicación respecto al cordón, protecciones) y los conflictos observados en raíces, tronco y ramas, **¿la vereda junto a ese árbol presenta daño?**

- **Tipo de problema:** Clasificación binaria supervisada.
- **Por qué es Machine Learning y no un cálculo:** el estado de la vereda (`sidewalk`) es una observación independiente; no puede obtenerse mediante ninguna fórmula a partir de las demás columnas.

---

## 2. Dataset

- **Nombre:** 2015 Street Tree Census – Tree Data (NYC Parks)
- **Fuente:** [NYC Open Data](https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh/about_data)
- **Tamaño original:** 683.788 filas × 45 columnas (un registro por árbol).
- **Tamaño tras la limpieza:** ~652.000 árboles vivos con estado de vereda registrado. Para el modelado se usa un submuestreo estratificado de 100.000 registros (80% train / 20% test).

> El archivo CSV (~285 MB) no se sube al repositorio por su tamaño. Hay que descargarlo desde el link y guardarlo en esta carpeta con el nombre `2015_Street_Tree_Census_-_Tree_Data_20260914.csv`.

---

## 3. Variable objetivo (y)

| Columna | Origen | Valores | Distribución |
|---|---|---|---|
| `target` | Derivada de `sidewalk` | `1` = Damage (vereda dañada), `0` = NoDamage | ~29% / ~71% (desbalance moderado) |

La columna `sidewalk` original y cualquier columna derivada de ella se **excluyen de X** para evitar fuga de información (target leakage).

---

## 4. Variables predictoras (X)

| Variable | Tipo | Descripción | Unidad / Valores |
|---|---|---|---|
| `tree_dbh` | Numérica | Diámetro del tronco medido a 1,37 m del suelo | Pulgadas (1 a 60) |
| `root_conflict_score` | Numérica (creada) | Cantidad de problemas en raíces: piedras, rejillas u otros (`root_stone`, `root_grate`, `root_other`) | 0 a 3 |
| `trunk_conflict_score` | Numérica (creada) | Cantidad de problemas en tronco: cables, luces u otros (`trunk_wire`, `trnk_light`, `trnk_other`) | 0 a 3 |
| `branch_conflict_score` | Numérica (creada) | Cantidad de problemas en ramas: luces, zapatillas colgadas u otros (`brch_light`, `brch_shoe`, `brch_other`) | 0 a 3 |
| `steward` | Ordinal | Cantidad de señales de cuidado vecinal | None < 1or2 < 3or4 < 4orMore |
| `health` | Ordinal | Estado de salud percibido del árbol | Poor < Fair < Good |
| `borough` | Nominal | Distrito de NYC | Manhattan, Bronx, Brooklyn, Queens, Staten Island |
| `guards` | Nominal | Tipo de protección alrededor del árbol | None, Helpful, Harmful, Unsure |
| `curb_loc` | Nominal | Ubicación respecto al cordón | OnCurb, OffsetFromCurb |
| `spc_clean` | Nominal (creada) | Especie (nombre científico): las 20 más frecuentes + `Other` | 21 categorías |

**Columnas descartadas:** identificadores (`tree_id`, `block_id`), coordenadas y códigos administrativos (sin señal analítica o redundantes), fechas y datos del relevamiento, y los árboles muertos o tocones (no tienen dato de `sidewalk`).

---

## 5. Metodología

1. **Limpieza:** filtrado de árboles vivos con target registrado, imputación de nulos con significado (`steward`, `guards` → `'None'`) y eliminación de diámetros imposibles (0 o > 60").
2. **EDA:** relación del daño con el diámetro, los conflictos de raíz y la especie.
3. **Feature engineering:** puntajes de conflicto y reducción de la cantidad de especies.
4. **Partición:** `train_test_split` estratificado; el test se separa **una sola vez** al inicio y no se usa hasta la evaluación final.
5. **Preprocesamiento en `Pipeline` + `ColumnTransformer`** (escalado, codificación ordinal y One-Hot) para evitar fugas entre folds.
6. **Métrica rectora: F1-Score de la clase Damage.** No se usa Accuracy porque las clases están desbalanceadas: un modelo que nunca detecta daño ya acierta el 71%. Una vereda rota no detectada (falso negativo) es el error más costoso.
7. **Comparación multimodelo con `StratifiedKFold` (5 folds):** Dummy Classifier, Regresión Logística, Árbol de Decisión y Random Forest, reportando métricas de train y de validación.
8. **Optimización del modelo final (Random Forest)** con `GridSearchCV`.
9. **Evaluación final en test:** comparación train vs test (sobreajuste), matriz de confusión, ROC-AUC y contraste contra el Dummy.
10. **Interpretabilidad:** importancia por permutación y dependencia parcial.

---

## 6. Resultados

| Modelo | F1 Damage (CV) | F1 Damage (Test) | Recall Damage (Test) | ROC-AUC (Test) |
|---|---|---|---|---|
| Dummy (most_frequent) | 0.000 | 0.000 | 0.000 | 0.500 |
| Árbol de Decisión (max_depth=6) | 0.540 | – | – | – |
| Regresión Logística | 0.541 | – | – | – |
| **Random Forest optimizado** | **0.548** | **0.543** | **0.606** | **0.737** |

- Los modelos que no se eligieron solo se evaluaron con validación cruzada. El test se usó **una sola vez**, con el modelo final.
- **Hiperparámetros finales:** `n_estimators=200`, `max_depth=16`, `min_samples_leaf=10`, `class_weight='balanced'`.
- **Sobreajuste:** el F1 da 0.564 en train y 0.543 en test, una brecha de 0.02.

**Variables más influyentes** (importancia por permutación, caída del F1 en test):
1. `root_conflict_score` (0.113)
2. `tree_dbh` (0.031)
3. `borough` (0.013)
4. `spc_clean` (0.011)

El riesgo de daño sube fuerte entre 5 y 15 pulgadas de diámetro y después se estabiliza. Tener el primer conflicto de raíz es lo que más aumenta el riesgo.

---

## 7. Cómo ejecutar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Final.ipynb
```

---

## 8. Limitaciones

- Es una foto única de 2015: no captura cómo evoluciona el daño en el tiempo.
- Los datos los relevaron voluntarios, así que puede haber errores de medición.
- Asociación no es causalidad: el modelo detecta patrones estadísticos, no relaciones causa-efecto.
