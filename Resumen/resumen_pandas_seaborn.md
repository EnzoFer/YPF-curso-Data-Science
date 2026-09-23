# Guía Completa de Pandas, Seaborn, Scikit-Learn, SciPy y Machine Learning en Ciencia de Datos

Este documento presenta un compendio integral, exhaustivo y estructurado de todas las funciones, métodos, estimadores, transformadores, conceptos teóricos y buenas prácticas de **Pandas**, **Seaborn**, **Scikit-Learn** (`sklearn`), **SciPy** (`scipy.stats`), **NumPy** y **Optuna** desarrollados y analizados a lo largo de las clases y notebooks del programa (Notebooks 6 al 17).

El contenido se encuentra estrictamente organizado según el ciclo de vida o pipeline de la **Ingeniería de Datos y el Modelado en Machine Learning**:

1. [Etapa 1: Ingestión y Carga de Datos](#etapa-1-ingestión-y-carga-de-datos)
   * [1.1 `pd.read_csv()`](#11-pdread_csv)
   * [1.2 `pd.DataFrame()`](#12-pddataframe)
2. [Etapa 2: Exploración e Inspección Estructural](#etapa-2-exploración-e-inspección-estructural)
   * [2.1 `df.head()` y `df.tail()`](#21-dfhead-y-dftail)
   * [2.2 `df.shape`](#22-dfshape)
   * [2.3 `df.columns`](#23-dfcolumns)
   * [2.4 `df.dtypes` y `Series.dtype`](#24-dfdtypes-y-seriesdtype)
   * [2.5 `len(df)`](#25-lendf)
   * [2.6 `df.info()`](#26-dfinfo)
3. [Etapa 3: Filtrado, Selección e Indexación](#etapa-3-filtrado-selección-e-indexación)
   * [3.1 `df.iloc[]`](#31-dfiloc)
   * [3.2 `df.loc[]`](#32-dfloc)
   * [3.3 Filtrado Booleano / Indexación Condicional](#33-filtrado-booleano--indexación-condicional)
   * [3.4 `Series.idxmax()` y `Series.idxmin()`](#34-seriesidxmax-y-seriesidxmin)
   * [3.5 `Series.isin()`](#35-seriesisin)
4. [Etapa 4: Limpieza, Diagnóstico de Distribución, Valores Faltantes, Duplicados y Outliers](#etapa-4-limpieza-diagnóstico-de-distribución-valores-faltantes-duplicados-y-outliers)
   * [4.1 `df.isnull()` y `df.isna()`](#41-dfisnull-y-dfisna)
   * [4.2 `df.isnull().sum()` y Conteo de Nulos](#42-dfisnullsum-y-conteo-de-nulos)
   * [4.3 `df.dropna()`](#43-dfdropna)
   * [4.4 `df.fillna()`](#44-dffillna)
   * [4.5 `df.duplicated()` y `df.drop_duplicates()`](#45-dfduplicated-y-dfdrop_duplicates)
   * [4.6 `pd.to_numeric()`](#46-pdto_numeric)
   * [4.7 `stats.zscore()` (Detección de Outliers por Puntaje Z con SciPy)](#47-statszscore-detección-de-outliers-por-puntaje-z-con-scipy)
   * [4.8 `Series.skew()` (Evaluación del Coeficiente de Asimetría)](#48-seriesskew-evaluación-del-coeficiente-de-asimetría)
   * [4.9 `stats.probplot()` (Gráficos Q-Q Plot con SciPy)](#49-statsprobplot-gráficos-q-q-plot-con-scipy)
5. [Etapa 5: Transformación, Reestructuración e Ingeniería de Funciones](#etapa-5-transformación-reestructuración-e-ingeniería-de-funciones)
   * [5.1 `df.rename()`](#51-dfrename)
   * [5.2 `df.drop()`](#52-dfdrop)
   * [5.3 `df.reset_index()`](#53-dfreset_index)
   * [5.4 `pd.melt()`](#54-pdmelt)
   * [5.5 `pd.concat()`](#55-pdconcat)
   * [5.6 `pd.get_dummies()` (One-Hot Encoding en Pandas)](#56-pdget_dummies-one-hot-encoding-en-pandas)
   * [5.7 `pd.cut()` (Discretización / Binning)](#57-pdcut-discretización--binning)
   * [5.8 `df.sort_values()`](#58-dfsort_values)
   * [5.9 `df.replace()`](#59-dfreplace)
   * [5.10 Métodos de Cadenas Vectorizadas (`Series.str`)](#510-métodos-de-cadenas-vectorizadas-seriesstr)
   * [5.11 `Series.astype()`](#511-seriesastype)
6. [Etapa 6: Agregación, Estadísticas Descriptivas y Análisis Multivariado](#etapa-6-agregación-estadísticas-descriptivas-y-análisis-multivariado)
   * [6.1 `df.describe()` y `Series.describe()`](#61-dfdescribe-y-seriesdescribe)
   * [6.2 Métodos Estadísticos de Agregación Simples](#62-métodos-estadísticos-de-agregación-simples)
   * [6.3 `Series.unique()`](#63-seriesunique)
   * [6.4 `Series.value_counts()`](#64-seriesvalue_counts)
   * [6.5 `df.groupby()`](#65-dfgroupby)
   * [6.6 `pd.crosstab()`](#66-pdcrosstab)
   * [6.7 `df.corr()`](#67-dfcorr)
   * [6.8 `df.corrwith()`](#68-dfcorrwith)
7. [Etapa 7: Visualización Exploratoria de Datos (Seaborn y Matplotlib)](#etapa-7-visualización-exploratoria-de-datos-seaborn-y-matplotlib)
   * [7.1 `sns.displot()`](#71-snsdisplot)
   * [7.2 `sns.histplot()`](#72-snshistplot)
   * [7.3 `sns.countplot()`](#73-snscountplot)
   * [7.4 `sns.barplot()`](#74-snsbarplot)
   * [7.5 `sns.boxplot()`](#75-snsboxplot)
   * [7.6 `sns.scatterplot()`](#76-snsscatterplot)
   * [7.7 `sns.pairplot()`](#77-snspairplot)
   * [7.8 `sns.heatmap()`](#78-snsheatmap)
8. [Etapa 8: Estilizado y Personalización Visual de Gráficos](#etapa-8-estilizado-y-personalización-visual-de-gráficos)
   * [8.1 `sns.despine()`](#81-snsdespine)
9. [Etapa 9: Preprocesamiento, Escalado, Transformación y Pipelines con Scikit-Learn](#etapa-9-preprocesamiento-escalado-transformación-y-pipelines-con-scikit-learn)
   * [9.1 `SimpleImputer` (Imputación Automática de Valores Faltantes)](#91-simpleimputer-imputación-automática-de-valores-faltantes)
   * [9.2 `LabelEncoder` (Codificación Ordinal / de Etiquetas) y su Regla de Uso](#92-labelencoder-codificación-ordinal--de-etiquetas-y-su-regla-de-uso)
   * [9.3 `OneHotEncoder` (Codificación Categórica Nominal)](#93-onehotencoder-codificación-categórica-nominal)
   * [9.4 `MinMaxScaler` (Re-escalado de Características a un Rango)](#94-minmaxscaler-re-escalado-de-características-a-un-rango)
   * [9.5 `StandardScaler` (Estandarización / Escala Z)](#95-standardscaler-estandarización--escala-z)
   * [9.6 `RobustScaler` (Escalado Robusto Resistente a Outliers)](#96-robustscaler-escalado-robusto-resistente-a-outliers)
   * [9.7 `Normalizer` (Normalización por Normas Vectoriales de Muestras)](#97-normalizer-normalización-por-normas-vectoriales-de-muestras)
   * [9.8 `PowerTransformer` (Transformación de Potencia Yeo-Johnson y Box-Cox)](#98-powertransformer-transformación-de-potencia-yeo-johnson-y-box-cox)
   * [9.9 `np.log1p()` y `np.expm1()` (Transformación Logarítmica de Forma)](#99-nplog1p-y-npexpm1-transformación-logarítmica-de-forma)
   * [9.10 `Pipeline` (Ensamblaje Secuencial de Preprocesamiento y Modelado)](#910-pipeline-ensamblaje-secuencial-de-preprocesamiento-y-modelado)
   * [9.11 `ColumnTransformer` (Transformaciones Diferenciadas por Tipo de Columna)](#911-columntransformer-transformaciones-diferenciadas-por-tipo-de-columna)
10. [Etapa 10: Flujo Correcto de Entrenamiento y Prevención de Data Leakage](#etapa-10-flujo-correcto-de-entrenamiento-y-prevención-de-data-leakage)
    * [10.1 `train_test_split()` (División del Dataset en Train y Test)](#101-train_test_split-división-del-dataset-en-train-y-test)
    * [10.2 Regla de Oro: Separación de `fit_transform` en Train vs `transform` en Test](#102-regla-de-oro-separación-de-fit_transform-en-train-vs-transform-en-test)
11. [Etapa 11: Fundamentos de Machine Learning y Formulación del Problema](#etapa-11-fundamentos-de-machine-learning-y-formulación-del-problema)
    * [11.1 Los Cuatro Componentes de Mitchell ($T$, $E$, $P$, $A$)](#111-los-cuatro-componentes-de-mitchell-t-e-p-a)
    * [11.2 Paradigmas: Aprendizaje Supervisado vs. No Supervisado](#112-paradigmas-aprendizaje-supervisado-vs-no-supervisado)
    * [11.3 Tareas Supervisadas: Regresión vs. Clasificación ($X$ vs. $y$)](#113-tareas-supervisadas-regresión-vs-clasificación-x-vs-y)
    * [11.4 Subajuste (*Underfitting*), Sobreajuste (*Overfitting*) y Compensación Sesgo-Varianza](#114-subajuste-underfitting-sobreajuste-overfitting-y-compensación-sesgo-varianza)
12. [Etapa 12: Modelos de Línea Base / Pisos de Referencia (*Baselines*)](#etapa-12-modelos-de-línea-base--pisos-de-referencia-baselines)
    * [12.1 `DummyClassifier`](#121-dummyclassifier)
    * [12.2 `DummyRegressor`](#122-dummyregressor)
13. [Etapa 13: Algoritmos de Aprendizaje Supervisado](#etapa-13-algoritmos-de-aprendizaje-supervisado)
    * [13.1 `LinearRegression` (Regresión Lineal Múltiple)](#131-linearregression-regresión-lineal-múltiple)
    * [13.2 `LogisticRegression` (Regresión Logística para Clasificación)](#132-logisticregression-regresión-logística-para-clasificación)
    * [13.3 `DecisionTreeClassifier` y `DecisionTreeRegressor` (Árboles de Decisión)](#133-decisiontreeclassifier-y-decisiontreeregressor-árboles-de-decisión)
    * [13.4 `plot_tree()` (Visualización Gráfica de Árboles)](#134-plot_tree-visualización-gráfica-de-árboles)
    * [13.5 `KNeighborsClassifier` (k-Nearest Neighbors / k-Vecinos Más Cercanos)](#135-kneighborsclassifier-k-nearest-neighbors--k-vecinos-más-cercanos)
    * [13.6 `SVC` (Support Vector Classifier / Máquinas de Vectores de Soporte)](#136-svc-support-vector-classifier--máquinas-de-vectores-de-soporte)
    * [13.7 `RandomForestClassifier` (Bosques Aleatorios / Ensamble Bagging)](#137-randomforestclassifier-bosques-aleatorios--ensamble-bagging)
14. [Etapa 14: Métricas de Evaluación y Diagnóstico del Rendimiento](#etapa-14-métricas-de-evaluación-y-diagnóstico-del-rendimiento)
    * [14.1 Métricas de Regresión: `mean_absolute_error` (MAE) y `mean_squared_error` (MSE/RMSE)](#141-métricas-de-regresión-mean_absolute_error-mae-y-mean_squared_error-msermse)
    * [14.2 Métricas de Regresión: `r2_score` ($R^2$ - Coeficiente de Determinación)](#142-métricas-de-regresión-r2_score-r2---coeficiente-de-determinación)
    * [14.3 Métricas de Clasificación: `accuracy_score` (Exactitud Global)](#143-métricas-de-clasificación-accuracy_score-exactitud-global)
    * [14.4 `confusion_matrix` y `ConfusionMatrixDisplay`](#144-confusion_matrix-y-confusionmatrixdisplay)
    * [14.5 `classification_report` (Precisión, Recall, F1-Score y Soporte)](#145-classification_report-precisión-recall-f1-score-y-soporte)
15. [Etapa 15: Validación Robusta y Curvas de Aprendizaje](#etapa-15-validación-robusta-y-curvas-de-aprendizaje)
    * [15.1 `cross_val_score` (Validación Cruzada K-Fold)](#151-cross_val_score-validación-cruzada-k-fold)
    * [15.2 `learning_curve` (Curvas de Diagnóstico de Aprendizaje)](#152-learning_curve-curvas-de-diagnóstico-de-aprendizaje)
16. [Etapa 16: Optimización de Hiperparámetros con Optuna](#etapa-16-optimización-de-hiperparámetros-con-optuna)
    * [16.1 Flujo de Búsqueda y Optimización Bayesiana con `optuna`](#161-flujo-de-búsqueda-y-optimización-bayesiana-con-optuna)
17. [Etapa 17: Exportación y Almacenamiento de Datos](#etapa-17-exportación-y-almacenamiento-de-datos)
    * [17.1 `df.to_csv()` (Exportación de DataFrames a Archivos CSV)](#171-dfto_csv-exportación-de-dataframes-a-archivos-csv)
18. [Tabla Resumen Integral: Mapeo de Funciones por Etapa del Pipeline](#tabla-resumen-integral-mapeo-de-funciones-por-etapa-del-pipeline)

---

## Etapa 1: Ingestión y Carga de Datos

En esta etapa inicial se importan los datos desde fuentes externas (archivos locales CSV, URLs o estructuras de datos en memoria) hacia objetos estructurados de Pandas.

### 1.1 `pd.read_csv()`
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Lee un archivo de texto con formato de valores separados por comas (o cualquier delimitador como `;` o tabuladores) y lo carga como un objeto `DataFrame`. También permite leer archivos remotos especificando la dirección URL del dataset.
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.read_csv(filepath_or_buffer, sep=',', errors='ignore', ...)`
  * **Ejemplo de código:**
    ```python
    import pandas as pd

    # Carga local de dataset de expectativa de vida
    expvida = pd.read_csv('life_expectancy_data.csv')
    expvida.head(2)
    ```
  * **Output esperado / Resultado:**
    ```text
          Country  Year      Status  life_expectancy  Adult Mortality  bmi  measles
    0  Afghanistan  2015  Developing             65.0            263.0 19.1      1150
    1  Afghanistan  2014  Developing             59.9            271.0 18.6       492
    ```

---

### 1.2 `pd.DataFrame()`
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Constructor fundamental de la estructura bidimensional `DataFrame`. Permite crear un DataFrame desde cero a partir de diccionarios, listas, arrays de NumPy o transformaciones de objetos resultantes de codificaciones de Machine Learning.
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.DataFrame(data=None, index=None, columns=None, dtype=None)`
  * **Ejemplo de código:**
    ```python
    import pandas as pd
    import numpy as np

    # Creación manual desde listas e índices personalizados
    df2 = pd.DataFrame(data=['perro', 'gato', 'flor'], 
                       index=['str1', 'str2', 'str3'], 
                       columns=['titulo'])
    print(df2)
    ```
  * **Output esperado / Resultado:**
    ```text
           titulo
    str1    perro
    str2     gato
    str3     flor
    ```

---

## Etapa 2: Exploración e Inspección Estructural

Una vez cargados los datos, es indispensable evaluar las dimensiones, los tipos de datos en cada columna y visualizar muestras iniciales/finales del dataset.

### 2.1 `df.head()` y `df.tail()`
* **Librería:** Pandas (Métodos de `DataFrame`)
* **¿Qué hace?:** 
  * `head(n)`: Devuelve las primeras `n` filas del DataFrame (por defecto 5).
  * `tail(n)`: Devuelve las últimas `n` filas del DataFrame (por defecto 5).
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.head(n=5)` / `df.tail(n=5)`
  * **Ejemplo de código:**
    ```python
    # Muestra las primeras 3 filas
    expvida.head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
          Country  Year      Status  life_expectancy  bmi
    0  Afghanistan  2015  Developing             65.0 19.1
    1  Afghanistan  2014  Developing             59.9 18.6
    2  Afghanistan  2013  Developing             59.9 18.1
    ```

---

### 2.2 `df.shape`
* **Librería:** Pandas (Atributo de `DataFrame`)
* **¿Qué hace?:** Retorna una tupla de enteros que representa las dimensiones del DataFrame en el formato `(cantidad_de_filas, cantidad_de_columnas)`.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.shape`
  * **Ejemplo de código:**
    ```python
    print(expvida.shape)
    ```
  * **Output esperado / Resultado:**
    ```text
    (2938, 22)
    ```

---

### 2.3 `df.columns`
* **Librería:** Pandas (Atributo de `DataFrame`)
* **¿Qué hace?:** Devuelve un objeto de tipo `Index` con los nombres de todas las columnas presentes en el DataFrame.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.columns`
  * **Ejemplo de código:**
    ```python
    print(expvida.columns)
    ```
  * **Output esperado / Resultado:**
    ```text
    Index(['Country', 'Year', 'Status', 'Life expectancy ', 'Adult Mortality',
           'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
           'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
           'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
           ' thinness  1-19 years', ' thinness 5-9 years',
           'Income composition of resources', 'Schooling'],
          dtype='object')
    ```

---

### 2.4 `df.dtypes` y `Series.dtype`
* **Librería:** Pandas (Atributo de `DataFrame` y `Series`)
* **¿Qué hace?:** `df.dtypes` indica el tipo de dato de cada columna (`int64`, `float64`, `object`, `bool`, etc.). `Series.dtype` indica el tipo de dato de una columna en particular.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.dtypes` / `df['columna'].dtype`
  * **Ejemplo de código:**
    ```python
    # Ver tipo de dato de una columna específica
    print(expvida['Year'].dtype)
    ```
  * **Output esperado / Resultado:**
    ```text
    int64
    ```

---

### 2.5 `len(df)`
* **Librería:** Python nativo aplicado a Pandas
* **¿Qué hace?:** Retorna la cantidad total de filas (longitud de la primera dimensión) del DataFrame.
* **¿Cómo usarla?:**
  * **Sintaxis:** `len(df)`
  * **Ejemplo de código:**
    ```python
    print(len(expvida))
    ```
  * **Output esperado / Resultado:**
    ```text
    2938
    ```

---

### 2.6 `df.info()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Imprime un resumen diagnóstico conciso del DataFrame: cantidad total de registros, rango de índices, nombres de columnas con su recuento de datos no nulos (`Non-Null Count`), tipo de dato de cada una y uso total de memoria RAM. Es la función de cabecera para auditorías rápidas de calidad inicial.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.info(verbose=True, memory_usage=True)`
  * **Ejemplo de código:**
    ```python
    expvida.info()
    ```
  * **Output esperado / Resultado:**
    ```text
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2938 entries, 0 to 2937
    Data columns (total 22 columns):
     #   Column                           Non-Null Count  Dtype  
    ---  ------                           --------------  -----  
     0   Country                          2938 non-null   object 
     1   Year                             2938 non-null   int64  
     2   Status                           2938 non-null   object 
     3   Life expectancy                  2928 non-null   float64
     4   Adult Mortality                  2928 non-null   float64
     ...
    dtypes: float64(16), int64(4), object(2)
    memory usage: 505.1+ KB
    ```

---

## Etapa 3: Filtrado, Selección e Indexación

En esta fase se recortan, seleccionan o filtran subconjuntos de datos según posiciones enteras (`iloc`), etiquetas de fila/columna (`loc`), condiciones lógicas booleanas o pertenencia a listas (`isin`).

### 3.1 `df.iloc[]`
* **Librería:** Pandas (Indexer de `DataFrame`)
* **¿Qué hace?:** Permite la selección e indexación puramente basada en la **posición entera** (0 a N-1) de filas y columnas, independientemente de los nombres de etiquetas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.iloc[filas, columnas]`
  * **Ejemplo de código:**
    ```python
    # Seleccionar las primeras 2 columnas de las primeras 3 filas
    expvida.iloc[0:3, 0:2]
    ```
  * **Output esperado / Resultado:**
    ```text
          Country  Year
    0  Afghanistan  2015
    1  Afghanistan  2014
    2  Afghanistan  2013
    ```

---

### 3.2 `df.loc[]`
* **Librería:** Pandas (Indexer de `DataFrame`)
* **¿Qué hace?:** Permite la selección basada en **etiquetas/nombres** de filas o columnas, o mediante **máscaras booleanas**.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.loc[filas_etiquetas, columnas_etiquetas]`
  * **Ejemplo de código:**
    ```python
    # Seleccionar filas por índice y columnas específicas por nombre
    expvida.loc[0:2, ['Country', 'Status', 'Life expectancy ']]
    ```
  * **Output esperado / Resultado:**
    ```text
          Country      Status  Life expectancy 
    0  Afghanistan  Developing              65.0
    1  Afghanistan  Developing              59.9
    2  Afghanistan  Developing              59.9
    ```

---

### 3.3 Filtrado Booleano / Indexación Condicional
* **Librería:** Pandas
* **¿Qué hace?:** Permite filtrar los registros de un DataFrame aplicando condiciones lógicas (usando operadores `&` para AND, `|` para OR, y `~` para NOT) sobre una o más columnas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df[(condicion_1) & (condicion_2)]`
  * **Ejemplo de código:**
    ```python
    # Países en desarrollo con expectativa de vida mayor a 75 años
    filtro = (expvida['Status'] == 'Developing') & (expvida['Life expectancy '] > 75)
    expvida[filtro][['Country', 'Year', 'Life expectancy ']].head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
              Country  Year  Life expectancy 
    48        Albania  2015              77.8
    49        Albania  2014              77.5
    50        Albania  2013              77.2
    ```

---

### 3.4 `Series.idxmax()` y `Series.idxmin()`
* **Librería:** Pandas (Métodos de `Series`)
* **¿Qué hace?:** Devuelve la etiqueta del **índice** donde se encuentra el valor máximo (`idxmax`) o mínimo (`idxmin`) de una columna o Serie.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['columna'].idxmax()` / `df['columna'].idxmin()`
  * **Ejemplo de código:**
    ```python
    # Obtener el índice del país con mayor expectativa de vida y extraer su fila
    col = 'Life expectancy '
    idx_max = expvida[col].idxmax()
    print("Índice máximo:", idx_max)
    print(expvida.loc[idx_max, ['Country', 'Year', col]])
    ```
  * **Output esperado / Resultado:**
    ```text
    Índice máximo: 241
    Country             Belgium
    Year                   2014
    Life expectancy        89.0
    Name: 241, dtype: object
    ```

---

### 3.5 `Series.isin()`
* **Librería:** Pandas (Método de `Series`)
* **¿Qué hace?:** Evalúa si cada elemento de una Serie está contenido dentro de un iterable (lista, tupla o conjunto), retornando una máscara booleana. Es la forma óptima y legible de filtrar múltiples categorías sin encadenar múltiples operadores `|` (`OR`).
* **¿Cómo usarla?:**
  * **Sintaxis:** `df[df['columna'].isin(['valor1', 'valor2', ...])]`
  * **Ejemplo de código:**
    ```python
    paises = ['Argentina', 'Chile', 'Uruguay', 'Brazil']
    filtro_cono_sur = expvida['Country'].isin(paises)
    expvida[filtro_cono_sur]['Country'].unique()
    ```
  * **Output esperado / Resultado:**
    ```text
    array(['Argentina', 'Brazil', 'Chile', 'Uruguay'], dtype=object)
    ```

---

## Etapa 4: Limpieza, Diagnóstico de Distribución, Valores Faltantes, Duplicados y Outliers

La calidad de datos asegura que la información esté libre de incoherencias, faltantes (`NaN`), anomalías, duplicados u *outliers*, e identifica el grado de asimetría de las distribuciones numéricas.

### 4.1 `df.isnull()` y `df.isna()`
* **Librería:** Pandas (Métodos de `DataFrame` y `Series`)
* **¿Qué hace?:** Detectan valores faltantes (`NaN`, `None`). Retornan un DataFrame/Serie del mismo tamaño compuesto por valores booleanos (`True` donde hay valor nulo, `False` donde hay valor válido).
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.isnull()` / `df.isna()`
  * **Ejemplo de código:**
    ```python
    expvida['Alcohol'].isna().head(4)
    ```
  * **Output esperado / Resultado:**
    ```text
    0    False
    1    False
    2    False
    3    False
    Name: Alcohol, dtype: bool
    ```

---

### 4.2 `df.isnull().sum()` y Conteo de Nulos
* **Librería:** Pandas (Combinación de `isnull()` / `isna()` con `sum()`)
* **¿Qué hace?:** Contabiliza la cantidad total de valores nulos por columna sumando los valores booleanos `True`.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.isnull().sum()`
  * **Ejemplo de código:**
    ```python
    faltantes = expvida.isnull().sum()
    faltantes[faltantes > 0].head(4)
    ```
  * **Output esperado / Resultado:**
    ```text
    Life expectancy     10
    Adult Mortality     10
    Alcohol            194
    Hepatitis B        553
    dtype: int64
    ```

---

### 4.3 `df.dropna()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Elimina las filas o columnas que contienen valores nulos.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.dropna(axis=0, how='any', subset=None, inplace=False)`
  * **Ejemplo de código:**
    ```python
    print("Dimensiones antes:", expvida.shape)
    expvida_clean = expvida.dropna(subset=['Life expectancy '])
    print("Dimensiones después:", expvida_clean.shape)
    ```
  * **Output esperado / Resultado:**
    ```text
    Dimensiones antes: (2938, 22)
    Dimensiones después: (2928, 22)
    ```

---

### 4.4 `df.fillna()`
* **Librería:** Pandas (Método de `DataFrame` y `Series`)
* **¿Qué hace?:** Imputa o rellena valores faltantes (`NaN`) utilizando valores constantes fijos o estadísticos descriptivos calculados directamente en Pandas (media, mediana o moda).
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['col'].fillna(valor, inplace=False)`
  * **Ejemplo de código:**
    ```python
    mediana_alcohol = expvida['Alcohol'].median()
    expvida['Alcohol_imp'] = expvida['Alcohol'].fillna(mediana_alcohol)
    print("Nulos restantes:", expvida['Alcohol_imp'].isna().sum())
    ```
  * **Output esperado / Resultado:**
    ```text
    Nulos restantes: 0
    ```

---

### 4.5 `df.duplicated()` y `df.drop_duplicates()`
* **Librería:** Pandas (Métodos de `DataFrame`)
* **¿Qué hace?:** 
  * `duplicated()`: Detecta filas completas o subconjuntos de columnas con registros duplicados idénticos, retornando una máscara booleana.
  * `drop_duplicates()`: Elimina los registros repetidos reteniendo por defecto la primera aparición.
* **¿Cómo usarla?:**
  * **Sintaxis:**
    * `df.duplicated(subset=None, keep='first').sum()`
    * `df.drop_duplicates(subset=None, keep='first', inplace=False)`
  * **Ejemplo de código:**
    ```python
    print("Duplicados detectados:", expvida.duplicated().sum())
    expvida_unicos = expvida.drop_duplicates()
    print("Dimensiones tras eliminar duplicados:", expvida_unicos.shape)
    ```
  * **Output esperado / Resultado:**
    ```text
    Duplicados detectados: 0
    Dimensiones tras eliminar duplicados: (2938, 22)
    ```

---

### 4.6 `pd.to_numeric()`
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Convierte una Serie a un tipo de dato numérico (`int` o `float`), transformando caracteres inválidos o cadenas no parseables en `NaN` cuando se especifica `errors='coerce'`.
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.to_numeric(arg, errors='raise', downcast=None)`
  * **Ejemplo de código:**
    ```python
    serie_sucia = pd.Series(['10.5', '20.3', 'desconocido', '45.0'])
    serie_num = pd.to_numeric(serie_sucia, errors='coerce')
    print(serie_num)
    ```
  * **Output esperado / Resultado:**
    ```text
    0    10.5
    1    20.3
    2     NaN
    3    45.0
    dtype: float64
    ```

---

### 4.7 `stats.zscore()` (Detección de Outliers por Puntaje Z con SciPy)
* **Librería:** SciPy (`scipy.stats.zscore`)
* **¿Qué hace?:** Calcula el puntaje Z (*Z-score*) para cada observación en una columna cuantitativa. Mide a cuántas desviaciones estándar de la media se encuentra cada valor ($Z = \frac{x - \mu}{\sigma}$). Permite identificar y filtrar *outliers* estableciendo un umbral (típicamente $|Z| > 2.0$ o $|Z| > 3.0$).
* **¿Cómo usarla?:**
  * **Sintaxis:** `z = stats.zscore(array_o_columna)`
  * **Ejemplo de código:**
    ```python
    from scipy import stats
    import numpy as np

    serie_gdp = expvida['GDP'].dropna()
    z = stats.zscore(serie_gdp)
    threshold = 3.0

    outliers = serie_gdp[np.abs(z) > threshold]
    print(f"Total outliers con |Z| > 3: {len(outliers)}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Total outliers con |Z| > 3: 54
    ```

---

### 4.8 `Series.skew()` (Evaluación del Coeficiente de Asimetría)
* **Librería:** Pandas (Método de `Series` / `DataFrame`)
* **¿Qué hace?:** Computa el coeficiente de asimetría (*skewness*) de distribuciones cuantitativas continuas. 
  * Cercano a `0`: distribución simétrica.
  * Mayor a `0.5` o `1.0`: **sesgo positivo (cola larga a la derecha)**.
  * Menor a `-0.5`: **sesgo negativo (cola larga a la izquierda)**.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['columna'].skew()` / `df.select_dtypes(include=np.number).skew()`
  * **Ejemplo de código:**
    ```python
    asimetrias = expvida.select_dtypes(include=np.number).skew().sort_values(ascending=False)
    print(asimetrias.head(4))
    ```
  * **Output esperado / Resultado:**
    ```text
    Population            15.955473
    Measles               9.441324
    under-five deaths     6.852109
    infant deaths         6.820202
    dtype: float64
    ```

---

### 4.9 `stats.probplot()` (Gráficos Q-Q Plot con SciPy)
* **Librería:** SciPy (`scipy.stats.probplot`)
* **¿Qué hace?:** Genera un gráfico de probabilidad o **Q-Q Plot** comparando visualmente los cuantiles empíricos observados contra los cuantiles teóricos de una distribución normal. Puntos sobre la diagonal a 45° indican normalidad; curvaturas o desviaciones indican colas pesadas o asimetría.
* **¿Cómo usarla?:**
  * **Sintaxis:** `stats.probplot(series_o_array, dist="norm", plot=plt)`
  * **Ejemplo de código:**
    ```python
    import matplotlib.pyplot as plt
    from scipy import stats

    gdp = expvida['GDP'].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    stats.probplot(gdp, dist="norm", plot=ax)
    plt.title('Q-Q Plot: GDP')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Plano cartesiano con una línea recta diagonal roja teórica
    y puntos azules de GDP con fuerte curvatura en los extremos superiores, evidenciando asimetría positiva.
    ```

---

## Etapa 5: Transformación, Reestructuración e Ingeniería de Funciones

En esta fase se renombran variables, eliminan columnas irrelevantes, aplican técnicas de estructuración de columnas, segmentación (*Binning*), reemplazo de valores imposibles y normalización de textos.

### 5.1 `df.rename()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Altera las etiquetas de las columnas o del índice especificando un diccionario con los pares `{"nombre_viejo": "nombre_nuevo"}`.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.rename(columns={"viejo": "nuevo"}, inplace=False)`
  * **Ejemplo de código:**
    ```python
    expvida = expvida.rename(columns={"Life expectancy ": "life_expectancy", " BMI ": "bmi"})
    print([c for c in expvida.columns if c in ['life_expectancy', 'bmi']])
    ```
  * **Output esperado / Resultado:**
    ```text
    ['life_expectancy', 'bmi']
    ```

---

### 5.2 `df.drop()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Remueve las filas (`axis=0`) o columnas (`axis=1` o `columns=[...]`) especificadas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.drop(columns=['col1', 'col2'], inplace=False)`
  * **Ejemplo de código:**
    ```python
    df_sin_anio = expvida.drop(columns=['Year'])
    print('Year' in df_sin_anio.columns)
    ```
  * **Output esperado / Resultado:**
    ```text
    False
    ```

---

### 5.3 `df.reset_index()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Reinicia el índice del DataFrame devolviéndolo a una secuencia entera limpia `0, 1, ..., N-1`. Si `drop=True`, descarta el índice anterior en lugar de guardarlo como columna.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.reset_index(drop=True, inplace=False)`
  * **Ejemplo de código:**
    ```python
    df_filtrado = expvida[expvida['Status'] == 'Developed'].reset_index(drop=True)
    print(df_filtrado.index[:3])
    ```
  * **Output esperado / Resultado:**
    ```text
    RangeIndex(start=0, stop=3, step=1)
    ```

---

### 5.4 `pd.melt()`
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Despivota un DataFrame pasando de formato ancho (*wide format*) a formato largo (*long format*). Concentra múltiples columnas de medición en dos columnas: variable identificadora y valor medido.
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.melt(df, id_vars=['id'], value_vars=['col1', 'col2'], var_name='variable', value_name='valor')`
  * **Ejemplo de código:**
    ```python
    melted = pd.melt(expvida, id_vars=['Country'], value_vars=['Adult Mortality', 'Alcohol'],
                     var_name='Metrica', value_name='Valor')
    melted.head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
          Country          Metrica  Valor
    0  Afghanistan  Adult Mortality  263.0
    1  Afghanistan  Adult Mortality  271.0
    2  Afghanistan  Adult Mortality  268.0
    ```

---

### 5.5 `pd.concat()`
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Une o concatena múltiples DataFrames a lo largo de un eje: verticalmente apilando filas (`axis=0`) u horizontalmente uniendo columnas (`axis=1`).
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.concat([df1, df2], axis=0, ignore_index=True)`
  * **Ejemplo de código:**
    ```python
    df_dev = expvida[expvida['Status'] == 'Developed']
    df_ing = expvida[expvida['Status'] == 'Developing']
    unificado = pd.concat([df_dev, df_ing], axis=0, ignore_index=True)
    print("Filas concatenadas:", len(unificado))
    ```
  * **Output esperado / Resultado:**
    ```text
    Filas concatenadas: 2938
    ```

---

### 5.6 `pd.get_dummies()` (One-Hot Encoding en Pandas)
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Convierte columnas categóricas en columnas binarias indicadoras ($0$ o $1$). El parámetro `drop_first=True` elimina la primera categoría para evitar la colinealidad perfecta (trampa de las variables ficticias).
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.get_dummies(df, columns=['cat1'], drop_first=True, dtype=int)`
  * **Ejemplo de código:**
    ```python
    dummies_status = pd.get_dummies(expvida[['Status']], drop_first=True, dtype=int)
    dummies_status.head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
       Status_Developing
    0                  1
    1                  1
    2                  1
    ```

---

### 5.7 `pd.cut()` (Discretización / Binning)
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Segmenta y agrupa los valores de una variable continua en intervalos discretos (*bins*), permitiendo asignar etiquetas categóricas cualitativas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.cut(x, bins, labels=None, include_lowest=True)`
  * **Ejemplo de código:**
    ```python
    bins = [0, 60, 75, 100]
    labels = ['Baja', 'Media', 'Alta']
    expvida['rango_vida'] = pd.cut(expvida['life_expectancy'], bins=bins, labels=labels)
    expvida['rango_vida'].value_counts()
    ```
  * **Output esperado / Resultado:**
    ```text
    rango_vida
    Media    1563
    Alta      921
    Baja      444
    Name: count, dtype: int64
    ```

---

### 5.8 `df.sort_values()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Ordena las filas del DataFrame según los valores de una o varias columnas, en orden ascendente o descendente.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.sort_values(by=['columna'], ascending=False)`
  * **Ejemplo de código:**
    ```python
    expvida.sort_values(by='life_expectancy', ascending=False)[['Country', 'Year', 'life_expectancy']].head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
              Country  Year  life_expectancy
    241       Belgium  2014             89.0
    996       Germany  2014             89.0
    2056     Portugal  2014             89.0
    ```

---

### 5.9 `df.replace()`
* **Librería:** Pandas (Método de `DataFrame` y `Series`)
* **¿Qué hace?:** Sustituye valores específicos, códigos de error o valores imposibles (por ejemplo valores negativos o centinelas como `-1`, `'?'`, `999`) por valores válidos o `np.nan`.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['col'].replace(to_replace, value, inplace=False)`
  * **Ejemplo de código:**
    ```python
    import numpy as np
    
    # Reemplazar valores centinela de años negativos o imposibles por NaN
    expvida['Adult Mortality'] = expvida['Adult Mortality'].replace(-1, np.nan)
    ```
  * **Output esperado / Resultado:**
    ```text
    Todos los valores centinela -1 son convertidos a datos faltantes (NaN) para su posterior imputación.
    ```

---

### 5.10 Métodos de Cadenas Vectorizadas (`Series.str`)
* **Librería:** Pandas (Accesor `Series.str`)
* **¿Qué hace?:** Aplica operaciones vectorizadas de texto sobre columnas de tipo string o sobre el índice `df.columns`. Es la técnica estándar para sanear nombres de columnas con espacios no deseados.
  * `.str.strip()`: Quita espacios en blanco al inicio y final.
  * `.str.lower()`: Pasa todo a minúsculas homogéneas.
  * `.str.replace(' ', '_')`: Sustituye espacios por guiones bajos.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')`
  * **Ejemplo de código:**
    ```python
    print("Columnas antes:", expvida.columns.tolist()[:3])
    expvida.columns = expvida.columns.str.strip().str.lower().str.replace(' ', '_')
    print("Columnas saneadas:", expvida.columns.tolist()[:3])
    ```
  * **Output esperado / Resultado:**
    ```text
    Columnas antes: ['Country', 'Year', 'Status']
    Columnas saneadas: ['country', 'year', 'status']
    ```

---

### 5.11 `Series.astype()`
* **Librería:** Pandas (Método de `Series`)
* **¿Qué hace?:** Convierte forzadamente el tipo de dato de una Serie a otro especificado (`int`, `float`, `str`, `category`). En Machine Learning supervisado es el método estándar para convertir máscaras lógicas booleanas en la variable objetivo binaria ($0$ y $1$).
* **¿Cómo usarla?:**
  * **Sintaxis:** `(condicion).astype(int)`
  * **Ejemplo de código:**
    ```python
    # Generar vector objetivo binario: 1 si expectativa >= 70, 0 si menor
    expvida['vida_alta'] = (expvida['life_expectancy'] >= 70).astype(int)
    print(expvida['vida_alta'].value_counts())
    ```
  * **Output esperado / Resultado:**
    ```text
    vida_alta
    1    1590
    0    1348
    Name: count, dtype: int64
    ```

---

## Etapa 6: Agregación, Estadísticas Descriptivas y Análisis Multivariado

En esta fase se realizan resúmenes cuantitativos, agrupamientos por categorías, tablas cruzadas de frecuencia y matrices de correlación bivariadas y multivariadas.

### 6.1 `df.describe()` y `Series.describe()`
* **Librería:** Pandas (Método de `DataFrame` y `Series`)
* **¿Qué hace?:** Genera un resumen completo de estadísticas descriptivas: conteo, media, desviación estándar, valor mínimo, percentiles 25%, 50% (mediana), 75% y valor máximo.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.describe()` / `df['columna'].describe()`
  * **Ejemplo de código:**
    ```python
    expvida[['life_expectancy', 'schooling']].describe()
    ```
  * **Output esperado / Resultado:**
    ```text
           life_expectancy    schooling
    count      2928.000000  2775.000000
    mean         69.224932    11.992793
    std           9.523867     3.358920
    min          36.300000     0.000000
    50%          72.100000    12.300000
    max          89.000000    20.700000
    ```

---

### 6.2 Métodos Estadísticos de Agregación Simples
* **Librería:** Pandas (Métodos de `Series` / `DataFrame`)
* **¿Qué hace?:** Computan medidas estadísticas puntuales: `.mean()`, `.median()`, `.min()`, `.max()`, `.quantile()`, `.std()`.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['col'].mean()`, `df['col'].quantile(0.75)`
  * **Ejemplo de código:**
    ```python
    q1 = expvida['life_expectancy'].quantile(0.25)
    q3 = expvida['life_expectancy'].quantile(0.75)
    ric = q3 - q1
    print(f"Q1: {q1}, Q3: {q3}, RIC: {ric}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Q1: 63.1, Q3: 75.7, RIC: 12.6
    ```

---

### 6.3 `Series.unique()`
* **Librería:** Pandas (Método de `Series`)
* **¿Qué hace?:** Devuelve un arreglo NumPy con todos los valores únicos (sin repeticiones) de una columna.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['columna'].unique()`
  * **Ejemplo de código:**
    ```python
    expvida['status'].unique()
    ```
  * **Output esperado / Resultado:**
    ```text
    array(['Developing', 'Developed'], dtype=object)
    ```

---

### 6.4 `Series.value_counts()`
* **Librería:** Pandas (Método de `Series`)
* **¿Qué hace?:** Cuenta la frecuencia absoluta de cada categoría o valor único. Permite verificar el balance de clases en problemas de clasificación.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['columna'].value_counts(normalize=False)`
  * **Ejemplo de código:**
    ```python
    expvida['status'].value_counts()
    ```
  * **Output esperado / Resultado:**
    ```text
    status
    Developing    2426
    Developed      512
    Name: count, dtype: int64
    ```

---

### 6.5 `df.groupby()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Implementa el patrón *Split-Apply-Combine*: separa el DataFrame en grupos por una o más categorías, aplica una función de agregación (media, suma, conteo) y combina el resultado en una tabla resumen.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.groupby('col_cat')['col_num'].mean()`
  * **Ejemplo de código:**
    ```python
    expvida.groupby('status')[['life_expectancy', 'schooling']].mean()
    ```
  * **Output esperado / Resultado:**
    ```text
                life_expectancy  schooling
    status                                
    Developed         79.197852  14.843750
    Developing        67.111465  11.383177
    ```

---

### 6.6 `pd.crosstab()`
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Computa una tabla de contingencia o frecuencia cruzada entre dos o más variables categóricas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.crosstab(df['cat1'], df['cat2'], normalize=False)`
  * **Ejemplo de código:**
    ```python
    pd.crosstab(expvida['status'], expvida['rango_vida'])
    ```
  * **Output esperado / Resultado:**
    ```text
    rango_vida  Baja  Media  Alta
    status                       
    Developed      0     72   440
    Developing   444   1491   481
    ```

---

### 6.7 `df.corr()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Calcula la matriz simétrica de correlación lineal de Pearson (por defecto) entre todas las columnas numéricas del DataFrame. Los valores oscilan entre `-1` (correlación negativa perfecta) y `+1` (correlación positiva perfecta).
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.corr(numeric_only=True)`
  * **Ejemplo de código:**
    ```python
    expvida[['life_expectancy', 'schooling', 'adult_mortality']].corr()
    ```
  * **Output esperado / Resultado:**
    ```text
                     life_expectancy  schooling  adult_mortality
    life_expectancy         1.000000   0.751975        -0.696359
    schooling               0.751975   1.000000        -0.454621
    adult_mortality        -0.696359  -0.454621         1.000000
    ```

---

### 6.8 `df.corrwith()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Computa la correlación bivariada entre cada una de las columnas numéricas de un DataFrame y una Serie objetivo independiente (`y`). En Machine Learning de regresión es fundamental para identificar de forma directa qué variables presentan mayor asociación lineal con la variable a predecir.
* **¿Cómo usarla?:**
  * **Sintaxis:** `X.corrwith(y).sort_values()`
  * **Ejemplo de código:**
    ```python
    X_num = expvida.select_dtypes(include=['float64', 'int64']).drop(columns=['life_expectancy'])
    y_target = expvida['life_expectancy']
    correlaciones = X_num.corrwith(y_target).sort_values(ascending=False)
    print(correlaciones.head(3))
    ```
  * **Output esperado / Resultado:**
    ```text
    schooling                          0.751975
    income_composition_of_resources    0.724814
    bmi                                0.567694
    dtype: float64
    ```

---

## Etapa 7: Visualización Exploratoria de Datos (Seaborn y Matplotlib)

En esta etapa se utilizan las funciones de la librería **Seaborn** (`sns`) combinadas con **Matplotlib** (`plt`) para realizar análisis exploratorio gráfico de distribuciones univariadas, relaciones bivariadas, diagramas de dispersión y mapas de calor.

### 7.1 `sns.displot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Dibuja una distribución univariada continua mediante histogramas y opcionalmente una estimación de densidad de kernel (`kde=True`).
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.displot(data=df, x='col', kde=True, color='#6E4AA1')`
  * **Ejemplo de código:**
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.displot(expvida['life_expectancy'], kde=True, color='#6E4AA1')
    plt.title('Distribución de Expectativa de Vida')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Histograma violeta con curva KDE superpuesta mostrando distribución asimétrica.
    ```

---

### 7.2 `sns.histplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Genera histogramas modernos permitiendo segmentar por categorías (`hue`), regular la cantidad de divisiones (`bins`), calcular frecuencias o densidades y trazar la curva suave KDE.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.histplot(data=df, x='col', bins=30, kde=True, hue=None)`
  * **Ejemplo de código:**
    ```python
    plt.figure(figsize=(7, 4))
    sns.histplot(data=expvida, x='life_expectancy', hue='status', bins=25, kde=True, palette='Set2')
    plt.title('Expectativa de Vida según Estado de Desarrollo')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Dos distribuciones superpuestas: países en desarrollo concentrados entre
    50 y 72 años, y países desarrollados desplazados notablemente a la derecha (78 a 85 años).
    ```

---

### 7.3 `sns.countplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Grafica las frecuencias absolutas observadas de una variable categórica mediante barras verticales u horizontales.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.countplot(x='cat_col', data=df, palette='Set2')`
  * **Ejemplo de código:**
    ```python
    sns.countplot(x='status', data=expvida, palette='Set2')
    plt.title('Distribución de Países por Estado')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Gráfico de barras comparando la cantidad de registros 'Developing' (~2400)
    frente a 'Developed' (~500).
    ```

---

### 7.4 `sns.barplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Muestra estimaciones puntuales de una variable cuantitativa (por defecto la media) desglosadas por categorías, agregando automáticamente intervalos de confianza.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.barplot(x='cat', y='num', data=df, palette='pastel')`
  * **Ejemplo de código:**
    ```python
    sns.barplot(x='status', y='life_expectancy', data=expvida, palette='Blues')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Barras comparativas de media: Developed (~79 años) vs Developing (~67 años)
    con sus barras de error de confianza.
    ```

---

### 7.5 `sns.boxplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Muestra la distribución de variables cuantitativas a través de sus cuartiles (Q1, Mediana, Q3) e identifica visualmente puntos aislados que representan valores atípicos (*outliers*).
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.boxplot(x='cat', y='num', data=df)`
  * **Ejemplo de código:**
    ```python
    sns.boxplot(x='status', y='schooling', data=expvida, palette='Set3')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Dos diagramas de caja mostrando mayor dispersión y presencia de outliers
    bajos en países en desarrollo en comparación con países desarrollados.
    ```

---

### 7.6 `sns.scatterplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Diagrama de dispersión bivariado para examinar relaciones y correlaciones lineales o no lineales entre dos variables continuas, permitiendo codificar variables adicionales mediante color (`hue`) o tamaño (`size`).
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.scatterplot(x='col_x', y='col_y', hue='cat', data=df)`
  * **Ejemplo de código:**
    ```python
    sns.scatterplot(x='schooling', y='life_expectancy', hue='status', data=expvida, alpha=0.6)
    plt.title('Expectativa de Vida vs. Escolaridad')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Nube de puntos con clara tendencia ascendente positiva: a mayor escolaridad,
    mayor expectativa de vida.
    ```

---

### 7.7 `sns.pairplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Traza una grilla completa de gráficos de dispersión bivariados para todas las combinaciones de columnas numéricas del dataset, junto con histogramas o KDE univariados en la diagonal principal.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.pairplot(df[['col1', 'col2', 'col3']], hue='status')`
  * **Ejemplo de código:**
    ```python
    sns.pairplot(expvida[['life_expectancy', 'schooling', 'adult_mortality', 'status']], hue='status')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Matriz 3x3 de subgráficos cruzando las 3 variables numéricas,
    diferenciando las muestras por color según el estado del país.
    ```

---

### 7.8 `sns.heatmap()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Representa valores bidimensionales (típicamente una matriz de correlación obtenida con `df.corr()`) en un mapa de calor codificado por colores, con opción de mostrar los valores numéricos exactos (`annot=True`).
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)`
  * **Ejemplo de código:**
    ```python
    matriz_corr = expvida[['life_expectancy', 'schooling', 'adult_mortality', 'gdp']].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Mapa de Calor de Correlaciones')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Matriz de celdas coloreadas en azul (correlación negativa) y rojo (positiva),
    destacando correlación fuerte de +0.75 entre escolaridad y expectativa de vida.
    ```

---

## Etapa 8: Estilizado y Personalización Visual de Gráficos

### 8.1 `sns.despine()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Remueve los ejes superiores y derechos ("espinas") de los gráficos creados con Seaborn o Matplotlib, logrando un diseño visual moderno, profesional y minimalista.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.despine(top=True, right=True, left=False, bottom=False)`
  * **Ejemplo de código:**
    ```python
    sns.histplot(expvida['life_expectancy'], kde=True)
    sns.despine()
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Efecto Visual]: El marco rectangular alrededor de la gráfica pierde los bordes superior y derecho,
    dejando únicamente los ejes cartesianos X e Y.
    ```

---

## Etapa 9: Preprocesamiento, Escalado, Transformación y Pipelines con Scikit-Learn

En esta etapa se aplican los estimadores y transformadores de **Scikit-Learn** (`sklearn`) para imputación, codificación de variables categóricas, escalado, transformación de forma y encadenamiento seguro con *Pipelines* y *ColumnTransformers*.

### 9.1 `SimpleImputer` (Imputación Automática de Valores Faltantes)
* **Librería:** Scikit-Learn (`sklearn.impute.SimpleImputer`)
* **¿Qué hace?:** Reemplaza automáticamente valores faltantes (`NaN`) en matrices o DataFrames numéricos o categóricos según estrategias univariadas: media (`"mean"`), mediana (`"median"`), moda (`"most_frequent"`) o valor constante (`"constant"`).
* **¿Cómo usarla?:**
  * **Sintaxis:** `imputer = SimpleImputer(strategy='median')`
  * **Ejemplo de código:**
    ```python
    from sklearn.impute import SimpleImputer
    import numpy as np

    imputer = SimpleImputer(strategy='median')
    expvida['schooling_imp'] = imputer.fit_transform(expvida[['schooling']])
    print("Nulos restantes en schooling:", expvida['schooling_imp'].isna().sum())
    ```
  * **Output esperado / Resultado:**
    ```text
    Nulos restantes en schooling: 0
    ```

---

### 9.2 `LabelEncoder` (Codificación Ordinal / de Etiquetas) y su Regla de Uso
* **Librería:** Scikit-Learn (`sklearn.preprocessing.LabelEncoder`)
* **¿Qué hace?:** Convierte etiquetas cualitativas o cadenas de texto en números enteros consecutivos ($0, 1, 2, \dots$).
* > ⚠️ **Advertencia Metodológica Fundamental (Notebook 13):**
  > `LabelEncoder` asigna un orden numérico implícito ($0 < 1 < 2$). Si se utiliza en variables predictoras ($X$), modelos basados en distancias (kNN, Regresión Lineal, Regresión Logística, SVM) asumirán artificialmente que la categoría codificada con $2$ es mayor o doble que la codificada con $1$.
  > * **Regla:** Utilizar `LabelEncoder` **únicamente** para codificar el vector objetivo categórico ($y$).
  > * Para variables predictoras ($X$), debe usarse **`OneHotEncoder`** o **`pd.get_dummies()`**.
* **¿Cómo usarla?:**
  * **Sintaxis:** `le = LabelEncoder(); y_encoded = le.fit_transform(y)`
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_clf = le.fit_transform(expvida['status'])
    print("Clases aprendidas:", le.classes_)
    print("Primeros valores codificados:", y_clf[:5])
    ```
  * **Output esperado / Resultado:**
    ```text
    Clases aprendidas: ['Developed' 'Developing']
    Primeros valores codificados: [1 1 1 1 1]
    ```

---

### 9.3 `OneHotEncoder` (Codificación Categórica Nominal)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.OneHotEncoder`)
* **¿Qué hace?:** Convierte variables categóricas cualitativas en matrices binarias ($0$ y $1$). Conserva el mapeo de categorías aprendidas (`categories_`), lo que permite transformar nuevos datos en test o producción sin riesgo de incompatibilidad dimensional.
* **¿Cómo usarla?:**
  * **Sintaxis:** `ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')`
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder(sparse_output=False, drop='first')
    status_ohe = ohe.fit_transform(expvida[['status']])
    print("Nombres de columnas generadas:", ohe.get_feature_names_out())
    ```
  * **Output esperado / Resultado:**
    ```text
    Nombres de columnas generadas: ['status_Developing']
    ```

---

### 9.4 `MinMaxScaler` (Re-escalado de Características a un Rango)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.MinMaxScaler`)
* **¿Qué hace?:** Transforma las variables numéricas para que queden acotadas exactamente dentro de un rango determinado (por defecto entre $0$ y $1$):
  $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
* **¿Cómo usarla?:**
  * **Sintaxis:** `scaler = MinMaxScaler(feature_range=(0, 1))`
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import MinMaxScaler

    scaler_mm = MinMaxScaler()
    schooling_mm = scaler_mm.fit_transform(expvida[['schooling']].dropna())
    print("Mínimo:", schooling_mm.min(), " Máximo:", schooling_mm.max())
    ```
  * **Output esperado / Resultado:**
    ```text
    Mínimo: 0.0  Máximo: 1.0
    ```

---

### 9.5 `StandardScaler` (Estandarización / Escala Z)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.StandardScaler`)
* **¿Qué hace?:** Estandariza las variables cuantitativas restando la media muestral ($\mu$) y dividiendo por la desviación estándar ($\sigma$), transformándolas para que tengan media igual a $0$ y varianza unitaria ($1$):
  $$Z = \frac{x - \mu}{\sigma}$$
* **¿Cómo usarla?:**
  * **Sintaxis:** `scaler = StandardScaler()`
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import StandardScaler

    scaler_std = StandardScaler()
    schooling_std = scaler_std.fit_transform(expvida[['schooling']].dropna())
    print("Media resultante:", schooling_std.mean().round(4))
    print("Desvío resultante:", schooling_std.std().round(4))
    ```
  * **Output esperado / Resultado:**
    ```text
    Media resultante: -0.0
    Desvío resultante: 1.0
    ```

---

### 9.6 `RobustScaler` (Escalado Robusto Resistente a Outliers)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.RobustScaler`)
* **¿Qué hace?:** Escala variables cuantitativas basándose en la **mediana** ($Q2$) y el **rango intercuartílico** ($	ext{RIC} = Q3 - Q1$). Es la mejor opción cuando los datos contienen *outliers* significativos que no deben eliminarse, evitando que las medias o varianzas distorsionen la escala.
  $$x' = \frac{x - 	ext{mediana}}{	ext{RIC}}$$
* **¿Cómo usarla?:**
  * **Sintaxis:** `scaler = RobustScaler()`
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import RobustScaler

    scaler_rb = RobustScaler()
    gdp_rb = scaler_rb.fit_transform(expvida[['gdp']].dropna())
    print("Mediana resultante:", np.median(gdp_rb).round(4))
    ```
  * **Output esperado / Resultado:**
    ```text
    Mediana resultante: 0.0
    ```

---

### 9.7 `Normalizer` (Normalización por Normas Vectoriales de Muestras)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.Normalizer`)
* **¿Qué hace?:** Escala muestras individuales (filas) para que tengan una norma vectorial unitaria ($\|x\|_2 = 1$).
  > ⚠️ **Advertencia Metodológica Clave:** `Normalizer` opera **por filas (horizontalmente)** y no por columnas. Si se aplica a una sola columna individual, la norma de cualquier escalar es él mismo, devolviendo siempre $1.0$. Solo debe usarse sobre múltiples variables cuando interesa la dirección y ángulo del vector (por ejemplo minería de texto o perfiles de clientes).
* **¿Cómo usarla?:**
  * **Sintaxis:** `normalizer = Normalizer(norm='l2')`
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import Normalizer

    norm = Normalizer(norm='l2')
    matriz_norm = norm.fit_transform(expvida[['schooling', 'adult_mortality']].dropna())
    print("Norma euclidiana de la primera fila:", np.linalg.norm(matriz_norm[0]).round(4))
    ```
  * **Output esperado / Resultado:**
    ```text
    Norma euclidiana de la primera fila: 1.0
    ```

---

### 9.8 `PowerTransformer` (Transformación de Potencia Yeo-Johnson y Box-Cox)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.PowerTransformer`)
* **¿Qué hace?:** Aplica transformaciones de potencia estabilizadoras de varianza para aproximar distribuciones asimétricas a distribuciones Gaussianas normales. Soporta `method='yeo-johnson'` (admite números positivos y negativos/cero) y `method='box-cox'` (solo valores estrictamente positivos).
* **¿Cómo usarla?:**
  * **Sintaxis:** `pt = PowerTransformer(method='yeo-johnson', standardize=True)`
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import PowerTransformer

    pt = PowerTransformer(method='yeo-johnson')
    gdp_pt = pt.fit_transform(expvida[['gdp']].dropna())
    print("Asimetría antes:", expvida['gdp'].skew().round(3))
    print("Asimetría después:", pd.Series(gdp_pt.ravel()).skew().round(3))
    ```
  * **Output esperado / Resultado:**
    ```text
    Asimetría antes: 3.212
    Asimetría después: 0.052
    ```

---

### 9.9 `np.log1p()` y `np.expm1()` (Transformación Logarítmica de Forma)
* **Librería:** NumPy (`np`)
* **¿Qué hace?:** `np.log1p(x)` computa $\ln(1 + x)$ de forma numéricamente estable para corregir variables con fuerte asimetría positiva que contienen ceros. `np.expm1(x)` aplica $e^x - 1$ como transformación inversa para devolver las predicciones a la escala monetaria o cuantitativa original.
* **¿Cómo usarla?:**
  * **Sintaxis:** `y_log = np.log1p(df['col'])` / `y_orig = np.expm1(y_log)`
  * **Ejemplo de código:**
    ```python
    import numpy as np

    poblacion_log = np.log1p(expvida['population'].dropna())
    print("Asimetría original:", expvida['population'].skew().round(2))
    print("Asimetría logarítmica:", poblacion_log.skew().round(2))
    ```
  * **Output esperado / Resultado:**
    ```text
    Asimetría original: 15.96
    Asimetría logarítmica: 0.21
    ```

---

### 9.10 `Pipeline` (Ensamblaje Secuencial de Preprocesamiento y Modelado)
* **Librería:** Scikit-Learn (`sklearn.pipeline.Pipeline`)
* **¿Qué hace?:** Encadena ordenadamente transformadores sucesivos (imputadores, escaladores) y un estimador final en un único objeto ejecutable. Garantiza que en cada paso se aplique `fit_transform` exclusivamente sobre los datos de entrenamiento y `transform` sobre los datos de prueba, blindando el flujo contra la filtración de datos (*Data Leakage*).
* **¿Cómo usarla?:**
  * **Sintaxis:**
    ```python
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ('imputar', SimpleImputer(strategy='median')),
        ('escalar', StandardScaler())
    ])
    ```
  * **Ejemplo de código:**
    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    pipe_num = Pipeline([
        ('imputador', SimpleImputer(strategy='median')),
        ('escalador', StandardScaler())
    ])

    X_train_proc = pipe_num.fit_transform(X_train[['schooling', 'gdp']])
    X_test_proc  = pipe_num.transform(X_test[['schooling', 'gdp']])
    print("Forma procesada en train:", X_train_proc.shape)
    ```
  * **Output esperado / Resultado:**
    ```text
    Forma procesada en train: (2203, 2)
    ```

---

### 9.11 `ColumnTransformer` (Transformaciones Diferenciadas por Tipo de Columna)
* **Librería:** Scikit-Learn (`sklearn.compose.ColumnTransformer`)
* **¿Qué hace?:** Aplica transformadores o *Pipelines* distintos en paralelo a subconjuntos de columnas específicos (por ejemplo numéricas vs categóricas) dentro del mismo DataFrame, unificando la salida en una única matriz lista para el modelo.
* **¿Cómo usarla?:**
  * **Sintaxis:**
    ```python
    from sklearn.compose import ColumnTransformer
    preprocesador = ColumnTransformer(transformers=[
        ('num', pipeline_numerico, lista_columnas_num),
        ('cat', pipeline_categorico, lista_columnas_cat)
    ])
    ```
  * **Ejemplo de código:**
    ```python
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    cols_num = ['schooling', 'gdp', 'adult_mortality']
    cols_cat = ['status']

    pipe_num = Pipeline([
        ('imputador', SimpleImputer(strategy='median')),
        ('escalador', StandardScaler())
    ])

    pipe_cat = Pipeline([
        ('imputador', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocesador = ColumnTransformer(transformers=[
        ('num', pipe_num, cols_num),
        ('cat', pipe_cat, cols_cat)
    ])

    X_train_final = preprocesador.fit_transform(X_train)
    X_test_final  = preprocesador.transform(X_test)
    print("Matriz unificada final:", X_train_final.shape)
    ```
  * **Output esperado / Resultado:**
    ```text
    Matriz unificada final: (2203, 4)
    ```

---

## Etapa 10: Flujo Correcto de Entrenamiento y Prevención de Data Leakage

En esta etapa se establecen los principios metodológicos fundamentales para separar los conjuntos de datos de entrenamiento y evaluación, garantizando la validez de los modelos predictivos.

### 10.1 `train_test_split()` (División del Dataset en Train y Test)
* **Librería:** Scikit-Learn (`sklearn.model_selection.train_test_split`)
* **¿Qué hace?:** Divide matrices o DataFrames en subconjuntos de Entrenamiento (*Train*) y Prueba/Testeo (*Test*) de forma aleatoria o estratificada (`stratify=y`).
* **¿Cómo usarla?:**
  * **Sintaxis:** `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)`
  * **Ejemplo de código:**
    ```python
    from sklearn.model_selection import train_test_split

    X = expvida[['schooling', 'adult_mortality', 'gdp']]
    y = expvida['life_expectancy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Entrenamiento: {X_train.shape[0]} casos | Prueba: {X_test.shape[0]} casos")
    ```
  * **Output esperado / Resultado:**
    ```text
    Entrenamiento: 2203 casos | Prueba: 735 casos
    ```

---

### 10.2 Regla de Oro: Separación de `fit_transform` en Train vs `transform` en Test
* **Librería:** Principio metodológico transversal de Scikit-Learn.
* **¿Qué hace?:** Previene la **filtración de datos (*Data Leakage*)**. El método `fit` aprende parámetros (medias, varianzas, mínimos, modas) exclusivamente a partir de las muestras del conjunto de **Entrenamiento (`X_train`)**. Luego, dichos parámetros aprendidos se aplican mediante `transform` al conjunto de **Prueba (`X_test`)**.
* **Regla Inflexible de Trabajo:**
  1. `fit_transform()` $
ightarrow$ **SOLO sobre `X_train`**.
  2. `transform()` $
ightarrow$ **SOLO sobre `X_test`** (y futuros datos de producción).
  3. **No se escalan** las variables indicadoras binarias (*dummies* resultantes de One-Hot Encoding).
* **Ejemplo de código:**
  ```python
  from sklearn.preprocessing import StandardScaler

  # Forma metodológicamente correcta
  scaler = StandardScaler()
  X_train_esc = scaler.fit_transform(X_train)  # fit_transform SOLO en train
  X_test_esc  = scaler.transform(X_test)       # solo transform en test

  print("Medias aprendidas únicamente de Train:", scaler.mean_.round(3))
  ```
* **Output esperado / Resultado:**
  ```text
  Medias aprendidas únicamente de Train: [11.954 165.210 7480.123]
  ```

---

## Etapa 11: Fundamentos de Machine Learning y Formulación del Problema

En esta etapa se establecen los conceptos formales de formulación de problemas de aprendizaje automático analizados en las Notebooks 14 y 15.

### 11.1 Los Cuatro Componentes de Mitchell ($T$, $E$, $P$, $A$)
* **Concepto Teórico:** Según la definición clásica de Tom Mitchell (1997), se dice que un programa de computadora aprende de la experiencia si su rendimiento en determinadas tareas mejora con dicha experiencia. Todo proyecto de Machine Learning se estructura formalmente sobre cuatro componentes:
  1. **Tarea ($T$):** La labor específica que se desea que el modelo resuelva (ej. clasificar si un estudiante aprueba o predecir la expectativa de vida en años).
  2. **Experiencia ($E$):** Los datos históricos organizados disponibles a partir de los cuales el algoritmo aprende patrones.
  3. **Medida de Rendimiento ($P$):** La métrica cuantitativa con la que se evalúa objetivamente la calidad del modelo (ej. Exactitud en clasificación, MAE o $R^2$ en regresión).
  4. **Algoritmo ($A$):** El procedimiento matemático o computacional que busca y ajusta los parámetros del modelo para optimizar $P$.

---

### 11.2 Paradigmas: Aprendizaje Supervisado vs. No Supervisado
* **Aprendizaje Supervisado (*Supervised Learning*):** Los datos de entrenamiento cuentan con una variable objetivo o etiqueta conocida ($y$). El objetivo del modelo es aprender una función de mapeo $f(X) \approx y$ capaz de predecir la etiqueta ante nuevas observaciones nunca antes vistas.
* **Aprendizaje No Supervisado (*Unsupervised Learning*):** Los datos carecen de etiquetas objetivo ($y$). El algoritmo explora la estructura intrínseca de los datos para descubrir agrupamientos naturales (*clustering* como K-Means), asociaciones o reducciones de dimensionalidad.

---

### 11.3 Tareas Supervisadas: Regresión vs. Clasificación ($X$ vs. $y$)
* **Matriz de Características ($X$):** Arreglo bidimensional de tamaño $(N, M)$ donde cada fila representa una observación y cada columna una variable predictora.
* **Vector Objetivo ($y$):** Arreglo unidimensional de tamaño $(N,)$ que contiene la verdad terreno o valor que se desea predecir.
* **Criterio de Clasificación de Problemas:**
  * **Regresión:** La variable objetivo $y$ es **cuantitativa continua** (ej. precios, temperaturas, años de vida).
  * **Clasificación:** La variable objetivo $y$ es **cualitativa discreta / categórica** (ej. binaria: `1` aprueba / `0` no aprueba; o multiclase).

---

### 11.4 Subajuste (*Underfitting*), Sobreajuste (*Overfitting*) y Compensación Sesgo-Varianza
* **Subajuste (*Underfitting* / Alto Sesgo):** El modelo es demasiado simple para capturar los patrones de los datos. Presenta un rendimiento pobre tanto en el conjunto de entrenamiento como en el de prueba.
* **Sobreajuste (*Overfitting* / Alta Varianza):** El modelo es excesivamente complejo y memoriza el ruido específico de los datos de entrenamiento. Obtiene métricas casi perfectas en Train, pero rinde mal en Test.
* **Compensación Sesgo-Varianza (*Bias-Variance Trade-off*):** El punto óptimo de generalización se encuentra regulando la complejidad del modelo (por ejemplo limitando la profundidad `max_depth` en árboles de decisión).

---

## Etapa 12: Modelos de Línea Base / Pisos de Referencia (*Baselines*)

Antes de entrenar algoritmos complejos, es metodológicamente obligatorio construir un modelo de referencia trivial (*baseline* o "modelo piso"). Si un algoritmo de Machine Learning no supera con creces el rendimiento de este modelo base, no está aportando valor real y no debe desplegarse.

### 12.1 `DummyClassifier`
* **Librería:** Scikit-Learn (`sklearn.dummy.DummyClassifier`)
* **¿Qué hace?:** Clasificador trivial que realiza predicciones utilizando reglas fijas sin analizar las variables predictoras ($X$). La estrategia más común (`strategy='most_frequent'`) predice siempre la clase mayoritaria del conjunto de entrenamiento.
* **¿Cómo usarla?:**
  * **Sintaxis:** `dummy = DummyClassifier(strategy='most_frequent')`
  * **Ejemplo de código:**
    ```python
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score

    # Entrenar modelo piso en clasificación
    dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy_clf.fit(X_train, y_train_clf)

    acc_piso = accuracy_score(y_test_clf, dummy_clf.predict(X_test))
    print(f"Piso de referencia (Dummy): {acc_piso:.1%}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Piso de referencia (Dummy): 82.6%
    ```
    *(Nota: Un modelo que logre un 83% de exactitud solo supera al piso por un 0.4%, demostrando que casi no aprendió).*

---

### 12.2 `DummyRegressor`
* **Librería:** Scikit-Learn (`sklearn.dummy.DummyRegressor`)
* **¿Qué hace?:** Regresor trivial que predice siempre una constante representativa del target en Train, típicamente la media (`strategy='mean'`) o la mediana (`strategy='median'`).
* **¿Cómo usarla?:**
  * **Sintaxis:** `dummy_reg = DummyRegressor(strategy='mean')`
  * **Ejemplo de código:**
    ```python
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error

    dummy_reg = DummyRegressor(strategy='mean')
    dummy_reg.fit(X_train, y_train_reg)

    mae_piso = mean_absolute_error(y_test_reg, dummy_reg.predict(X_test))
    print(f"MAE del piso de referencia: {mae_piso:.2f} años")
    ```
  * **Output esperado / Resultado:**
    ```text
    MAE del piso de referencia: 7.64 años
    ```

---

## Etapa 13: Algoritmos de Aprendizaje Supervisado

En esta etapa se desarrollan los estimadores supervisados para tareas de regresión y clasificación vistos en las Notebooks 14, 16 y 17.

### 13.1 `LinearRegression` (Regresión Lineal Múltiple)
* **Librería:** Scikit-Learn (`sklearn.linear_model.LinearRegression`)
* **¿Qué hace?:** Ajusta un modelo lineal por mínimos cuadrados ordinarios para estimar una relación continua entre las variables predictoras y la variable objetivo:
  $$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p$$
  Los coeficientes $w_i$ se acceden con `.coef_` y el intercepto $w_0$ con `.intercept_`.
* **¿Cómo usarla?:**
  * **Sintaxis:** `lr = LinearRegression()`
  * **Ejemplo de código:**
    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_reg)
    y_pred_lr = lr.predict(X_test_scaled)

    print(f"MAE Regresión Lineal: {mean_absolute_error(y_test_reg, y_pred_lr):.2f}")
    print(f"R2 Regresión Lineal:  {r2_score(y_test_reg, y_pred_lr):.3f}")
    ```
  * **Output esperado / Resultado:**
    ```text
    MAE Regresión Lineal: 2.95
    R2 Regresión Lineal:  0.812
    ```

---

### 13.2 `LogisticRegression` (Regresión Logística para Clasificación)
* **Librería:** Scikit-Learn (`sklearn.linear_model.LogisticRegression`)
* **¿Qué hace?:** Clasificador lineal que modela la probabilidad de que una observación pertenezca a la clase positiva utilizando la función logística sigmoide:
  $$P(y=1|x) = \frac{1}{1 + e^{-z}}$$
  Requiere escalado previo de las variables cuantitativas para garantizar la convergencia del optimizador numérico.
* **¿Cómo usarla?:**
  * **Sintaxis:** `log_reg = LogisticRegression(max_iter=1000, random_state=42)`
  * **Ejemplo de código:**
    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train_clf)
    y_pred_log = log_reg.predict(X_test_scaled)

    print(f"Exactitud Regresión Logística: {accuracy_score(y_test_clf, y_pred_log):.3f}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Exactitud Regresión Logística: 0.912
    ```

---

### 13.3 `DecisionTreeClassifier` y `DecisionTreeRegressor` (Árboles de Decisión)
* **Librería:** Scikit-Learn (`sklearn.tree.DecisionTreeClassifier` / `DecisionTreeRegressor`)
* **¿Qué hace?:** Modela relaciones no lineales dividiendo recursivamente el espacio de características mediante reglas condicionales ("si $x_i \le 	ext{umbral}$"). Son altamente interpretables y no requieren escalado de variables. El hiperparámetro fundamental **`max_depth`** controla la profundidad máxima para evitar el sobreajuste (*overfitting*). Además, provee el atributo `.feature_importances_` para evaluar qué variables aportan más a las decisiones.
* **¿Cómo usarla?:**
  * **Sintaxis:**
    * Clasificación: `tree = DecisionTreeClassifier(max_depth=3, random_state=42)`
    * Regresión: `tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42)`
  * **Ejemplo de código:**
    ```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train_clf)
    y_pred_tree = tree.predict(X_test)

    print(f"Exactitud Árbol (max_depth=3): {accuracy_score(y_test_clf, y_pred_tree):.3f}")
    print("Importancia de características:")
    print(pd.Series(tree.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(3))
    ```
  * **Output esperado / Resultado:**
    ```text
    Exactitud Árbol (max_depth=3): 0.924
    Importancia de características:
    schooling                          0.684
    income_composition_of_resources    0.210
    adult_mortality                    0.106
    dtype: float64
    ```

---

### 13.4 `plot_tree()` (Visualización Gráfica de Árboles)
* **Librería:** Scikit-Learn (`sklearn.tree.plot_tree`)
* **¿Qué hace?:** Renderiza visualmente la estructura completa del árbol de decisión ajustado, mostrando las preguntas de corte en cada nodo, la impureza (Gini o MSE), la cantidad de muestras y la clase predominante.
* **¿Cómo usarla?:**
  * **Sintaxis:** `plot_tree(tree, feature_names=..., class_names=..., filled=True, ax=ax)`
  * **Ejemplo de código:**
    ```python
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(16, 6))
    plot_tree(tree, feature_names=X_train.columns, class_names=['Developed', 'Developing'],
              filled=True, rounded=True, ax=ax)
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Diagrama jerárquico de nodos coloreados según la clase predicha,
    mostrando las condiciones lógicas de decisión desde la raíz hasta las hojas.
    ```

---

### 13.5 `KNeighborsClassifier` (k-Nearest Neighbors / k-Vecinos Más Cercanos)
* **Librería:** Scikit-Learn (`sklearn.neighbors.KNeighborsClassifier`)
* **¿Qué hace?:** Clasifica cada nueva muestra según el voto mayoritario de sus $K$ vecinos más cercanos en el espacio de características mediante distancia Euclidiana.
  > ⚠️ **Requisito Metodológico Estricto:** Al basarse íntegramente en distancias geométricas, **requiere obligatoriamente escalado previo** (`StandardScaler` o `MinMaxScaler`). Una variable con números grandes distorsionaría por completo la distancia.
  * $K$ muy chico ($K=1$): frontera de decisión muy irregular y sensible al ruido (**sobreajuste**).
  * $K$ muy grande: frontera excesivamente suavizada (**subajuste**).
* **¿Cómo usarla?:**
  * **Sintaxis:** `knn = KNeighborsClassifier(n_neighbors=5)`
  * **Ejemplo de código:**
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)

    print(f"Exactitud kNN (K=5): {accuracy_score(y_test, y_pred_knn):.3f}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Exactitud kNN (K=5): 0.932
    ```

---

### 13.6 `SVC` (Support Vector Classifier / Máquinas de Vectores de Soporte)
* **Librería:** Scikit-Learn (`sklearn.svm.SVC`)
* **¿Qué hace?:** Busca el hiperplano óptimo que separa las clases maximizando el **margen** de separación entre los puntos más cercanos de cada clase (los *vectores de soporte*).
  * `kernel='linear'`: Separa linealmente mediante hiperplanos planos.
  * `kernel='rbf'` (*Radial Basis Function*): Proyecta las características a un espacio de dimensión superior para resolver fronteras no lineales complejas.
  * Requiere siempre datos estandarizados.
* **¿Cómo usarla?:**
  * **Sintaxis:** `svm = SVC(kernel='rbf', random_state=42)`
  * **Ejemplo de código:**
    ```python
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    svm_rbf = SVC(kernel='rbf', random_state=42)
    svm_rbf.fit(X_train_scaled, y_train)
    y_pred_svm = svm_rbf.predict(X_test_scaled)

    print(f"Exactitud SVM (RBF): {accuracy_score(y_test, y_pred_svm):.3f}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Exactitud SVM (RBF): 0.941
    ```

---

### 13.7 `RandomForestClassifier` (Bosques Aleatorios / Ensamble Bagging)
* **Librería:** Scikit-Learn (`sklearn.ensemble.RandomForestClassifier`)
* **¿Qué hace?:** Modelo de ensamble basado en *Bagging* (*Bootstrap Aggregating*). Entrena una multitud de árboles de decisión independientes (`n_estimators`), cada uno entrenado sobre un subconjunto aleatorio de observaciones y evaluando un subconjunto aleatorio de características en cada nodo. Combina sus predicciones por votación mayoritaria, reduciendo drásticamente la varianza y ofreciendo alta resistencia al sobreajuste.
* **¿Cómo usarla?:**
  * **Sintaxis:** `rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)`
  * **Ejemplo de código:**
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    print(f"Exactitud Random Forest (100 árboles): {accuracy_score(y_test, y_pred_rf):.3f}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Exactitud Random Forest (100 árboles): 0.958
    ```

---

## Etapa 14: Métricas de Evaluación y Diagnóstico del Rendimiento

En esta etapa se detallan las métricas cuantitativas estándar de evaluación analizadas en las Notebooks 14, 16 y 17 para medir objetivamente la calidad de los modelos.

### 14.1 Métricas de Regresión: `mean_absolute_error` (MAE) y `mean_squared_error` (MSE/RMSE)
* **Librería:** Scikit-Learn (`sklearn.metrics`)
* **¿Qué hace?:**
  * **MAE (Error Absoluto Medio):** Promedio de las diferencias absolutas entre valores reales y predichos: $	ext{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$. Es fácilmente interpretable ya que se expresa en las mismas unidades que la variable objetivo.
  * **MSE (Error Cuadrático Medio) / RMSE:** Promedio de los errores al cuadrado: $	ext{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$. Penaliza de forma mucho más severa los errores grandes.
* **¿Cómo usarla?:**
  * **Sintaxis:**
    * `mae = mean_absolute_error(y_true, y_pred)`
    * `mse = mean_squared_error(y_true, y_pred)`
  * **Ejemplo de código:**
    ```python
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    mae = mean_absolute_error(y_test_reg, y_pred_lr)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_lr))
    print(f"MAE:  {mae:.2f} años")
    print(f"RMSE: {rmse:.2f} años")
    ```
  * **Output esperado / Resultado:**
    ```text
    MAE:  2.95 años
    RMSE: 3.82 años
    ```

---

### 14.2 Métricas de Regresión: `r2_score` ($R^2$ - Coeficiente de Determinación)
* **Librería:** Scikit-Learn (`sklearn.metrics.r2_score`)
* **¿Qué hace?:** Cuantifica la proporción de la varianza total de la variable objetivo que es explicada por las variables predictoras del modelo:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  * $R^2 = 1.0$: Predicción perfecta.
  * $R^2 = 0.0$: El modelo equivale a predecir siempre la media (igual que el `DummyRegressor`).
  * $R^2 < 0$: El modelo es peor que la simple media de los datos.
* **¿Cómo usarla?:**
  * **Sintaxis:** `r2 = r2_score(y_true, y_pred)`
  * **Ejemplo de código:**
    ```python
    from sklearn.metrics import r2_score

    r2 = r2_score(y_test_reg, y_pred_lr)
    print(f"Coeficiente R2: {r2:.3f}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Coeficiente R2: 0.812
    ```

---

### 14.3 Métricas de Clasificación: `accuracy_score` (Exactitud Global)
* **Librería:** Scikit-Learn (`sklearn.metrics.accuracy_score`)
* **¿Qué hace?:** Mide la proporción de predicciones correctas sobre el total de casos evaluados:
  $$	ext{Exactitud} = \frac{	ext{VP} + 	ext{VN}}{	ext{Total}}$$
  > ⚠️ **Advertencia sobre clases desbalanceadas:** Si el 90% de los casos pertenecen a la clase $0$, un modelo trivial que siempre prediga $0$ tendrá 90% de exactitud sin haber aprendido nada. Por eso debe cotejarse siempre contra el `DummyClassifier` y complementarse con la matriz de confusión.
* **¿Cómo usarla?:**
  * **Sintaxis:** `acc = accuracy_score(y_true, y_pred)`
  * **Ejemplo de código:**
    ```python
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_test, y_pred_rf)
    print(f"Exactitud global: {acc:.1%}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Exactitud global: 95.8%
    ```

---

### 14.4 `confusion_matrix` y `ConfusionMatrixDisplay`
* **Librería:** Scikit-Learn (`sklearn.metrics.confusion_matrix`, `ConfusionMatrixDisplay`)
* **¿Qué hace?:** La **matriz de confusión** desglosa detalladamente las predicciones cruzando los valores reales con los predichos en cuatro cuadrantes:
  * **Verdaderos Positivos (VP):** Casos positivos correctamente clasificados.
  * **Verdaderos Negativos (VN):** Casos negativos correctamente clasificados.
  * **Falsos Positivos (FP) - Error Tipo I:** Casos negativos predichos erróneamente como positivos.
  * **Falsos Negativos (FN) - Error Tipo II:** Casos positivos predichos erróneamente como negativos.
  `ConfusionMatrixDisplay.from_estimator()` grafica esta matriz como un mapa de calor etiquetado.
* **¿Cómo usarla?:**
  * **Sintaxis:** `ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, display_labels=...)`
  * **Ejemplo de código:**
    ```python
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        rf, X_test_scaled, y_test,
        display_labels=['Developed', 'Developing'],
        cmap='Blues', ax=ax
    )
    plt.title('Matriz de Confusión - Random Forest')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Matriz 2x2 donde se aprecian los aciertos en la diagonal principal
    y los errores de clasificación en las celdas fuera de la diagonal.
    ```

---

### 14.5 `classification_report` (Precisión, Recall, F1-Score y Soporte)
* **Librería:** Scikit-Learn (`sklearn.metrics.classification_report`)
* **¿Qué hace?:** Genera un informe de texto exhaustivo con las métricas fundamentales desglosadas por cada clase individual:
  * **Precisión (*Precision*):** $\frac{	ext{VP}}{	ext{VP} + 	ext{FP}}$ (De todos los que el modelo predijo como positivos, ¿cuántos lo eran realmente?).
  * **Exhaustividad (*Recall* / Sensibilidad):** $\frac{	ext{VP}}{	ext{VP} + 	ext{FN}}$ (De todos los positivos reales que existían, ¿cuántos logró detectar el modelo?).
  * **F1-Score:** Media armónica entre Precisión y Recall: $2 \cdot \frac{	ext{Precisión} \cdot 	ext{Recall}}{	ext{Precisión} + 	ext{Recall}}$.
  * **Soporte (*Support*):** Cantidad de muestras reales de cada clase en el conjunto de prueba.
* **¿Cómo usarla?:**
  * **Sintaxis:** `print(classification_report(y_true, y_pred, target_names=...))`
  * **Ejemplo de código:**
    ```python
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred_rf, target_names=['Developed', 'Developing']))
    ```
  * **Output esperado / Resultado:**
    ```text
                  precision    recall  f1-score   support

       Developed       0.91      0.84      0.87       128
      Developing       0.97      0.98      0.97       607

        accuracy                           0.96       735
       macro avg       0.94      0.91      0.92       735
    weighted avg       0.96      0.96      0.96       735
    ```

---

## Etapa 15: Validación Robusta y Curvas de Aprendizaje

En esta etapa se aplican técnicas de validación cruzada y curvas de aprendizaje para evaluar la estabilidad de los modelos y diagnosticar problemas de sesgo vs. varianza.

### 15.1 `cross_val_score` (Validación Cruzada K-Fold)
* **Librería:** Scikit-Learn (`sklearn.model_selection.cross_val_score`)
* **¿Qué hace?:** Divide el conjunto de entrenamiento en $K$ particiones o pliegues (*folds*). Entrena el modelo en $K-1$ pliegues y evalúa en el pliegue restante de forma rotativa $K$ veces. Devuelve un arreglo con los puntajes obtenidos en cada iteración, permitiendo computar el promedio y la desviación estándar para obtener una estimación robusta que no depende de la suerte de una partición única.
* **¿Cómo usarla?:**
  * **Sintaxis:** `scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')`
  * **Ejemplo de código:**
    ```python
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')

    print(f"Scores por fold: {scores.round(3)}")
    print(f"Exactitud media CV: {scores.mean():.3f} (+/- {scores.std():.3f})")
    ```
  * **Output esperado / Resultado:**
    ```text
    Scores por fold: [0.955 0.961 0.950 0.964 0.959]
    Exactitud media CV: 0.958 (+/- 0.005)
    ```

---

### 15.2 `learning_curve` (Curvas de Diagnóstico de Aprendizaje)
* **Librería:** Scikit-Learn (`sklearn.model_selection.learning_curve`)
* **¿Qué hace?:** Evalúa el rendimiento del modelo en Train y Validación a medida que se incrementa progresivamente la cantidad de muestras de entrenamiento ($N$). Permite diagnosticar con certeza:
  * Si el modelo sufre de **alto sesgo (subajuste):** Ambas curvas convergen rápidamente pero a un nivel de puntaje bajo.
  * Si el modelo sufre de **alta varianza (sobreajuste):** Hay una gran brecha (*gap*) persistente entre la curva de Train (muy alta) y la de Validación (notoriamente más baja).
* **¿Cómo usarla?:**
  * **Sintaxis:** `tamanos, score_train, score_val = learning_curve(modelo, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5))`
  * **Ejemplo de código:**
    ```python
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    tamanos, train_scores, test_scores = learning_curve(
        DecisionTreeClassifier(max_depth=3, random_state=42),
        X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5)
    )

    plt.figure(figsize=(7, 4))
    plt.plot(tamanos, train_scores.mean(axis=1), 'o-', color='#6E4AA1', label='Entrenamiento')
    plt.plot(tamanos, test_scores.mean(axis=1), 'o-', color='#35BEAE', label='Validación')
    plt.xlabel('Cantidad de muestras')
    plt.ylabel('Exactitud')
    plt.title('Curva de Aprendizaje')
    plt.legend()
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Dos curvas convergiendo hacia un rendimiento conjunto de ~92%,
    confirmando que el árbol generaliza correctamente sin sobreajuste severo.
    ```

---

## Etapa 16: Optimización de Hiperparámetros con Optuna

En la Notebook 17 se introduce **Optuna**, un framework moderno y eficiente de optimización Bayesiana que reemplaza la búsqueda exhaustiva por grilla (*GridSearchCV*) mediante un muestreo probabilístico inteligente que aprende de los intentos previos (*Tree-structured Parzen Estimator - TPE*).

### 16.1 Flujo de Búsqueda y Optimización Bayesiana con `optuna`
* **Librería:** Optuna (`import optuna`)
* **Componentes del Flujo de Trabajo:**
  1. **Función Objetivo (`objective(trial)`):** Función que define el espacio de búsqueda usando el objeto `trial` y retorna la métrica a optimizar (usualmente evaluada con validación cruzada sobre Train).
  2. **Muestreo de Hiperparámetros:**
     * `trial.suggest_int('param', min, max)`: Enteros (ej. `n_estimators`, `max_depth`).
     * `trial.suggest_float('param', min, max, log=True)`: Reales continuos (ej. tasa de aprendizaje o regularización $C$).
     * `trial.suggest_categorical('param', ['opc1', 'opc2'])`: Opciones discretas (ej. tipos de kernel).
  3. **Estudio (`optuna.create_study`):** Administrador de la búsqueda donde se define si se desea maximizar (`direction='maximize'`) o minimizar (`direction='minimize'`).
  4. **Optimización (`study.optimize`):** Ejecuta la cantidad de pruebas especificadas (`n_trials`).
  5. **Reentrenamiento Final:** Se extraen los mejores hiperparámetros (`study.best_params`) y se entrena un modelo final definitivo sobre todo el conjunto de entrenamiento para ser evaluado en el conjunto de prueba (`X_test`).
* **¿Cómo usarla?:**
  * **Ejemplo de código:**
    ```python
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    # Silenciar logs verbosos
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 1. Definir la función objetivo
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='accuracy')
        return scores.mean()

    # 2. Crear y ejecutar el estudio
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25, show_progress_bar=False)

    print("Mejor exactitud obtenida en CV:", round(study.best_value, 4))
    print("Mejores hiperparámetros encontrados:", study.best_params)

    # 3. Entrenar el modelo final con los mejores parámetros y evaluar en Test
    rf_optuna = RandomForestClassifier(**study.best_params, random_state=42)
    rf_optuna.fit(X_train_scaled, y_train)
    acc_test_optuna = accuracy_score(y_test, rf_optuna.predict(X_test_scaled))
    print(f"Exactitud definitiva en Test: {acc_test_optuna:.3f}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Mejor exactitud obtenida en CV: 0.9632
    Mejores hiperparámetros encontrados: {'n_estimators': 142, 'max_depth': 11, 'min_samples_split': 4}
    Exactitud definitiva en Test: 0.965
    ```

---

## Etapa 17: Exportación y Almacenamiento de Datos

En esta etapa final se persisten y guardan los datos limpios y procesados en disco en formatos estándar para su consumo posterior en producción, modelos o reportes analíticos.

### 17.1 `df.to_csv()` (Exportación de DataFrames a Archivos CSV)
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Escribe y guarda el contenido de un DataFrame procesado en un archivo físico delimitado en disco (CSV). Permite controlar parámetros de codificación, separador e inclusión/exclusión del índice de Pandas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.to_csv(path_or_buf, sep=',', index=True, encoding='utf-8')`
  * **Parámetro `index`:** Si se establece en `False`, no escribe las etiquetas numéricas de las filas en el archivo resultante.
  * **Ejemplo de código:**
    ```python
    # Guardar el dataset limpio y procesado excluyendo el índice ordinal
    expvida.to_csv('expvida_procesado.csv', index=False)
    ```
  * **Output esperado / Resultado:**
    ```text
    [Archivo Generado]: Se crea el archivo 'expvida_procesado.csv' en disco
    con todas las variables limpias y procesadas listo para su utilización.
    ```

---

## Tabla Resumen Integral: Mapeo de Funciones por Etapa del Pipeline

| Etapa del Pipeline | Librería | Función / Estimador / Atributo | Propósito Principal | Output Representativo |
| :--- | :--- | :--- | :--- | :--- |
| **1. Ingestión y Carga** | Pandas | `pd.read_csv()` | Carga de archivos CSV o URLs | `DataFrame` tabular |
| | Pandas | `pd.DataFrame()` | Constructor manual de DataFrames | Objeto `DataFrame` |
| **2. Exploración Estructural** | Pandas | `df.head()` / `df.tail()` | Muestra inicial o final de registros | Primeras / últimas $n$ filas |
| | Pandas | `df.shape` | Dimensiones totales del dataset | Tupla `(filas, columnas)` |
| | Pandas | `df.columns` | Lista de nombres de columnas | Objeto `Index(['col1', ...])` |
| | Pandas | `df.dtypes` / `Series.dtype` | Tipos de datos por columna | Serie con dtypes |
| | Python / Pandas | `len(df)` | Cantidad de filas totales | Entero $N$ |
| | Pandas | `df.info()` | Diagnóstico conciso: nulos, dtypes y memoria | Resumen completo en consola |
| **3. Filtrado y Selección** | Pandas | `df.iloc[]` | Selección por posición entera ordinal | Subconjunto por índices |
| | Pandas | `df.loc[]` | Selección por etiquetas o condiciones | Subconjunto por nombres |
| | Pandas | `df[condicion]` | Filtrado booleano condicional | Subconjunto filtrado |
| | Pandas | `Series.idxmax()` / `idxmin()` | Índice del valor máximo o mínimo | Etiqueta de índice |
| | Pandas | `Series.isin()` | Filtrado por pertenencia a listas/conjuntos | Subconjunto de categorías |
| **4. Limpieza y Diagnóstico** | Pandas | `df.isnull()` / `df.isna()` | Detección de valores faltantes (`NaN`) | Máscara booleana |
| | Pandas | `df.isnull().sum()` | Recuento de nulos por columna | Serie con conteo de faltantes |
| | Pandas | `df.dropna()` | Eliminación de filas/columnas con nulos | DataFrame sin faltantes |
| | Pandas | `df.fillna()` | Imputación directa con Pandas | DataFrame con nulos rellenados |
| | Pandas | `df.duplicated()` / `drop_duplicates()` | Detección y remoción de duplicados | DataFrame sin filas repetidas |
| | Pandas | `pd.to_numeric()` | Conversión forzada y segura a números | Serie numérica |
| | SciPy | `stats.zscore()` | Cálculo del puntaje Z para outliers | Array de Z-scores |
| | Pandas | `Series.skew()` | Coeficiente de asimetría de distribución | Valor flotante de asimetría |
| | SciPy | `stats.probplot()` | Gráficos Q-Q Plot de normalidad | Gráfica Q-Q sobre diagonal |
| **5. Transformación e Ing. Funciones** | Pandas | `df.rename()` | Renombra columnas o índices | Columnas actualizadas |
| | Pandas | `df.drop()` | Remueve columnas o filas | DataFrame reducido |
| | Pandas | `df.reset_index()` | Reinicia el índice a enteros limpios | Índice entero `0..N-1` |
| | Pandas | `pd.melt()` | Despivota de formato ancho a largo | DataFrame en formato largo |
| | Pandas | `pd.concat()` | Une DataFrames vertical u horizontalmente | DataFrame unificado |
| | Pandas | `pd.get_dummies()` | One-Hot Encoding en Pandas | Columnas binarias indicadoras |
| | Pandas | `pd.cut()` | Discretización / Binning continuo | Serie categórica segmentada |
| | Pandas | `df.sort_values()` | Ordenamiento por valores de columna | DataFrame ordenado |
| | Pandas | `df.replace()` | Sustitución de valores imposibles o centinelas | Datos saneados |
| | Pandas | `Series.str` (`strip`, `lower`, `replace`) | Saneamiento de textos y columnas | Nombres normalizados |
| | Pandas | `Series.astype()` | Conversión explícita de tipos (ej. booleanos a binarios) | Serie de tipo entero o deseado |
| **6. Agregación y Estadística** | Pandas | `df.describe()` | Resumen estadístico descriptivo completo | Tabla estadística univariada |
| | Pandas | `.mean()`, `.median()`, `.min()`, `.max()` | Estadísticos descriptivos puntuales | Valores escalares |
| | Pandas | `Series.unique()` | Arreglo de categorías únicas sin duplicar | Array NumPy de categorías |
| | Pandas | `Series.value_counts()` | Frecuencias de clases (balance del target) | Serie de frecuencias |
| | Pandas | `df.groupby()` | Agrupación Split-Apply-Combine | Datos agregados por categoría |
| | Pandas | `pd.crosstab()` | Tablas de contingencia cruzada | Frecuencias bivariadas |
| | Pandas | `df.corr()` | Matriz simétrica de correlación lineal | Matriz de correlación |
| | Pandas | `df.corrwith()` | Correlación de variables con el target ($y$) | Serie ordenada por correlación |
| **7. Visualización (Seaborn)** | Seaborn | `sns.displot()` | Distribución univariada con KDE | Gráfico continuo de distribución |
| | Seaborn | `sns.histplot()` | Histograma univariado con bins y hue | Histograma moderno |
| | Seaborn | `sns.countplot()` | Gráfico de barras de conteo categórico | Barras de frecuencias |
| | Seaborn | `sns.barplot()` | Gráfico de medias categóricas con IC | Barras con intervalos de confianza |
| | Seaborn | `sns.boxplot()` | Diagrama de cajas y detección de outliers | Cajas intercuartílicas |
| | Seaborn | `sns.scatterplot()` | Diagrama de dispersión bivariado | Nube de puntos bivariada |
| | Seaborn | `sns.pairplot()` | Matriz de dispersión multivariada pareada | Grilla de gráficos $N 	imes N$ |
| | Seaborn | `sns.heatmap()` | Mapa de calor de correlaciones numéricas | Matriz coloreada |
| **8. Estilizado Visual** | Seaborn | `sns.despine()` | Remueve bordes superior y derecho | Gráficos minimalistas limpios |
| **9. Preprocesamiento (sklearn)** | Scikit-Learn | `SimpleImputer` | Imputación univariada (media, mediana, moda) | Columnas imputadas sin NaN |
| | Scikit-Learn | `LabelEncoder` | Codificación de etiquetas (SOLO para $y$) | Vector de enteros consecutivos |
| | Scikit-Learn | `OneHotEncoder` | Codificación One-Hot formal para $X$ | Matriz binaria indicadora |
| | Scikit-Learn | `MinMaxScaler` | Escalado a rango acotado $[0, 1]$ | Columna en escala $[0, 1]$ |
| | Scikit-Learn | `StandardScaler` | Estandarización a media 0 y desvío 1 | Columna estandarizada $Z$ |
| | Scikit-Learn | `RobustScaler` | Escalado robusto con mediana y RIC | Columna escalada resistente |
| | Scikit-Learn | `Normalizer` | Normalización por norma unitaria por fila | Matriz con norma vectorial 1 |
| | Scikit-Learn | `PowerTransformer` | Transformación Yeo-Johnson / Box-Cox | Columna normalizada |
| | NumPy | `np.log1p()` / `np.expm1()` | Transformación logarítmica e inversa | Columna corregida por sesgo |
| | Scikit-Learn | `Pipeline` | Encadenamiento ordenado de pasos | Flujo reproducible sin Data Leakage |
| | Scikit-Learn | `ColumnTransformer` | Transformación selectiva por tipo de columna | Matriz unificada preprocesada |
| **10. Flujo Train/Test** | Scikit-Learn | `train_test_split()` | Partición en Train y Test | `X_train`, `X_test`, `y_train`, `y_test` |
| | Metodología | Regla de Oro | `fit_transform` en Train, `transform` en Test | Blindaje contra Data Leakage |
| **11. Fundamentos de ML** | Metodología | Componentes de Mitchell | Tarea ($T$), Experiencia ($E$), Rendimiento ($P$), Algoritmo ($A$) | Formulación formal del problema |
| | Metodología | Paradigmas de ML | Supervisado vs No Supervisado | Taxonomía de algoritmos |
| | Metodología | Tareas Supervisadas | Regresión ($y$ continua) vs Clasificación ($y$ discreta) | Selección de la arquitectura |
| | Metodología | Sesgo vs Varianza | Subajuste (*Underfitting*) vs Sobreajuste (*Overfitting*) | Diagnóstico de complejidad |
| **12. Modelos Baseline** | Scikit-Learn | `DummyClassifier` | Clasificador trivial piso (`most_frequent`) | Exactitud piso de referencia |
| | Scikit-Learn | `DummyRegressor` | Regresor trivial piso (`mean`) | MAE/MSE piso de referencia |
| **13. Algoritmos Supervisados** | Scikit-Learn | `LinearRegression` | Regresión lineal múltiple | Coeficientes e intercepto |
| | Scikit-Learn | `LogisticRegression` | Regresión logística probabilística sigmoide | Probabilidades y clases |
| | Scikit-Learn | `DecisionTreeClassifier` / `Regressor` | Árboles de decisión particionales | Reglas interpretables y feature importances |
| | Scikit-Learn | `plot_tree()` | Visualización gráfica interpretable del árbol | Gráfica jerárquica del árbol |
| | Scikit-Learn | `KNeighborsClassifier` | Clasificación geométrica por $K$ vecinos | Votación por distancias |
| | Scikit-Learn | `SVC` | Máquinas de vectores de soporte (linear / rbf) | Hiperplano de margen máximo |
| | Scikit-Learn | `RandomForestClassifier` | Ensamble Bagging de múltiples árboles | Predicción robusta por consenso |
| **14. Métricas de Evaluación** | Scikit-Learn | `mean_absolute_error` (MAE) | Error absoluto medio en unidades de $y$ | Valor escalar de error |
| | Scikit-Learn | `mean_squared_error` / RMSE | Error cuadrático medio con penalización | Valor escalar de error |
| | Scikit-Learn | `r2_score` ($R^2$) | Coeficiente de determinación de varianza | Valor escalar $[-\infty, 1.0]$ |
| | Scikit-Learn | `accuracy_score` | Porcentaje total de aciertos globales | Proporción de aciertos $[0, 1]$ |
| | Scikit-Learn | `confusion_matrix` | Matriz de VP, VN, FP y FN | Matriz numérica de confusión |
| | Scikit-Learn | `ConfusionMatrixDisplay` | Visualización gráfica de la matriz | Mapa de calor con etiquetas |
| | Scikit-Learn | `classification_report` | Precisión, Recall, F1-Score y Soporte | Reporte textual detallado por clase |
| **15. Validación y Diagnóstico** | Scikit-Learn | `cross_val_score` | Validación cruzada $K$-Fold | Vector de métricas por pliegue |
| | Scikit-Learn | `learning_curve` | Diagnóstico de aprendizaje según tamaño $N$ | Curvas de Train vs Validación |
| **16. Optimización de Hiperparámetros** | Optuna | `optuna.create_study()` | Creación de estudio de optimización Bayesiana | Objeto Study de Optuna |
| | Optuna | `trial.suggest_*` | Muestreo inteligente de hiperparámetros | Valores sugeridos por ensayo |
| | Optuna | `study.optimize()` | Búsqueda y convergencia Bayesiana TPE | Historial de ensayos optimizados |
| | Optuna | `study.best_params` | Extracción de la mejor combinación | Diccionario de hiperparámetros óptimos |
| **17. Exportación de Datos** | Pandas | `df.to_csv()` | Guarda y persiste DataFrame en archivo CSV | Archivo `.csv` generado en disco |
