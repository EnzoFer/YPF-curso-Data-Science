# Guía Completa de Pandas, Seaborn, Scikit-Learn y SciPy en Ingeniería de Datos

Este documento presenta un resumen completo, estructurado y ordenado de todas las funciones, métodos, transformadores y conceptos fundamentales de las librerías **Pandas**, **Seaborn**, **Scikit-Learn** (`sklearn`), **SciPy** (`scipy.stats`) y **NumPy** analizados a lo largo de las clases (Notebooks 6, 7, 8, 9, 10, 11, 12 y 12 C8).

El contenido se encuentra organizado según el ciclo de vida o pipeline de la **Ingeniería y Análisis de Datos**:

1. [Etapa 1: Ingestión y Carga de Datos](#etapa-1-ingestión-y-carga-de-datos)
   * [1.1 `pd.read_csv()`](#11-pdread_csv)
   * [1.2 `pd.DataFrame()`](#12-pddataframe)
2. [Etapa 2: Exploración e Inspección Estructural](#etapa-2-exploración-e-inspección-estructural)
   * [2.1 `df.head()` y `df.tail()`](#21-dfhead-y-dftail)
   * [2.2 `df.shape`](#22-dfshape)
   * [2.3 `df.columns`](#23-dfcolumns)
   * [2.4 `df.dtypes` y `Series.dtype`](#24-dfdtypes-y-seriesdtype)
   * [2.5 `len(df)`](#25-lendf)
3. [Etapa 3: Filtrado, Selección e Indexación](#etapa-3-filtrado-selección-e-indexación)
   * [3.1 `df.iloc[]`](#31-dfiloc)
   * [3.2 `df.loc[]`](#32-dfloc)
   * [3.3 Filtrado Booleano / Indexación Condicional](#33-filtrado-booleano--indexación-condicional)
   * [3.4 `Series.idxmax()` y `Series.idxmin()`](#34-seriesidxmax-y-seriesidxmin)
4. [Etapa 4: Limpieza, Diagnóstico de Distribución, Valores Faltantes y Outliers](#etapa-4-limpieza-diagnóstico-de-distribución-valores-faltantes-y-outliers)
   * [4.1 `df.isnull()` y `df.isna()`](#41-dfisnull-y-dfisna)
   * [4.2 `df.isnull().sum()` y Conteo de Nulos](#42-dfisnullsum-y-conteo-de-nulos)
   * [4.3 `df.dropna()`](#43-dfdropna)
   * [4.4 `pd.to_numeric()`](#44-pdto_numeric)
   * [4.5 `stats.zscore()` (Detección de Outliers por Puntaje Z con SciPy)](#45-statszscore-detección-de-outliers-por-puntaje-z-con-scipy)
   * [4.6 `Series.skew()` (Evaluación del Coeficiente de Asimetría)](#46-seriesskew-evaluación-del-coeficiente-de-asimetría)
   * [4.7 `stats.probplot()` (Gráficos Q-Q Plot con SciPy)](#47-statsprobplot-gráficos-q-q-plot-con-scipy)
5. [Etapa 5: Transformación, Reestructuración e Ingeniería de Funciones](#etapa-5-transformación-reestructuración-e-ingeniería-de-funciones)
   * [5.1 `df.rename()`](#51-dfrename)
   * [5.2 `df.drop()`](#52-dfdrop)
   * [5.3 `df.reset_index()`](#53-dfreset_index)
   * [5.4 `pd.melt()`](#54-pdmelt)
   * [5.5 `pd.concat()`](#55-pdconcat)
   * [5.6 `pd.get_dummies()` (One-Hot Encoding en Pandas)](#56-pdget_dummies-one-hot-encoding-en-pandas)
   * [5.7 `pd.cut()` (Discretización / Binning)](#57-pdcut-discretización--binning)
   * [5.8 `df.sort_values()`](#58-dfsort_values)
6. [Etapa 6: Agregación, Estadísticas Descriptivas y Análisis Multivariado](#etapa-6-agregación-estadísticas-descriptivas-y-análisis-multivariado)
   * [6.1 `df.describe()` y `Series.describe()`](#61-dfdescribe-y-seriesdescribe)
   * [6.2 Métodos Estadísticos de Agregación Simples](#62-métodos-estadísticos-de-agregación-simples)
   * [6.3 `Series.unique()`](#63-seriesunique)
   * [6.4 `Series.value_counts()`](#64-seriesvalue_counts)
   * [6.5 `df.groupby()`](#65-dfgroupby)
   * [6.6 `pd.crosstab()`](#66-pdcrosstab)
   * [6.7 `df.corr()`](#67-dfcorr)
7. [Etapa 7: Visualización Exploratoria de Datos (Seaborn)](#etapa-7-visualización-exploratoria-de-datos-seaborn)
   * [7.1 `sns.displot()`](#71-snsdisplot)
   * [7.2 `sns.countplot()`](#72-snscountplot)
   * [7.3 `sns.barplot()`](#73-snsbarplot)
   * [7.4 `sns.boxplot()`](#74-snsboxplot)
   * [7.5 `sns.scatterplot()`](#75-snsscatterplot)
   * [7.6 `sns.pairplot()`](#76-snspairplot)
   * [7.7 `sns.heatmap()`](#77-snsheatmap)
8. [Etapa 8: Estilizado y Personalización Visual de Gráficos](#etapa-8-estilizado-y-personalización-visual-de-gráficos)
   * [8.1 `sns.despine()`](#81-snsdespine)
9. [Etapa 9: Preprocesamiento, Escalado y Transformación con Scikit-Learn (`sklearn`)](#etapa-9-preprocesamiento-escalado-y-transformación-con-scikit-learn-sklearn)
   * [9.1 `SimpleImputer` (Imputación Automática de Valores Faltantes)](#91-simpleimputer-imputación-automática-de-valores-faltantes)
   * [9.2 `LabelEncoder` (Codificación Ordinal / de Etiquetas)](#92-labelencoder-codificación-ordinal--de-etiquetas)
   * [9.3 `OneHotEncoder` (Codificación Categórica Nominal)](#93-onehotencoder-codificación-categórica-nominal)
   * [9.4 `MinMaxScaler` (Re-escalado de Características a un Rango)](#94-minmaxscaler-re-escalado-de-características-a-un-rango)
   * [9.5 `StandardScaler` (Estandarización / Escala Z)](#95-standardscaler-estandarización--escala-z)
   * [9.6 `RobustScaler` (Escalado Robusto Resistente a Outliers)](#96-robustscaler-escalado-robusto-resistente-a-outliers)
   * [9.7 `Normalizer` (Normalización por Normas Vectoriales de Muestras)](#97-normalizer-normalización-por-normas-vectoriales-de-muestras)
   * [9.8 `PowerTransformer` (Transformación de Potencia Yeo-Johnson y Box-Cox)](#98-powertransformer-transformación-de-potencia-yeo-johnson-y-box-cox)
   * [9.9 `np.log1p()` y `np.expm1()` (Transformación Logarítmica de Forma)](#99-nplog1p-y-npexpm1-transformación-logarítmica-de-forma)
10. [Etapa 10: Flujo Correcto de Entrenamiento y Prevención de Data Leakage](#etapa-10-flujo-correcto-de-entrenamiento-y-prevención-de-data-leakage)
   * [10.1 `train_test_split()` (División del Dataset en Train y Test)](#101-train_test_split-división-del-dataset-en-train-y-test)
   * [10.2 Regla de Oro: Separación de `fit_transform` en Train vs `transform` en Test](#102-regla-de-oro-separación-de-fit_transform-en-train-vs-transform-en-test)
11. [Etapa 11: Exportación y Almacenamiento de Datos](#etapa-11-exportación-y-almacenamiento-de-datos)
   * [11.1 `df.to_csv()` (Exportación de DataFrames a Archivos CSV)](#111-dfto_csv-exportación-de-dataframes-a-archivos-csv)
12. [Tabla Resumen: Mapeo de Funciones por Etapa del Pipeline](#tabla-resumen-mapeo-de-funciones-por-etapa-del-pipeline)

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
    print(blackfriday['Age'].dtype)
    ```
  * **Output esperado / Resultado:**
    ```text
    object
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

## Etapa 3: Filtrado, Selección e Indexación

En esta fase se recortan, seleccionan o filtran subconjuntos de datos según posiciones enteras (`iloc`), etiquetas de fila/columna (`loc`) o condiciones lógicas booleanas.

### 3.1 `df.iloc[]`
* **Librería:** Pandas (Indexer de `DataFrame`)
* **¿Qué hace?:** Permite la selección e indexación puramente basada en la **posición entera** (0 a N-1) de filas y columnas, independientemente de los nombres de etiquetas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.iloc[filas, columnas]`
  * **Ejemplo de código:**
    ```python
    # Seleccionar las primeras 2 columnas de las primeras 3 filas
    df.iloc[0:3, 0:2]
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
  * **Sintaxis:** `df.loc[etiqueta_filas, etiqueta_columnas]`
  * **Ejemplo de código:**
    ```python
    # Seleccionar filas con índices 1 y 3 para columnas específicas
    df.loc[[1, 3], ['gender', 'lunch']]
    ```
  * **Output esperado / Resultado:**
    ```text
       gender     lunch
    1  female  standard
    3    male  free/reduced
    ```

---

### 3.3 Filtrado Booleano / Indexación Condicional
* **Librería:** Pandas (Sintaxis `df[condicion]`)
* **¿Qué hace?:** Evalúa expresiones lógicas que devuelven Series booleanas (`True`/`False`) para filtrar y quedarse únicamente con los registros que satisfacen la condición.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df[condicion_1 & condicion_2]` (Usa `&` para AND, `|` para OR, `~` para NOT).
  * **Ejemplo de código:**
    ```python
    # Filtrar estudiantes con puntaje de matemática strictly mayor a 70
    df_math = df[df['math score'] > 70]
    df_math[['gender', 'math score']].head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
       gender  math score
    0  female          72
    2  female          90
    4    male          76
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
    idx_max = expvida['life_expectancy'].idxmax()
    print("Índice máximo:", idx_max)
    print(expvida.loc[idx_max, ['Country', 'Year', 'life_expectancy']])
    ```
  * **Output esperado / Resultado:**
    ```text
    Índice máximo: 241
    Country            Belgium
    Year                  2014
    life_expectancy       89.0
    Name: 241, dtype: object
    ```

---

## Etapa 4: Limpieza, Diagnóstico de Distribución, Valores Faltantes y Outliers

La calidad de datos asegura que la información esté libre de incoherencias, faltantes (`NaN`), anomalías u *outliers*, e identifica el grado de asimetría de las distribuciones numéricas.

### 4.1 `df.isnull()` y `df.isna()`
* **Librería:** Pandas (Métodos de `DataFrame` y `Series`)
* **¿Qué hace?:** Detectan valores faltantes (`NaN`, `None`). Retornan un DataFrame/Serie del mismo tamaño compuesto por valores booleanos (`True` donde hay valor nulo, `False` donde hay valor válido).
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.isnull()` / `df.isna()`
  * **Ejemplo de código:**
    ```python
    # Detectar si hay valores faltantes por fila en 'Gender'
    blackfriday['Gender'].isna().head(4)
    ```
  * **Output esperado / Resultado:**
    ```text
    0    False
    1    False
    2    False
    3     True
    Name: Gender, dtype: bool
    ```

---

### 4.2 `df.isnull().sum()` y Conteo de Nulos
* **Librería:** Pandas (Combinación de `isnull()` / `isna()` con `sum()`)
* **¿Qué hace?:** Contabiliza la cantidad total de valores nulos por columna.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.isnull().sum()`
  * **Ejemplo de código:**
    ```python
    datos_faltantes = expvida.isnull().sum()
    datos_faltantes[datos_faltantes > 0].head(4)
    ```
  * **Output esperado / Resultado:**
    ```text
    life_expectancy     10
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
    # Eliminar filas con nulos en 'Gender'
    print("Dimensiones antes:", blackfriday.shape)
    blackfriday.dropna(subset=['Gender'], inplace=True)
    print("Dimensiones después:", blackfriday.shape)
    ```
  * **Output esperado / Resultado:**
    ```text
    Dimensiones antes: (537577, 12)
    Dimensiones después: (537540, 12)
    ```

---

### 4.4 `pd.to_numeric()`
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Convierte una Serie a un tipo de dato numérico (`int` o `float`), transformando caracteres inválidos en `NaN` cuando se usa `errors='coerce'`.
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.to_numeric(arg, errors='raise', downcast=None)`
  * **Ejemplo de código:**
    ```python
    blackfriday["Age"] = pd.to_numeric(blackfriday["Age"], errors='coerce')
    print(blackfriday["Age"].dtype)
    ```
  * **Output esperado / Resultado:**
    ```text
    float64
    ```

---

### 4.5 `stats.zscore()` (Detección de Outliers por Puntaje Z con SciPy)
* **Librería:** SciPy (`scipy.stats.zscore`)
* **¿Qué hace?:** Calcula el puntaje Z (*Z-score*) para cada observación en una columna cuantitativa. Mide a cuántas desviaciones estándar de la media se encuentra cada valor ($Z = \frac{x - \mu}{\sigma}$). Permite identificar y filtrar *outliers* estableciendo un umbral (típicamente $|Z| > 2.0$ o $|Z| > 3.0$).
* **¿Cómo usarla?:**
  * **Sintaxis:** `z = stats.zscore(array_o_columna)`
  * **Ejemplo de código:**
    ```python
    from scipy import stats
    import numpy as np

    # Calcular Z-score sobre la columna 'Purchase'
    z = stats.zscore(np.array(blackfriday['Purchase']))
    threshold = 2.0  # Filtrar valores situados a más de 2 desvíos estándar de la media

    # Seleccionar índices de registros no atípicos
    z_index = blackfriday['Purchase'][np.abs(z) < threshold].index
    blackfriday_withzscore = blackfriday.loc[z_index]

    print("Registros originales:", len(blackfriday))
    print("Registros sin outliers Z-Score:", len(blackfriday_withzscore))
    ```
  * **Output esperado / Resultado:**
    ```text
    Registros originales: 537577
    Registros sin outliers Z-Score: 512400
    ```

---

### 4.6 `Series.skew()` (Evaluación del Coeficiente de Asimetría)
* **Librería:** Pandas (Método de `Series` / `DataFrame`)
* **¿Qué hace?:** Computa el coeficiente de asimetría (*skewness*) de distribuciones cuantitativas continuas. 
  * Un valor cercano a `0` indica una distribución simétrica.
  * Un valor mayor a `0.5` o `1.0` indica **sesgo positivo (cola larga a la derecha)**, requiriendo transformaciones como logaritmo o Yeo-Johnson.
  * Un valor menor a `-0.5` indica **sesgo negativo (cola larga a la izquierda)**.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['columna'].skew()` / `df.select_dtypes(include=np.number).skew()`
  * **Ejemplo de código:**
    ```python
    # Ordenar todas las columnas numéricas de mayor a menor asimetría
    asimetrias = vida.select_dtypes(include=np.number).skew().sort_values(ascending=False)
    print(asimetrias.head(4))
    ```
  * **Output esperado / Resultado:**
    ```text
    Measles        9.441324
    Population    15.955473
    GDP            3.212040
    under-five     6.852109
    dtype: float64
    ```

---

### 4.7 `stats.probplot()` (Gráficos Q-Q Plot con SciPy)
* **Librería:** SciPy (`scipy.stats.probplot`)
* **¿Qué hace?:** Genera un gráfico de probabilidad o **Q-Q Plot** (*Quantile-Quantile Plot*) comparando visualmente los cuantiles empíricos de los datos observados contra los cuantiles teóricos de una distribución normal Gaussiana. Si los puntos se alinean sobre la recta diagonal a 45°, los datos siguen una distribución normal; si se curva en los extremos, evidencia sesgo o colas pesadas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `stats.probplot(series_o_array, dist="norm", plot=plt)`
  * **Ejemplo de código:**
    ```python
    import matplotlib.pyplot as plt
    from scipy import stats

    gdp = vida['GDP'].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    stats.probplot(gdp, dist="norm", plot=ax)
    plt.title('Q-Q Plot: GDP Original')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Un plano cartesiano con una línea recta diagonal roja de referencia teórica
    y puntos azules representando los valores del GDP. La gráfica muestra una marcada curvatura despegada
    de la diagonal en la parte superior derecha, evidenciando una fuerte asimetría positiva.
    ```

---

## Etapa 5: Transformación, Reestructuración e Ingeniería de Funciones

En esta fase se renombran variables, eliminan columnas irrelevantes, aplican técnicas de estructuración de columnas, segmentación (*Binning*) y despivote/concatenación.

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
* **¿Qué hace?:** Remueve las filas o columnas especificadas según sus etiquetas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.drop(labels, axis=0, inplace=False)`
  * **Ejemplo de código:**
    ```python
    blackfriday.drop(['Product_Category_2', 'Product_Category_3'], axis=1, inplace=True)
    print('Product_Category_2' in blackfriday.columns)
    ```
  * **Output esperado / Resultado:**
    ```text
    False
    ```

---

### 5.3 `df.reset_index()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Reinicia el índice del DataFrame devolviéndolo a una secuencia numérica entera limpia `0, 1, 2, ..., N-1`.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.reset_index(drop=False, inplace=False)`
  * **Ejemplo de código:**
    ```python
    bf.reset_index(drop=True, inplace=True)
    print(bf.index)
    ```
  * **Output esperado / Resultado:**
    ```text
    RangeIndex(start=0, stop=500000, step=1)
    ```

---

### 5.4 `pd.melt()`
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Transforma un DataFrame de formato ancho (*wide format*) a formato largo (*long format* / *tidy data*).
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.melt(df, id_vars=None, value_vars=None, var_name=None, value_name='value')`
  * **Ejemplo de código:**
    ```python
    boxplot_blackfriday = pd.melt(blackfriday, id_vars='City_Category', value_vars=['Purchase'])
    boxplot_blackfriday.head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
      City_Category  variable  value
    0             A  Purchase   8370
    1             C  Purchase  15200
    2             A  Purchase   1422
    ```

---

### 5.5 `pd.concat()`
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Concatena u une objetos de Pandas (DataFrames o Series) horizontalmente (por columnas) o verticalmente (por filas).
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.concat(objs, axis=0, ignore_index=False)`
  * **Ejemplo de código:**
    ```python
    new_df = pd.concat([blackfriday, one_hot_gender], axis=1)
    new_df[['Gender', 'F', 'M']].head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
      Gender  F  M
    0      F  1  0
    1      M  0  1
    2      M  0  1
    ```

---

### 5.6 `pd.get_dummies()` (One-Hot Encoding en Pandas)
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Realiza codificación categórica mediante *One-Hot Encoding*, convirtiendo variables categóricas en columnas binarias (1 y 0).
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.get_dummies(data, prefix=None, drop_first=False, dtype=None)`
  * **Ejemplo de código:**
    ```python
    pd.get_dummies(blackfriday["Gender"]).head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
       F  M
    0  1  0
    1  0  1
    2  0  1
    ```

---

### 5.7 `pd.cut()` (Discretización / Binning)
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Discretiza o segmenta los valores de una columna continua en rangos o contenedores (*bins*) discretos.
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.cut(x, bins, labels=None, right=True)`
  * **Ejemplo de código:**
    ```python
    bin_age = [10, 17, 70, 80]
    labels = ["Adolescente", "Adulto", "Anciano"]
    age_categories = pd.cut(blackfriday["Age"], bins=bin_age, labels=labels)
    age_categories.head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
    0    Adulto
    1    Adulto
    2    Adulto
    Name: Age, dtype: category
    Categories (3, object): ['Adolescente' < 'Adulto' < 'Anciano']
    ```

---

### 5.8 `df.sort_values()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Ordena las filas del DataFrame según los valores contenidos en una o varias columnas especificadas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.sort_values(by, ascending=True, inplace=False)`
  * **Ejemplo de código:**
    ```python
    blackfriday.sort_values(by='Occupation')[['Occupation', 'Purchase']].head(3)
    ```
  * **Output esperado / Resultado:**
    ```text
            Occupation  Purchase
    10234            0      5200
    4031             0     11400
    88912            0      8910
    ```

---

## Etapa 6: Agregación, Estadísticas Descriptivas y Análisis Multivariado

En esta fase se realizan resúmenes cuantitativos, agrupamientos por categorías, tablas cruzadas de frecuencia y matrices de correlación.

### 6.1 `df.describe()` y `Series.describe()`
* **Librería:** Pandas (Método de `DataFrame` y `Series`)
* **¿Qué hace?:** Genera un resumen completo de estadísticas descriptivas (conteo, media, std, mín, percentiles 25%, 50%, 75%, máx).
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.describe()` / `df['columna'].describe()`
  * **Ejemplo de código:**
    ```python
    blackfriday['Purchase'].describe()
    ```
  * **Output esperado / Resultado:**
    ```text
    count    537577.000000
    mean       9263.968713
    std        5023.065394
    min         185.000000
    25%        5823.000000
    50%        8047.000000
    75%       12054.000000
    max       23961.000000
    Name: Purchase, dtype: float64
    ```

---

### 6.2 Métodos Estadísticos de Agregación Simples
* **Librería:** Pandas (Métodos de `Series` / `DataFrame`)
* **¿Qué hace?:** Computan medidas estadísticas individuales (`.mean()`, `.min()`, `.max()`, `.quantile()`).
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['col'].mean()`, `df['col'].quantile(0.25)`
  * **Ejemplo de código:**
    ```python
    q1 = blackfriday['Purchase'].quantile(0.25)
    q3 = blackfriday['Purchase'].quantile(0.75)
    iqr = q3 - q1
    print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Q1: 5823.0, Q3: 12054.0, IQR: 6231.0
    ```

---

### 6.3 `Series.unique()`
* **Librería:** Pandas (Método de `Series`)
* **¿Qué hace?:** Devuelve una matriz de NumPy con los valores únicos (sin duplicados) de una columna.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['columna'].unique()`
  * **Ejemplo de código:**
    ```python
    expvida['Status'].unique()
    ```
  * **Output esperado / Resultado:**
    ```text
    array(['Developing', 'Developed'], dtype=object)
    ```

---

### 6.4 `Series.value_counts()`
* **Librería:** Pandas (Método de `Series`)
* **¿Qué hace?:** Cuenta la frecuencia absoluta de cada valor único dentro de una columna.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['columna'].value_counts()`
  * **Ejemplo de código:**
    ```python
    expvida['Status'].value_counts()
    ```
  * **Output esperado / Resultado:**
    ```text
    Developing    2426
    Developed      512
    Name: Status, dtype: int64
    ```

---

### 6.5 `df.groupby()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Divide el dataset en subgrupos basados en categorías, aplica agregaciones y combina los resultados.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.groupby(by=['col_grupo'])['col_objetivo'].mean()`
  * **Ejemplo de código:**
    ```python
    blackfriday.groupby('City_Category')['Purchase'].mean()
    ```
  * **Output esperado / Resultado:**
    ```text
    City_Category
    A    8993.364426
    B    9150.362145
    C    9498.423976
    Name: Purchase, dtype: float64
    ```

---

### 6.6 `pd.crosstab()`
* **Librería:** Pandas (`pd`)
* **¿Qué hace?:** Calcula una tabla de contingencia de frecuencias para dos o más factores categóricos.
* **¿Cómo usarla?:**
  * **Sintaxis:** `pd.crosstab(index, columns)`
  * **Ejemplo de código:**
    ```python
    pd.crosstab(blackfriday["City_Category"], blackfriday["Gender"])
    ```
  * **Output esperado / Resultado:**
    ```text
    Gender              F       M
    City_Category                
    A               36113  112028
    B               57297  174182
    C               42409  115548
    ```

---

### 6.7 `df.corr()`
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Calcula la matriz de correlación lineal entre todas las columnas numéricas del DataFrame.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.corr(method='pearson')`
  * **Ejemplo de código:**
    ```python
    corr = blackfriday[['Age', 'Stay_In_Current_City_Years', 'Purchase']].corr()
    print(corr)
    ```
  * **Output esperado / Resultado:**
    ```text
                                  Age  Stay_In_Current_City_Years  Purchase
    Age                      1.000000                    0.231002  0.054320
    Stay_In_Current_City_Y   0.231002                    1.000000  0.005517
    Purchase                 0.054320                    0.005517  1.000000
    ```

---

## Etapa 7: Visualización Exploratoria de Datos (Seaborn)

En esta etapa se utilizan las funciones de la librería **Seaborn** (`sns`) para realizar análisis exploratorio gráfico (EDA) mediante gráficos de distribución, gráficos categóricos, diagramas de dispersión y mapas de calor.

### 7.1 `sns.displot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Dibuja una distribución univariada de una variable continua combinando un histograma y opcionalmente una curva de estimación de densidad de kernel (`kde=True`).
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.displot(data=None, x=None, color=None, kde=False)`
  * **Ejemplo de código:**
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.displot(blackfriday["Purchase"], color="#5ea88e", kde=True)
    plt.xlabel('Monto de la compra')
    plt.ylabel('Frecuencia')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Un gráfico de histograma con barras verticales verde menta (#5ea88e)
    que representan el volumen de compras según el monto en el eje X (de 0 a 25,000), sobrepuesto 
    con una curva suave continua de densidad (KDE) que muestra picos alrededor de 8,000 y 15,000.
    ```

---

### 7.2 `sns.countplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Grafica las frecuencias observadas de una variable categórica utilizando barras verticales u horizontales.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.countplot(x=None, y=None, data=None, palette=None)`
  * **Ejemplo de código:**
    ```python
    plt.figure(figsize=(10, 5))
    sns.countplot(x="Gender", data=blackfriday, palette="Set3")
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Un gráfico de barras categórico con 2 barras principales en el eje X:
    - Barra 'M' (Male): Altura ~400,000 eventos (color pastel verde/azul).
    - Barra 'F' (Female): Altura ~135,000 eventos (color pastel amarillo/naranja).
    ```

---

### 7.3 `sns.barplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Muestra estimaciones puntuales de una variable numérica (por defecto la media) desglosadas por categorías, agregando automáticamente barras de error.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.barplot(x=None, y=None, data=None, color=None, palette=None)`
  * **Ejemplo de código:**
    ```python
    sns.barplot(x='Gender', y='Purchase', data=blackfriday, color='#FF0000')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Dos barras rojas que comparan el promedio gastado:
    - 'F': Altura cercana a 8,700 en el eje Y.
    - 'M': Altura cercana a 9,400 en el eje Y.
    Ambas incluyen una pequeña línea negra vertical en el tope representando el intervalo de confianza.
    ```

---

### 7.4 `sns.boxplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Muestra la distribución de variables cuantitativas a través de sus cuartiles (Q1, Mediana, Q3) e identifica visualmente los valores atípicos (*outliers*).
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.boxplot(x=None, y=None, data=None, hue=None, palette=None)`
  * **Ejemplo de código:**
    ```python
    sns.boxplot(x="variable", y="value", data=boxplot_blackfriday, palette="Set2", hue='City_Category')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Tres cajas rectangulares de colores (Set2) agrupadas horizontalmente:
    - Caja A, B y C mostrando la línea central de la mediana (~8,000-9,000), los bordes inferior/superior
      del rango intercuartílico (Q1-Q3) y puntos individuales por encima del bigote superior (Outliers > 21,000).
    ```

---

### 7.5 `sns.scatterplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Representa la relación entre dos variables numéricas continuas mediante puntos dibujados en un plano cartesiano.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.scatterplot(x=None, y=None, data=None, palette=None)`
  * **Ejemplo de código:**
    ```python
    sns.scatterplot(x="Age", y="Purchase", data=blackfriday, palette="spring")
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Una nube dispersa de puntos en el plano cartesiano donde el eje X muestra
    las categorías de edad y el eje Y muestra la distribución continua del monto de compra (0 a 25,000).
    ```

---

### 7.6 `sns.pairplot()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Construye una matriz de gráficos de dispersión para evaluar simultáneamente todas las parejas posibles de variables cuantitativas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.pairplot(data, hue=None, palette=None)`
  * **Ejemplo de código:**
    ```python
    sns.pairplot(blackfriday[['Stay_In_Current_City_Years', 'Age', 'Purchase', 'Gender']], 
                 hue='Gender', 
                 palette='PuRd')
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Una grilla de 3x3 gráficos:
    - En la diagonal: 3 histogramas/KDEs individuales por variable coloreados en tonos violetas (PuRd) por Género.
    - Fuera de la diagonal: 6 diagramas de dispersión bivariados comparando los pares de variables.
    ```

---

### 7.7 `sns.heatmap()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Representa una matriz de datos bidimensional (típicamente la matriz de correlación de Pandas) mediante celdas de colores codificados con anotaciones numéricas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.heatmap(data, cmap=None, annot=False)`
  * **Ejemplo de código:**
    ```python
    corr = blackfriday[['Age', 'Stay_In_Current_City_Years', 'Purchase']].corr()
    sns.heatmap(corr, cmap='YlGnBu', annot=True)
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Gráfica Renderizada]: Una matriz cuadrada de 3x3 de celdas coloreadas según la escala YlGnBu (Amarillo-Verde-Azul):
    - Celdas diagonales con valor 1.0 (Azul oscuro).
    - Celdas cruzadas con valores anotados impresos (ej. 0.23, 0.05, 0.01) y gradiente de color proporcional.
    ```

---

## Etapa 8: Estilizado y Personalización Visual de Gráficos

Esta última etapa aborda el refinamiento estético de las figuras para informes y presentaciones.

### 8.1 `sns.despine()`
* **Librería:** Seaborn (`sns`)
* **¿Qué hace?:** Remueve los ejes superiores y derechos ("espinas") de los gráficos creados con Seaborn o Matplotlib, logrando un diseño visual moderno y limpio.
* **¿Cómo usarla?:**
  * **Sintaxis:** `sns.despine(top=True, right=True, left=False, bottom=False)`
  * **Ejemplo de código:**
    ```python
    sns.displot(blackfriday["Purchase"], kde=True)
    sns.despine()
    plt.show()
    ```
  * **Output esperado / Resultado:**
    ```text
    [Efecto Visual Visualizado]: El marco rectangular alrededor de la gráfica pierde el borde superior 
    y el borde derecho, dejando únicamente la línea de ejes X (inferior) y Y (izquierdo).
    ```

---

## Etapa 9: Preprocesamiento, Escalado y Transformación con Scikit-Learn (`sklearn`)

En esta etapa dedicada se agrupan todos los estimadores, transformadores, escaladores y transformadores de potencia de **Scikit-Learn** utilizados en las clases (Notebooks 9, 11, 12 y 12 C8) para la preparación formal de datos previa al modelado.

### 9.1 `SimpleImputer` (Imputación Automática de Valores Faltantes)
* **Librería:** Scikit-Learn (`sklearn.impute.SimpleImputer`)
* **¿Qué hace?:** Reemplaza automáticamente los valores faltantes (`NaN`) en datasets cuantitativos o cualitativos utilizando estrategias de cálculo como la media (`"mean"`), mediana (`"median"`), la moda o valor más frecuente (`"most_frequent"`) o una constante.
* **¿Cómo usarla?:**
  * **Sintaxis:** `imputer = SimpleImputer(missing_values=np.nan, strategy='mean')`
  * **Ejemplo de código:**
    ```python
    from sklearn.impute import SimpleImputer
    import numpy as np

    # Imputación de variable categórica/discreta usando la moda (most_frequent)
    imputer_occ = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    blackfriday["Occupation"] = imputer_occ.fit_transform(blackfriday[['Occupation']]).ravel()

    # Imputación de variable numérica usando la media
    imputer_age = SimpleImputer(missing_values=np.nan, strategy="mean")
    blackfriday_age_imp = imputer_age.fit_transform(blackfriday[['Age']])
    print("Nulos restantes en Occupation:", blackfriday['Occupation'].isna().sum())
    ```
  * **Output esperado / Resultado:**
    ```text
    Nulos restantes en Occupation: 0
    ```

---

### 9.2 `LabelEncoder` (Codificación Ordinal / de Etiquetas)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.LabelEncoder`)
* **¿Qué hace?:** Convierte etiquetas categóricas numéricas o de texto en enteros consecutivos (`0, 1, 2, ..., n_classes - 1`). 
  > ⚠️ **Nota metodológica vista en clase:** `LabelEncoder` asigna un orden implícito ($0 < 1 < 2$), por lo que debe utilizarse en variables categóricas ordinales o en el objetivo (*target*), no en predictoras nominales.
* **¿Cómo usarla?:**
  * **Sintaxis:**
    ```python
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df['col_encoded'] = encoder.fit_transform(df['col'])
    ```
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import LabelEncoder

    test_encoder = LabelEncoder()
    # Codificar la categoría de ciudad ('A', 'B', 'C') en enteros (0, 1, 2)
    blackfriday['City_Category_Encoded'] = test_encoder.fit_transform(blackfriday['City_Category'])
    print(blackfriday[['City_Category', 'City_Category_Encoded']].head(4))
    ```
  * **Output esperado / Resultado:**
    ```text
      City_Category  City_Category_Encoded
    0             A                      0
    1             C                      2
    2             A                      0
    3             B                      1
    ```

---

### 9.3 `OneHotEncoder` (Codificación Categórica Nominal)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.OneHotEncoder`)
* **¿Qué hace?:** Codificador oficial de Scikit-Learn para convertir variables categóricas nominales en vectores binarios (ceros y unos). Permite guardar el estado de las categorías con `.categories_` para transformar nuevos datos de prueba o producción.
* **¿Cómo usarla?:**
  * **Sintaxis:**
    ```python
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_array = encoder.fit_transform(df[['col_categorica']])
    ```
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd

    # Instanciar el encoder
    gender_encoder = OneHotEncoder()

    # Ajustar y transformar la columna 'Gender' a array denso
    encoded_matrix = gender_encoder.fit_transform(blackfriday[['Gender']]).toarray()

    # Extraer los nombres de las categorías automáticamente
    niveles = gender_encoder.categories_[0].tolist()

    # Convertir el resultado en un DataFrame organizado
    one_hot_gender = pd.DataFrame(encoded_matrix, columns=niveles)
    print(one_hot_gender.head(3))
    ```
  * **Output esperado / Resultado:**
    ```text
         F    M
    0  1.0  0.0
    1  0.0  1.0
    2  0.0  1.0
    ```

---

### 9.4 `MinMaxScaler` (Re-escalado de Características a un Rango)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.MinMaxScaler`)
* **¿Qué hace?:** Re-escala y transforma características numéricas continuas acortándolas a un rango específico dado (por defecto $[0, 1]$, o personalizado como $[0, 100]$). Mantiene intacta la forma de la distribución original eliminando sesgos provocados por diferentes magnitudes o unidades de medida.
  * **Fórmula de transformación:**
    $$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}} \times (max_{rango} - min_{rango}) + min_{rango}$$
* **¿Cómo usarla?:**
  * **Sintaxis:**
    ```python
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['col_scaled'] = scaler.fit_transform(df[['col_continua']])
    ```
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import MinMaxScaler

    # Re-escalar el puntaje de matemática al rango de 0 a 100
    scaler = MinMaxScaler(feature_range=(0, 100), copy=True)
    students['math score'] = scaler.fit_transform(students[['math score']])
    students['math score'].describe()
    ```
  * **Output esperado / Resultado:**
    ```text
    count    1000.000000
    mean       66.089000
    std        15.163080
    min         0.000000
    25%        57.000000
    50%        66.000000
    75%        77.000000
    max       100.000000
    Name: math score, dtype: float64
    ```

---

### 9.5 `StandardScaler` (Estandarización / Escala Z)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.StandardScaler`)
* **¿Qué hace?:** Estandariza variables numéricas centrando la media en $\mu = 0$ y escalando la varianza a una desviación estándar $\sigma = 1$. Es indispensable para algoritmos de aprendizaje automático basados en distancias (como Regresión Lineal/Logística, KNN, SVM, PCA o Redes Neuronales).
  * **Fórmula de transformación:**
    $$Z = \frac{X - \mu}{\sigma}$$
* **¿Cómo usarla?:**
  * **Sintaxis:**
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['col1', 'col2']])
    ```
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import StandardScaler

    # Estandarizar la columna 'reading score'
    scaler = StandardScaler()
    reading_scale = scaler.fit_transform(students[['reading score']])
    print(f"Media transformada: {reading_scale.mean():.2f}, Desvío Estándar: {reading_scale.std():.2f}")
    ```
  * **Output esperado / Resultado:**
    ```text
    Media transformada: 0.00, Desvío Estándar: 1.00
    ```

---

### 9.6 `RobustScaler` (Escalado Robusto Resistente a Outliers)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.RobustScaler`)
* **¿Qué hace?:** Escala variables numéricas apoyándose en la **mediana** ($Q2$) y el **rango intercuartílico** ($\text{RIC} = Q3 - Q1$) en lugar de la media y la desviación estándar. Ésta es la mejor alternativa cuando existen *outliers* extremos que no se desean eliminar del dataset, ya que la mediana y el RIC no son distorsionados por valores atípicos.
  * **Fórmula de transformación:**
    $$x' = \frac{x - \text{mediana}}{\text{RIC}}$$
* **¿Cómo usarla?:**
  * **Sintaxis:**
    ```python
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    df['col_robust'] = scaler.fit_transform(df[['columna']])
    ```
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import RobustScaler

    # Escalado robusto sobre la columna 'math score'
    scaler_rb = RobustScaler()
    students['math_robust'] = scaler_rb.fit_transform(students[['math score']])
    students[['math score', 'math_robust']].describe()
    ```
  * **Output esperado / Resultado:**
    ```text
                 math score  math_robust
    count       1000.000000  1000.000000
    mean          66.089000     0.004450
    50% (mediana) 66.000000     0.000000
    25% (Q1)      57.000000    -0.450000
    75% (Q3)      77.000000     0.550000
    ```

---

### 9.7 `Normalizer` (Normalización por Normas Vectoriales de Muestras)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.Normalizer`)
* **¿Qué hace?:** Escala muestras individuales de manera independiente para que tengan una norma vectorial unitaria (magnitud o longitud igual a 1). 
  > ⚠️ **Advertencia metodológica clave:** `Normalizer` opera **por filas (muestras vectoriales multivariadas)** y no por columnas. Si se aplica a una sola columna individual `[x]`, la norma del vector escalar es $|x|$, devolviendo siempre $1.0$ y destruyendo la información de la variable. Debe aplicarse sobre múltiples columnas donde interese la dirección vectorial de las observaciones (ej. minería de texto o perfiles de clientes).
* **¿Cómo usarla?:**
  * **Sintaxis:**
    ```python
    from sklearn.preprocessing import Normalizer
    normalizer = Normalizer(norm='l2') # Soporta norm='l1', 'l2', 'max'
    normalized_matrix = normalizer.fit_transform(df[['col1', 'col2']])
    ```
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import Normalizer
    import numpy as np

    # Aplicación correcta sobre 2 columnas numéricas simultáneas
    ejemplo = students[['reading score', 'writing score']].head(3)
    normalizado = Normalizer().fit_transform(ejemplo)
    print("Muestras normalizadas (norma 1 por fila):\n", normalizado.round(3))
    print("Normas calculadas:", np.linalg.norm(normalizado, axis=1).round(3))
    ```
  * **Output esperado / Resultado:**
    ```text
    Muestras normalizadas (norma 1 por fila):
     [[0.707 0.707]
      [0.669 0.743]
      [0.697 0.717]]
    Normas calculadas: [1. 1. 1.]
    ```

---

### 9.8 `PowerTransformer` (Transformación de Potencia Yeo-Johnson y Box-Cox)
* **Librería:** Scikit-Learn (`sklearn.preprocessing.PowerTransformer`)
* **¿Qué hace?:** Aplica transformaciones de potencia estabilizadoras de varianza para **modificar la forma de la distribución** de una variable continua con fuerte sesgo (cola larga) y aproximarla lo máximo posible a una distribución Normal Gaussiana. A diferencia del escalado simple (que solo mueve la variable de lugar), `PowerTransformer` busca automáticamente el parámetro de potencia óptimo $\lambda$.
  * **Métodos:**
    * `method='yeo-johnson'`: Admite valores de cero y negativos (método por defecto).
    * `method='box-cox'`: Requiere valores estrictamente positivos ($x > 0$).
* **¿Cómo usarla?:**
  * **Sintaxis:**
    ```python
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    df['col_pt'] = pt.fit_transform(df[['col_sesgada']])
    # Para recuperar la escala original:
    df['col_recuperada'] = pt.inverse_transform(df[['col_pt']])
    ```
  * **Ejemplo de código:**
    ```python
    from sklearn.preprocessing import PowerTransformer
    import pandas as pd

    gdp = vida['GDP'].dropna()
    pt = PowerTransformer(method='yeo-johnson')
    gdp_pt = pt.fit_transform(gdp.values.reshape(-1, 1))[:, 0]

    print("Asimetría original del GDP:", round(gdp.skew(), 3))
    print("Asimetría post Yeo-Johnson:", round(pd.Series(gdp_pt).skew(), 3))
    ```
  * **Output esperado / Resultado:**
    ```text
    Asimetría original del GDP: 3.212
    Asimetría post Yeo-Johnson: -0.015
    ```

---

### 9.9 `np.log1p()` y `np.expm1()` (Transformación Logarítmica de Forma)
* **Librería:** NumPy (`np.log1p` / `np.expm1`)
* **¿Qué hace?:** 
  * `np.log1p(x)`: Calcula el logaritmo natural de $1 + x$ ($\ln(1+x)$). Es la solución estándar para corregir distribuciones con fuerte sesgo positivo (cola larga a la derecha), comprimiendo valores elevados. Tolera valores iguales a cero evitando errores indeterminados ($\ln(0) = -\infty$).
  * `np.expm1(x)`: Función inversa que calcula $e^x - 1$, permitiendo revertir las predicciones o transformaciones logarítmicas de vuelta a la escala numérica original.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df['col_log'] = np.log1p(df['col'])` / `df['col_orig'] = np.expm1(df['col_log'])`
  * **Ejemplo de código:**
    ```python
    import numpy as np

    gdp = vida['GDP'].dropna()
    gdp_log = np.log1p(gdp)

    print("Asimetría original:", round(gdp.skew(), 3))
    print("Asimetría con log1p:", round(gdp_log.skew(), 3))
    ```
  * **Output esperado / Resultado:**
    ```text
    Asimetría original: 3.212
    Asimetría con log1p: 0.164
    ```

---

## Etapa 10: Flujo Correcto de Entrenamiento y Prevención de Data Leakage

En esta etapa se establecen los principios metodológicos fundamentales para separar los conjuntos de datos de entrenamiento y evaluación, garantizando la invalidez de supuestos falsos por filtración de información (*Data Leakage*).

### 10.1 `train_test_split()` (División del Dataset en Train y Test)
* **Librería:** Scikit-Learn (`sklearn.model_selection.train_test_split`)
* **¿Qué hace?:** Divide arreglos o DataFrames en subconjuntos de Entrenamiento (*Train*) y Prueba/Testeo (*Test*) de forma aleatoria o estratificada.
* **¿Cómo usarla?:**
  * **Sintaxis:** `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`
  * **Ejemplo de código:**
    ```python
    from sklearn.model_selection import train_test_split

    X = students[['math score', 'reading score', 'writing score']]
    y = students['gender']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train:", X_train.shape, " Test:", X_test.shape)
    ```
  * **Output esperado / Resultado:**
    ```text
    Train: (800, 3)  Test: (200, 3)
    ```

---

### 10.2 Regla de Oro: Separación de `fit_transform` en Train vs `transform` en Test
* **Librería:** Principio metodológico aplicado a todo estimador de Scikit-Learn (`StandardScaler`, `SimpleImputer`, `OneHotEncoder`, `PowerTransformer`).
* **¿Qué hace?:** Previene la **filtración de datos (*Data Leakage*)**. El método `fit` aprende parámetros (medias, varianzas, mínimos, máximos, modas, etc.) exclusivamente a partir de las muestras del conjunto de **Entrenamiento (`X_train`)**. Luego, dichos parámetros aprendidos se aplican mediante `transform` al conjunto de **Prueba (`X_test`)**.
* **Regla Inflexible de Trabajo:**
  1. `fit_transform()` $\rightarrow$ **SOLO sobre `X_train`**.
  2. `transform()` $\rightarrow$ **SOLO sobre `X_test`** (y futuros datos de producción).
  3. **No se escalan ni transforman** las variables indicadoras binarias (*dummies* resultantes de One-Hot Encoding).
* **Ejemplo de código:**
  ```python
  from sklearn.preprocessing import StandardScaler

  # ✅ FORMA CORRECTA: Aprendizaje de parámetros aislado en Train
  scaler = StandardScaler()
  X_train_esc = scaler.fit_transform(X_train) # fit_transform SOLO en train
  X_test_esc  = scaler.transform(X_test)      # solo transform en test

  print("Medias aprendidas únicamente de Train:", scaler.mean_.round(3))
  ```
* **Output esperado / Resultado:**
  ```text
  Medias aprendidas únicamente de Train: [66.021 69.105 68.044]
  ```

---

## Etapa 11: Exportación y Almacenamiento de Datos

En esta etapa final se persisten y guardan los datos limpios y procesados en disco en formatos estándar para consumo en modelos de Machine Learning, reportes o bases de datos.

### 11.1 `df.to_csv()` (Exportación de DataFrames a Archivos CSV)
* **Librería:** Pandas (Método de `DataFrame`)
* **¿Qué hace?:** Escribe y guarda el contenido de un objeto `DataFrame` procesado en un archivo físico delimitado en disco (CSV). Permite controlar parámetros de codificación, separador e inclusión/exclusión del índice de Pandas.
* **¿Cómo usarla?:**
  * **Sintaxis:** `df.to_csv(path_or_buf, sep=',', index=True, encoding='utf-8')`
  * **Parámetro `index`:** Si se establece en `False`, no escribe las etiquetas numéricas de las filas en el archivo resultante.
  * **Ejemplo de código:**
    ```python
    # Guardar el DataFrame procesado excluyendo la columna del índice ordinal
    students.to_csv('students_limpio.csv', index=False)
    ```
  * **Output esperado / Resultado:**
    ```text
    [Archivo Generado]: Se crea el archivo 'students_limpio.csv' en el directorio de trabajo actual
    con todas las variables escaladas y procesadas listo para la fase de modelado.
    ```

---

## Tabla Resumen: Mapeo de Funciones por Etapa del Pipeline

| Etapa del Pipeline | Librería | Función / Método / Atributo | Propósito Principal | Output Representativo |
| :--- | :--- | :--- | :--- | :--- |
| **1. Ingestión y Carga** | Pandas | `pd.read_csv()` | Carga de archivos CSV a DataFrame | `DataFrame` tabular |
| | Pandas | `pd.DataFrame()` | Constructor manual de DataFrame | `DataFrame` bidimensional |
| **2. Exploración Inicial** | Pandas | `df.head()` / `df.tail()` | Muestra inicial/final de filas | Primeras / últimas $n$ filas |
| | Pandas | `df.shape` | Dimensiones del DataFrame | Tupla `(filas, columnas)` |
| | Pandas | `df.columns` | Lista de nombres de columnas | Objeto `Index(['col1', ...])` |
| | Pandas | `df.dtypes` / `Series.dtype` | Tipos de datos por columna | Serie con tipos de datos |
| | Python / Pandas | `len(df)` | Cantidad de registros totales | Entero `N` |
| **3. Filtrado y Selección** | Pandas | `df.iloc[]` | Indexación por posición entera | Subconjunto por posición |
| | Pandas | `df.loc[]` | Indexación por etiquetas o condiciones | Subconjunto por etiquetas |
| | Pandas | `df[condicion]` | Filtrado booleano condicional | Subconjunto filtrado |
| | Pandas | `Series.idxmax()` / `idxmin()` | Índice de valores extremos | Etiqueta de índice de fila |
| **4. Limpieza y Diagnóstico** | Pandas | `df.isnull()` / `df.isna()` | Detección de valores faltantes | Máscara booleana |
| | Pandas | `df.isnull().sum()` | Conteo de nulos por columna | Serie con recuento de nulos |
| | Pandas | `df.dropna()` | Eliminación de filas/columnas con nulos | DataFrame reducido sin nulos |
| | Pandas | `pd.to_numeric()` | Conversión segura de tipos a numérico | Serie de tipo numérico |
| | SciPy | `stats.zscore()` | Cálculo de puntaje Z para outliers | Array de Z-scores / Filtro |
| | Pandas | `Series.skew()` | Coeficiente de asimetría de distribuciones | Valor flotante de asimetría |
| | SciPy | `stats.probplot()` | Gráfico Q-Q Plot para evaluar normalidad | Gráfica Q-Q sobre diagonal |
| **5. Transformación e Ing. Funciones** | Pandas | `df.rename()` | Cambia nombres de columnas | Lista de columnas actualizadas |
| | Pandas | `df.drop()` | Elimina columnas o filas | DataFrame sin columnas borradas |
| | Pandas | `df.reset_index()` | Reinicia el índice ordinal de filas | Índice entero limpio `0..N-1` |
| | Pandas | `pd.melt()` | Despivota de formato ancho a largo | DataFrame en formato largo |
| | Pandas | `pd.concat()` | Concatena DataFrames a lo ancho o largo | DataFrame unificado |
| | Pandas | `pd.get_dummies()` | Codificación One-Hot Encoding (Pandas) | Columnas binarias indicadoras |
| | Pandas | `pd.cut()` | Discretización/Binning de variables continuas | Serie categórica segmentada |
| | Pandas | `df.sort_values()` | Ordenamiento por valores de columna | DataFrame reordenado |
| **6. Agregación y Estadística** | Pandas | `df.describe()` | Resumen estadístico descriptivo completo | Tabla estadística completa |
| | Pandas | `.mean()`, `.min()`, `.max()`, `.quantile()` | Métodos estadísticos individuales | Valores escalares resumidos |
| | Pandas | `Series.unique()` | Extrae valores únicos sin duplicados | Array NumPy con categóricos |
| | Pandas | `Series.value_counts()` | Conteo de frecuencias categóricas | Serie ordenada por frecuencia |
| | Pandas | `df.groupby()` | Agrupación Split-Apply-Combine | Objeto agrupado / Serie agregada |
| | Pandas | `pd.crosstab()` | Tabla de contingencia cruzada | Tabla de frecuencias 2D |
| | Pandas | `df.corr()` | Matriz de correlación lineal | Matriz cuadrada de correlación |
| **7. Visualización (Seaborn)** | Seaborn | `sns.displot()` | Gráfico de distribución e histograma con KDE | Gráfico continuo de distribución |
| | Seaborn | `sns.countplot()` | Conteo de barras categóricas | Gráfico de barras categórico |
| | Seaborn | `sns.barplot()` | Barras de agregación con intervalos de confianza | Gráfico de medias categóricas |
| | Seaborn | `sns.boxplot()` | Diagrama de cajas y detección de outliers | Diagrama de cuartiles / outliers |
| | Seaborn | `sns.scatterplot()` | Diagrama de dispersión bivariado | Nube de puntos 2D |
| | Seaborn | `sns.pairplot()` | Matriz de dispersión pareada multivariada | Grilla de gráficos $N \times N$ |
| | Seaborn | `sns.heatmap()` | Mapa de calor de correlaciones | Matriz de celdas coloreadas |
| **8. Estilizado Visual** | Seaborn | `sns.despine()` | Remueve espinas/bordes externos del gráfico | Marco de gráfico simplificado |
| **9. Scikit-Learn (sklearn)** | Scikit-Learn | `SimpleImputer` | Imputación automática de nulos (media/moda) | Columnas imputadas sin `NaN` |
| | Scikit-Learn | `LabelEncoder` | Codificación ordinal/etiquetas enteras | Serie codificada `0, 1, 2...` |
| | Scikit-Learn | `OneHotEncoder` | Codificación nominal formal (Scikit-Learn) | Matriz binaria / DataFrame |
| | Scikit-Learn | `MinMaxScaler` | Re-escalado de características a rango (0, 100) | Columna re-escalada |
| | Scikit-Learn | `StandardScaler` | Estandarización a media 0 y varianza 1 | Columna estandarizada $Z$ |
| | Scikit-Learn | `RobustScaler` | Escalado resistente apoyado en mediana e IQR | Columna escalada robusta |
| | Scikit-Learn | `Normalizer` | Normalización vectorial por muestra (filas) | Matriz con norma unitaria |
| | Scikit-Learn | `PowerTransformer` | Transformación Yeo-Johnson / Box-Cox de potencia | Columna normalizada |
| | NumPy | `np.log1p()` / `np.expm1()` | Transformación logarítmica $\ln(1+x)$ e inversa | Columna corregida por sesgo |
| **10. Flujo Train/Test** | Scikit-Learn | `train_test_split()` | División del dataset en Train y Test | `X_train`, `X_test`, `y_train`, `y_test` |
| | Metodología | Regla de `fit_transform` | Aplicar `fit_transform` en Train y `transform` en Test | Prevención de Data Leakage |
| **11. Exportación de Datos** | Pandas | `df.to_csv()` | Guarda y persiste DataFrame en archivo CSV | Archivo `.csv` generado en disco |
