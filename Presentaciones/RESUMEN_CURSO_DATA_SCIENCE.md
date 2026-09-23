# Resumen Integral del Curso de Data Science
**Módulos II, III y IV: Manipulación de Datos, Análisis Exploratorio (AED), Feature Engineering y Fundamentos de Machine Learning**

---

## 📌 Índice General
1. [Introducción a Pandas (Módulo II - Clase 4)](#1-introducción-a-pandas-módulo-ii---clase-4)
2. [Análisis Exploratorio de Datos I: Descripción y Visualización (Módulo III - Clase 5)](#2-análisis-exploratorio-de-datos-i-descripción-y-visualización-módulo-iii---clase-5)
3. [Análisis Exploratorio de Datos II: Valores Faltantes, Outliers e IA (Módulo III - Clase 6)](#3-análisis-exploratorio-de-datos-ii-valores-faltantes-outliers-e-ia-módulo-iii---clase-6)
4. [Transformación de Datos I: Feature Engineering y Codificación (Módulo III - Clase 7)](#4-transformación-de-datos-i-feature-engineering-y-codificación-módulo-iii---clase-7)
5. [Transformación de Datos II: Estandarización y Normalización Clásica (Módulo III - Clase 8)](#5-transformación-de-datos-ii-estandarización-y-normalización-clásica-módulo-iii---clase-8)
6. [Diagnóstico Avanzado de Distribuciones, Transformaciones de Forma y Buenas Prácticas (Módulo III - Clase 10)](#6-diagnóstico-avanzado-de-distribuciones-transformaciones-de-forma-y-buenas-prácticas-módulo-iii---clase-10)
7. [Fundamentos de Machine Learning: ¿Qué es el ML y Familias de Algoritmos? (Módulo IV - Clase 11)](#7-fundamentos-de-machine-learning-qué-es-el-ml-y-familias-de-algoritmos-módulo-iv---clase-11)
8. [Clases Prácticas, Flujo de Notebooks y Pre-entregas (Clases 9, 10 y 11)](#8-clases-prácticas-flujo-de-notebooks-y-pre-entregas-clases-9-10-y-11)
9. [Matriz Comparativa Maestra de Técnicas de Preprocesamiento y Escalado](#9-matriz-comparativa-maestra-de-técnicas-de-preprocesamiento-y-escalado)
10. [Taxonomía de Machine Learning: Guía de Selección de Algoritmos](#10-taxonomía-de-machine-learning-guía-de-selección-de-algoritmos)

---

## 1. Introducción a Pandas (Módulo II - Clase 4)
*Material de referencia: `PANDAS.pdf`*

### 1.1. ¿Qué es Pandas?
**Pandas** (*PANel DAta / Python Data Analysis*) es la librería fundamental de Python para la manipulación, limpieza y análisis estructurado de datos. Construida sobre **NumPy**, extiende la capacidad de cálculo numérico agregando estructuras tabulares con etiquetas explícitas.

### 1.2. Estructuras Principales de Datos
* **Series**: Vector unidimensional homogéneo con un índice asociativo.
* **DataFrame**: Estructura bidimensional (tabla) donde:
  * Las **columnas** representan variables o *features* (tipo de dato homogéneo por columna).
  * Las **filas** representan observaciones/registros (pueden contener tipos heterogéneos).
  * Posee **dos índices**: uno de filas (*index*) y uno de columnas (*columns*).

### 1.3. Capacidades Clave
* **I/O (Entrada/Salida)**: Carga y exportación directa de formatos `CSV`, `Excel`, `SQL`, `JSON`, etc.
* **Indexación y Selección**: Acceso por etiquetas (`.loc[]`) o por posiciones enteras (`.iloc[]`).
* **Manipulación**: Reordenamiento, filtrado condicional, unión/concatenación (`merge`, `concat`), reagrupamiento (`groupby`) y reformateo de tablas.

---

## 2. Análisis Exploratorio de Datos I: Descripción y Visualización (Módulo III - Clase 5)
*Material de referencia: `clase5_intro_analisis_exploratorio2026.pptx.pdf`*

### 2.1. ¿Qué es el Análisis Exploratorio de Datos (AED / EDA)?
Es el proceso inicial indispensable en cualquier proyecto de Data Science cuyo objetivo es comprender la estructura subyacente de los datos mediante **medidas resumen** y **métodos gráficos**, detectando patrones, tendencias y valores atípicos.

### 2.2. Preguntas Clave durante el AED
1. **Dimensiones del Dataset**: ¿Cuántas observaciones (filas) y características (*features*/columnas) existen?
2. **Tipos de Variables**:
   * **Cualitativas / Categóricas**: Nominales u ordinales (representadas como `string` o `category`).
   * **Cuantitativas**: Discretas o Continuas (representadas como `int64` o `float64`).
3. **Calidad de Datos**: ¿Existen registros vacíos/faltantes? ¿Hay anomalías o errores de carga?

### 2.3. Análisis Univariado y Multivariado
* **Variables Continuas**:
  * *Gráficos*: Histogramas, Boxplots, Distplots.
  * *Métricas*: Media, Mediana, Rango, Cuartiles, Desviación Estándar.
  * *Evalúa*: Si la distribución es Normal, logarítmica o con sesgo (skewness).
* **Variables Categóricas**:
  * *Gráficos*: Countplots, Gráficos de barras.
  * *Métricas*: Tablas de frecuencias absolutas y relativas.
  * *Evalúa*: Balance/Desbalance de clases (crucial antes de entrenar clasificadores).
* **Relaciones entre Variables**:
  * *Scatterplots*, *Pairplots* y *Heatmaps* de correlación.
  * *Objetivo*: Identificar relaciones target-feature y **prevenir la Multicolinealidad** (evitar variables independientes altamente correlacionadas entre sí).

### 2.4. Librerías de Visualización
* **Matplotlib** (John Hunter, 2002): Librería base inspirada en MATLAB. Ofrece un control fino sobre todos los componentes de un gráfico.
* **Seaborn**: Construida sobre Matplotlib, optimizada para gráficos estadísticos de alto nivel, agregaciones automáticas y paletas estéticas listas para producción.

---

## 3. Análisis Exploratorio de Datos II: Valores Faltantes, Outliers e IA (Módulo III - Clase 6)
*Material de referencia: `clase6_exploracion_datos_parte2-2026.pdf`*

### 3.1. Tratamiento de Valores Faltantes (*Missing Values*)
Los valores faltantes ocurren cuando una variable no registra dato en ciertas instancias. Es vital analizar su **mecanismo de ausencia** antes de tomar decisiones:

#### Clasificación Teórica de Faltantes:
1. **MCAR (*Missing Completely at Random*)**: La probabilidad de falta es independiente de cualquier variable observada o no observada. No introduce sesgo al eliminar.
2. **MAR (*Missing at Random*)**: La falta depende de datos observados en otras variables del dataset.
3. **MNAR (*Missing Not at Random*)**: La falta depende directamente del valor no observado (ej. personas con altos ingresos que no responden el sueldo).

#### Estrategias de Manejo:
* **Eliminación**: Borrar filas o columnas. *Riesgo*: Puede perder gran volumen de información o introducir sesgos si no es MCAR.
* **Imputación**: Relleno con métricas estadísticas (media, mediana, moda) o algoritmos predictivos.
* **Imputación Asistida por IA**: Uso de copilotos de IA para proponer estrategias coherentes con las reglas de negocio.

---

### 3.2. Detección y Filtrado de Outliers (Valores Atípicos)
Un **Outlier** es una observación distante del resto de los datos que puede distorsionar significativamente los parámetros estadísticos.

#### Métodos de Filtrado según la Distribución:
1. **Método Z-Score** *(Recomendado para distribuciones simétricas/normales)*:
   $$\text{Z-Score} = \frac{X - \mu}{\sigma}$$
   *Criterio*: Se eliminan o marcan aquellas observaciones cuyo $|Z| > 3$ (más de 3 desviaciones estándar de la media $\mu$).

2. **Rango Intercuartílico - RIC / IQR** *(Recomendado para distribuciones sesgadas o no normales)*:
   $$\text{RIC} = Q_3 - Q_1$$
   *Límites de Aceptación*:
   $$\text{Límite Inferior} = Q_1 - 1.5 \times \text{RIC}$$
   $$\text{Límite Superior} = Q_3 + 1.5 \times \text{RIC}$$
   *Criterio*: Todo valor fuera del intervalo $[\text{Límite Inferior}, \text{Límite Superior}]$ es considerado outlier.

---

## 4. Transformación de Datos I: Feature Engineering y Codificación (Módulo III - Clase 7)
*Material de referencia: `clase7_Análisis exploratorio de datos_ Transformación de datos2026.pptx.pdf`*

### 4.1. Concepto de Feature Engineering
El **Feature Engineering** busca extraer, transformar y construir las variables más relevantes para que el problema de Machine Learning sea optimizado eficientemente.

### 4.2. Obtención de Variables Derivadas
Creación de nuevas columnas mediante operaciones matemáticas entre variables existentes (suma, resta, multiplicación, cocientes/ratios) para capturar interacciones complejas.

### 4.3. Categorical Encoding (Codificación Categórica)
Los algoritmos de Machine Learning requieren entradas estrictamente numéricas.

| Técnica | Descripción | Método/Librería | Tipo de Variable |
| :--- | :--- | :--- | :--- |
| **LabelEncoder** | Mapea categorías a enteros de $0$ a $k-1$. | `sklearn.preprocessing.LabelEncoder` | Ordinales (con jerarquía) |
| **get_dummies** | Genera columnas binarias indicadoras ($0$ o $1$). Modifica el DataFrame original. | `pandas.get_dummies()` | Nominales |
| **OneHotEncoder** | Genera columnas binarias indicadoras. Pensado para pipelines. No modifica in-place. | `sklearn.preprocessing.OneHotEncoder` | Nominales |

### 4.4. Discretización (Binning)
Conversión de variables continuas en rangos o intervalos categóricos para reducir ruido o modelar comportamientos no lineales.

### 4.5. Re-escalado de Variables con MinMaxScaler
Escala las características a un rango acotado $[0, 1)$.
$$\text{MinMaxScaler}(X) = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$
* **Aplicación fundamental**: Algoritmos basados en distancias como **K-Nearest Neighbors (KNN)** y **Support Vector Machines (SVM)**.

---

## 5. Transformación de Datos II: Estandarización y Normalización Clásica (Módulo III - Clase 8)
*Material de referencia: `clase8_Transformación de datos y evaluación de distribuciones 2026.pptx.pdf`*

### 5.1. Estandarización (`StandardScaler`)
Transforma las variables para que tengan **media $\mu = 0$** y **desviación estándar $\sigma = 1$**.

$$Z = \frac{X - \mu}{\sigma}$$

* **Efecto**: Preserva la forma de la distribución original pero iguala la escala de variabilidad entre características.
* **Algoritmos recomendados**:
  * Support Vector Machines (SVM)
  * Regresión Logística
  * Redes Neuronales

### 5.2. Normalización Vectorial (`Normalizer`)
Escala las observaciones (vectores de fila) de modo que tengan una norma unitaria ($L_1$ o $L_2$).
> ⚠️ **Aclaración crítica (ver sección 6.3):** `Normalizer` no altera la distribución estadística de una variable ni la vuelve gaussiana. Opera transversalmente por registro.

---

## 6. Diagnóstico Avanzado de Distribuciones, Transformaciones de Forma y Buenas Prácticas (Módulo III - Clase 10)
*Material de referencia: `Clase10_Transformación_de_datos_y_evaluación_de_distribuciones.pdf`*

### 6.1. ¿Por qué es Crucial Evaluar la Distribución de las Variables?
1. **Asunciones de Modelos Paramétricos**: Ciertos algoritmos asumen que los residuos de las variables siguen una distribución normal (ej. Mínimos Cuadrados Ordinarios en Regresión Lineal).
2. **Sensibilidad a Escalas en Modelos basados en Distancias**: En algoritmos como **KNN**, **K-Means** o **SVM**, una variable con escala de miles (ej. Ingresos en $) domina numéricamente sobre variables con escala unitaria (ej. Edad en años), anulando su peso predictivo.
3. **Velocidad de Convergencia en Gradiente Descendente**: En modelos optimizados por descenso por gradiente (**Regresión Logística**, **Redes Neuronales**), escalas dispares provocan oscilaciones en el plano de pérdida y enlentecen drásticamente la convergencia hacia el mínimo global.
4. **Distorsión en Métricas Descriptivas**: Cuando una variable es marcadamente asimétrica, la media aritmética y la desviación estándar dejan de ser medidas representativas del centro y la dispersión.

---

### 6.2. Herramientas de Diagnóstico: Métodos Gráficos, Numéricos y Tests

#### A. Diagnóstico Gráfico:
* **Histograma**: Revela la forma global, bimodalidad y acumulación de frecuencias.
* **Boxplot**: Visualiza la mediana, dispersión intercuartílica (RIC) y la presencia de candidatos a outliers.
* **Q-Q Plot (*Quantile-Quantile Plot*)**: Gráfica que compara los cuantiles empíricos observados frente a los cuantiles teóricos de una distribución normal estándar. Si los puntos se alinean sobre la diagonal de 45°, la variable es aproximadamente normal. Curvaturas en los extremos señalan colas pesadas o asimetría.

#### B. Métricas Numéricas:
* **Asimetría (*Skewness*) — `df['col'].skew()`**:
  * $\text{Skew} \approx 0$: Distribución simétrica (forma acampanada).
  * $\text{Skew} > 0$: Asimetría positiva (cola alargada a la derecha; común en ingresos, precios, costos).
  * $\text{Skew} < 0$: Asimetría negativa (cola alargada a la izquierda; común en tasas de graduación, edades de jubilación).
  * *Criterio de decisión*: $|\text{Skew}| < 0.5$ se considera aceptable; $|\text{Skew}| > 1$ **exige aplicar transformaciones de forma**.
* **Curtosis (*Kurtosis*) — `df['col'].kurt()`**:
  * Mide la concentración de datos en las colas respecto a la distribución normal (mesocúrtica).
  * Valores marcadamente positivos (leptocúrtica) indican colas pesadas y anticipan la presencia recurrente de valores atípicos (*outliers*).

#### C. Tests Estadísticos de Normalidad:
* **Shapiro-Wilk (`scipy.stats.shapiro`)**: Recomendado para muestras pequeñas ($n < 5000$).
* **D'Agostino-Pearson (`scipy.stats.normaltest`)**: Recomendado para muestras medianas y grandes combinando asimetría y curtosis.
* **Kolmogorov-Smirnov (`scipy.stats.kstest`)**: Compara contra cualquier distribución continua teórica.

> ⚠️ **La Trampa del $p$-value en Grandes Datasets ($n$ grande):**  
> La hipótesis nula es $H_0$: *los datos provienen de una distribución normal*. Si $p < 0.05$, se rechaza $H_0$.  
> Sin embargo, **con muestras de cientos de miles de registros (como los 683.000 árboles de NYC), cualquier desviación infinitesimal de la perfección matemática arrojará $p < 0.0001$**. Un test que rechaza la normalidad en un dataset grande no significa que la variable sea inutilizable; **el test estadístico nunca debe decidir solo**: siempre se debe complementar con la inspección del Q-Q Plot y el valor de asimetría.

---

### 6.3. Los Tres Scalers por Columna vs. `Normalizer()`

> 💡 **Principio Fundamental:** **Escalar NO cambia la forma de la distribución.**  
> Escalar simplemente traslada el origen (centro) y modifica la escala de los ejes. Si una variable tiene asimetría positiva antes de aplicar `StandardScaler` o `MinMaxScaler`, seguirá teniendo exactamente la misma asimetría después de escalarla.

1. **`StandardScaler()` (Estandarización Z-Score)**:
   * Centra en media $\mu = 0$ y escala a desviación estándar $\sigma = 1$.
   * Rango libre (no acotado).
   * **Sensible a outliers:** Tanto la media como el desvío se ven severamente distorsionados por valores extremos.
   * *Ideal para:* Regresión Logística, SVM, Redes Neuronales y PCA.

2. **`MinMaxScaler()` (Re-escalado Acotado)**:
   * Transforma todas las observaciones al intervalo cerrado $[0, 1]$.
   * **Extremadamente sensible a outliers:** Un único valor atípico gigante comprimirá a todo el 99.9% de los datos contra el valor 0, anulando la resolución de la variable.
   * *Ideal para:* KNN, K-Means, Redes Neuronales e imágenes (píxeles).

3. **`RobustScaler()` (Escalado Robusto basado en Cuartiles)**:
   * Utiliza la **Mediana** ($Q_2$) como centro y el **Rango Intercuartílico** ($\text{RIC} = Q_3 - Q_1$) como escala:
     $$\text{RobustScaler}(X) = \frac{X - \text{Mediana}}{\text{RIC}}$$
   * **Inmune a outliers:** Ni la mediana ni el RIC se inmutan ante la presencia de valores anómalos o extremos.
   * *Ideal para:* Datasets del mundo real donde los outliers son **información legítima y representativa del negocio** (ej. transacciones bancarias, salarios) y no meros errores de tipeo.

4. **El Gran Error Conceptual: `Normalizer()`**:
   * `Normalizer()` **NO convierte una variable en una Distribución Normal**, ni escala las columnas.
   * Trabaja **por fila (a nivel de registro)**: reescala cada observación como un vector para que su norma euclidiana ($L_2$) o Manhattan ($L_1$) sea igual a $1$.
   * Si se aplica sobre una sola columna, todos los registros se transformarán trivialmente en $1.0$.
   * *Uso real:* Procesamiento de Lenguaje Natural (NLP, TF-IDF), comparación de textos por similitud coseno o clustering de documentos.

---

### 6.4. Transformaciones de la Forma de la Distribución (Reducción de Asimetría)

Cuando una variable presenta una asimetría marcada ($|\text{skew}| > 1$), es necesario **alterar la geometría de la distribución**:

1. **Transformación Logarítmica — `np.log1p(x)`**:
   * Calcula $\ln(1 + x)$. Se utiliza `log1p` en vez de `log` para evitar la indeterminación matemática de $\ln(0)$ cuando la variable contiene ceros.
   * Comprime fuertemente los valores astronómicos de la cola derecha y expande los valores pequeños.
   * Requiere $x \ge 0$.
   * *Inversión para interpretación:* Si el modelo predice en la escala transformada, se debe aplicar `np.expm1()` para devolver los resultados a la escala original de negocio ($).

2. **Raíz Cuadrada — `np.sqrt(x)`**:
   * Produce una compresión más moderada que el logaritmo. Ideal para variables que representan conteos discretos (ej. cantidad de llamadas, visitas).

3. **PowerTransformer (`sklearn.preprocessing.PowerTransformer`)**:
   * Busca paramétricamente el exponente óptimo $\lambda$ que maximiza la similitud de la variable transformada con una campana de Gauss:
     * **Box-Cox**: Exige valores estrictamente positivos ($x > 0$).
     * **Yeo-Johnson**: Generalización moderna que acepta valores iguales a cero y negativos ($x \le 0$).

4. **QuantileTransformer (`sklearn.preprocessing.QuantileTransformer`)**:
   * Aplica un mapeo no lineal basado en la función de distribución empírica para forzar los datos hacia una distribución uniforme o normal. Es la herramienta definitiva cuando la distribución es bimodal, multimodal o no cede ante transformaciones de potencia.

---

### 6.5. Buenas Prácticas Rigurosas: Prevención de Data Leakage

```
                       DATASET COMPLETO (X, y)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
          CONJUNTO DE TRAIN                CONJUNTO DE TEST
                  │                               │
        scaler.fit_transform()              scaler.transform()
                  │                               │
       (Aprende media y desvío)         (Aplica los parámetros de train)
```

#### A. La Regla de Oro: `fit` en Train, `transform` en Test
* **El Scaler es un modelo en sí mismo**: aprende parámetros de los datos ($\mu$, $\sigma$, mínimo, máximo, mediana, RIC).
* Si se ejecuta `scaler.fit_transform(X)` sobre el dataset completo antes del split, **la información estadística del conjunto de prueba (test) se filtra dentro del entrenamiento**. Esto constituye **Data Leakage (Fuga de Datos)**: produce métricas de rendimiento artificialmente optimistas que fracasan rotundamente al desplegar el modelo en producción.
* **Regla mnemotécnica:** `fit_transform()` se ejecuta **una única vez sobre el conjunto de entrenamiento (`X_train`)**. Sobre el conjunto de prueba (`X_test`), validación o nuevos datos futuros, se ejecuta **exclusivamente `transform()`**.

#### B. ¿Qué Variables se Escalan y Cuáles NO?
* **SÍ se escalan**: Variables cuantitativas continuas con magnitudes o unidades de medida diferentes (ej. altura en metros vs. peso en kg).
* **NO se escalan**:
  * Variables indicadoras / dummies binarias ($0$ o $1$).
  * Variables ordinales codificadas donde el espacio entre clases posee un significado específico.
  * La variable objetivo o target ($y$) en problemas de clasificación.

#### C. Sensibilidad según la Familia de Algoritmos:
* **Exigen Escalado Obligatorio**: KNN, K-Means, SVM, Regresión Logística, Redes Neuronales (MLP/Deep Learning), PCA.
* **Inmunes al Escalado (No lo necesitan)**: **Árboles de Decisión, Random Forest y Gradient Boosting (XGBoost, LightGBM, CatBoost)**. Los algoritmos basados en árboles toman decisiones mediante divisiones ortogonales (*splits*) basadas en percentiles o umbrales individuales de cada feature ($X_i > c$), por lo que una transformación monotónica de escala no altera en absoluto la estructura de las particiones.

---

## 7. Fundamentos de Machine Learning: ¿Qué es el ML y Familias de Algoritmos? (Módulo IV - Clase 11)
*Material de referencia: `Clase11_Fundamentos_de_ML.pdf`*

### 7.1. Definición y Ubicación Disciplinar
El **Machine Learning (Aprendizaje Automático)** es la rama de la **Inteligencia Artificial** que se enfoca en el desarrollo de algoritmos capaces de identificar patrones complejos en los datos para aprender a resolver tareas específicas de manera autónoma, sin requerir reglas programadas explícitamente a mano.

```
┌─────────────────────────────────────────────────────────────┐
│  INTELIGENCIA ARTIFICIAL (IA)                               │
│  Sistemas capaces de emular capacidades cognitivas humanas  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  MACHINE LEARNING (ML)                                │  │
│  │  Modelos que aprenden patrones a partir de datos      │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  DEEP LEARNING / REDES NEURONALES               │  │  │
│  │  │  Arquitecturas profundas de capas jerárquicas   │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

### 7.2. Los Cuatro Conceptos Fundamentales (Definición Formal de Aprendizaje)
Siguiendo la formalización canónica del aprendizaje automático (propuesta originalmente por Tom Mitchell):

> *"Se dice que un programa de computadora **aprende** de la experiencia **E**, con respecto a una clase de tareas **T** y una medida de rendimiento **P**, si su rendimiento en las tareas en **T**, medido por **P**, mejora con la experiencia **E**."*

1. **Tarea ($T$)**: La función objetivo concreta que se busca resolver en el mundo real.
   * *Ejemplos:* Clasificar correos como Spam o No-Spam, detectar transacciones fraudulentas, diagnosticar la salud de un árbol, estimar el valor comercial de una propiedad.
2. **Experiencia ($E$)**: El conjunto de datos a partir del cual el modelo extrae el conocimiento. Sus pilares de viabilidad son:
   * **Cantidad:** Volumen estadístico suficiente de filas/registros.
   * **Calidad:** Datos limpios, sin inconsistencias severas ni sesgos de recolección.
   * **Representatividad:** Las observaciones de entrenamiento deben reflejar fielmente la distribución de los casos reales que el modelo encontrará en producción.
   * **Supervisión:** Presencia o ausencia de la variable respuesta / target etiquetada.
3. **Algoritmo ($A$)**: La secuencia matemática y computacional que explora el espacio de hipótesis para ajustar los parámetros de un modelo a partir de las variables de entrada ($X$).
4. **Aprendizaje / Rendimiento ($P$)**: Métrica cuantitativa que evalúa el éxito del modelo. Hay aprendizaje cuando $P$ sobre datos nunca antes vistos mejora progresivamente a medida que el algoritmo asimila mayor experiencia $E$.

---

### 7.3. Los Tres Componentes de un Proyecto de Machine Learning
En cualquier flujo de trabajo de ML intervienen tres entidades bien diferenciadas:

```
    [ DATOS (Experiencia) ]  +  [ ALGORITMO (Procedimiento) ]
                               │
                        PROCESO DE AJUSTE
                            (Entrenamiento)
                               │
                               ▼
                   [ PREDICTOR / MODELO FINAL ]
                               │
            Recibe nuevo caso (X) ──► Devuelve Predicción (y)
```

1. **Datos (Entrada del proceso)**: La materia prima histórica que contiene las covariables predictoras y, en aprendizaje supervisado, la etiqueta objetivo.
2. **Algoritmos (Mecanismo de optimización)**: El procedimiento que busca, entre una familia de modelos posibles, cuál minimiza el error empírico sobre los datos.
3. **Predictores (Salida del proceso / Modelo entrenado)**: Es el artefacto computacional resultante del ajuste (ej. los pesos entrenados de una regresión, la estructura de ramas de un árbol de decisión). Es el objeto que se exporta y consume en producción para recibir datos nuevos y devolver inferencias.

---

### 7.4. Taxonomía: ¿Cómo Elegir la Familia de Algoritmos Adecuada?
La selección del modelo no se realiza al azar; depende estrictamente de dos factores: la **Tarea** y la **Información disponible**:

```
                              ¿SE TIENE VARIABLE TARGET ETIQUETADA?
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
                   SÍ (Supervisado)                         NO (No Supervisado)
                     │                                         │
        ¿Qué tipo de Target se predice?          ¿Cuál es el objetivo analítico?
          ┌──────────┴──────────┐                     ┌────────┴────────┐
          ▼                     ▼                     ▼                 ▼
   CATEGÓRICA / CLASES      NUMÉRICA CONTINUA     DESCUBRIR GRUPOS   REDUCIR VARIABLES
     (Clasificación)           (Regresión)          (Clustering)     (Reducción Dim.)
```

1. **Aprendizaje Supervisado: Clasificación**:
   * *Naturaleza de la salida:* Variable discreta, cualitativa o categórica ($y \in \{0, 1\}$ o $\{c_1, c_2, \dots, c_k\}$).
   * *Casos de uso:* Detectar spam, abandono de clientes (*churn*), diagnóstico médico, predecir si un árbol dañará la vereda (`sidewalk`).
   * *Modelos representativos:* Regresión Logística, Árboles de Decisión, Random Forest, SVM, Naive Bayes.
2. **Aprendizaje Supervisado: Regresión**:
   * *Naturaleza de la salida:* Variable numérica cuantitativa continua ($y \in \mathbb{R}$).
   * *Casos de uso:* Estimación de precios inmobiliarios, pronóstico de ventas, estimación del diámetro de un árbol (`tree_dbh`).
   * *Modelos representativos:* Regresión Lineal, Ridge/Lasso, Árboles de Regresión, Random Forest Regressor, SVR.
3. **Aprendizaje No Supervisado: Clustering**:
   * *Naturaleza de la salida:* No existe variable target; el objetivo es descubrir agrupamientos naturales o patrones de afinidad intrínsecos en los datos.
   * *Casos de uso:* Segmentación de clientes por comportamiento de compra, detección de zonas urbanas críticas.
   * *Modelos representativos:* K-Means, DBSCAN, Clustering Jerárquico.
4. **Aprendizaje No Supervisado: Reducción de Dimensionalidad**:
   * *Naturaleza de la salida:* Proyección de un espacio de alta dimensionalidad ($p$ columnas) a un subespacio de menor dimensión ($k < p$) preservando la máxima varianza posible.
   * *Casos de uso:* Visualización de datos complejos en 2D/3D, compresión de señales, eliminación de multicolinealidad.
   * *Modelos representativos:* PCA (Análisis de Componentes Principales), t-SNE, UMAP.

---

## 8. Clases Prácticas, Flujo de Notebooks y Pre-entregas (Clases 9, 10 y 11)

### 8.1. Mapa de Ruta de Notebooks del Curso
* **Notebooks 5 y 7**: Fundamentos de computación científica con NumPy y manejo tabular con Pandas (Base de la Primera Pre-entrega).
* **Notebook 6**: Operaciones avanzadas de filtrado, indexación y transformación con Pandas.
* **Notebook 8**: Estadística descriptiva inicial y visualización con Matplotlib y Seaborn.
* **Notebook 9**: Limpieza de datos: diagnóstico de mecanismos de faltantes y detección de outliers mediante IQR y Z-Score.
* **Notebook 10**: Exploración guiada e integradora de datasets reales.
* **Notebook 11**: Feature Engineering I: generación de variables derivadas, discretización (*binning*) y codificación categórica (`LabelEncoder`, `OneHotEncoder`).
* **Notebook 12**: Transformación de datos y evaluación de distribuciones: aplicación comparativa de `StandardScaler`, `MinMaxScaler`, `RobustScaler`, transformaciones de forma (`log1p`, `PowerTransformer`) y prevención de data leakage con split train/test.
* **Notebook 13**: Taller práctico integrador de transformación y preparación completa de datos para modelado.
* **Notebook 14**: Primeros pasos en Machine Learning: definición de Tarea, Experiencia y Algoritmo sobre un caso real. Entrenamiento de un Árbol de Decisión (`DecisionTreeClassifier`), evaluación del impacto de la profundidad máxima (`max_depth`) y medición del sobreajuste (*overfitting*) sobre datos no vistos.

### 8.2. Hitos de Pre-entrega y Evaluación
* **Primera Pre-entrega**: Entrega de repositorios en GitHub con las Notebooks 5 y 7 resueltas.
* **Segunda Pre-entrega (Consolidación Módulo III)**:
  1. Elección del tema y dataset de trabajo grupal.
  2. Justificación y definición del objetivo analítico del proyecto.
  3. Limpieza, análisis de nulos y tratamiento de outliers con Pandas.
  4. Evaluación de distribuciones y selección justificada del escalado/transformación correspondiente.
  5. Visualizaciones de soporte e interpretaciones documentadas en GitHub.

---

## 9. Matriz Comparativa Maestra de Técnicas de Preprocesamiento y Escalado

| Técnica | Fórmula / Método Matemático | Rango de Salida | Sensibilidad a Outliers | ¿Cambia la Forma de la Distribución? | Modelos Recomendados |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **StandardScaler** | $Z = \frac{X - \mu}{\sigma}$ | Sin límites (centrada en $0$, $\sigma=1$) | **Alta** (media y desvío se distorsionan) | **NO** (mantiene asimetría original) | Regresión Logística, SVM, Redes Neuronales, PCA |
| **MinMaxScaler** | $\frac{X - X_{\min}}{X_{\max} - X_{\min}}$ | Acotado a $[0, 1]$ | **Extrema** (un outlier comprime los demás datos) | **NO** (mantiene asimetría original) | KNN, K-Means, Redes Neuronales, Imágenes |
| **RobustScaler** | $\frac{X - \text{Mediana}}{\text{RIC}}$ | Sin límites (centrada en mediana $0$) | **NULA / Robusto** (mediana y RIC no se alteran) | **NO** (mantiene asimetría original) | Modelos lineales o de distancia cuando los outliers son legítimos |
| **Log Transformation** | $\ln(1 + X)$ (`np.log1p`) | $[0, \infty)$ para $X \ge 0$ | Reduce el impacto de valores gigantes | **SÍ** (comprime cola derecha, reduce asimetría positiva) | Modelos que asumen normalidad con variables de ingresos, precios o costos |
| **PowerTransformer** | Box-Cox ($X>0$) o Yeo-Johnson ($\forall X$) | Aproximado a normal estándar | Reduce la influencia de extremos | **SÍ** (estabiliza varianza y fuerza simetría gaussiana) | Regresión Lineal, LDA, Naive Bayes |
| **Normalizer** | $\frac{\mathbf{x}}{\|\mathbf{x}\|_2}$ (opera por fila) | Norma unitaria por registro | No aplica a nivel columna | Modifica la longitud del vector fila | NLP, TF-IDF, Similitud Coseno, Text Mining |
| **LabelEncoder** | Categoría $\to \{0, 1, \dots, k-1\}$ | Enteros $[0, k-1]$ | No aplica | Asigna orden numérico arbitrario | Variables ordinales; Árboles de Decisión, Random Forest |
| **OneHotEncoder / get_dummies** | Categoría $\to$ Vector binario | $\{0, 1\}$ por cada nivel | No aplica | Expande cardinalidad a $k$ columnas | Regresión Lineal/Logística, SVM, Redes Neuronales |

---

## 10. Taxonomía de Machine Learning: Guía de Selección de Algoritmos

| Paradigma | Tipo de Tarea | Variable Objetivo ($y$) | Pregunta Central de Negocio | Algoritmos Clásicos | Métricas de Evaluación Clave |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Supervisado** | **Clasificación Binaria** | Categórica (2 clases: $\{0, 1\}$) | *¿El árbol causará daño a la vereda? (Sí/No)* | Regresión Logística, Random Forest, SVM | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **Supervisado** | **Clasificación Multiclase** | Categórica ($>2$ clases) | *¿Cuál es el estado de salud del árbol? (Good/Fair/Poor)* | Random Forest, Decision Tree, Naive Bayes | Macro/Weighted F1-Score, Matriz de Confusión |
| **Supervisado** | **Regresión Continua** | Numérica continua ($y \in \mathbb{R}$) | *¿Cuál será el diámetro del tronco del árbol en pulgadas?* | Regresión Lineal, Ridge, Lasso, SVR, Gradient Boosting | MAE, MSE, RMSE, $R^2$ Score |
| **No Supervisado** | **Clustering (Agrupamiento)** | Sin etiqueta objetivo | *¿Existen zonas geográficas de la ciudad con patrones similares de arbolado?* | K-Means, DBSCAN, Hierarchical Clustering | Silhouette Score, Davies-Bouldin Index |
| **No Supervisado** | **Reducción de Dimensionalidad** | Sin etiqueta objetivo | *¿Cómo sintetizar 45 características urbanas en 2 componentes visualizables?* | PCA, t-SNE, UMAP | Varianza Explicada Acumulada |

---
*Resumen exhaustivo del curso de Data Science — Actualizado con Módulos II, III y IV (Clases 4 a 11).*
