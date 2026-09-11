# Resumen Integral del Curso de Data Science
**Módulos II y III: Manipulación de Datos, Análisis Exploratorio (AED) y Feature Engineering**

---

## 📌 Índice General
1. [Introducción a Pandas (Módulo II - Clase 4)](#1-introducción-a-pandas-módulo-ii---clase-4)
2. [Análisis Exploratorio de Datos I: Descripción y Visualización (Módulo III - Clase 5)](#2-análisis-exploratorio-de-datos-i-descripción-y-visualización-módulo-iii---clase-5)
3. [Análisis Exploratorio de Datos II: Valores Faltantes, Outliers e IA (Módulo III - Clase 6)](#3-análisis-exploratorio-de-datos-ii-valores-faltantes-outliers-e-ia-módulo-iii---clase-6)
4. [Transformación de Datos I: Feature Engineering y Codificación (Módulo III - Clase 7)](#4-transformación-de-datos-i-feature-engineering-y-codificación-módulo-iii---clase-7)
5. [Transformación de Datos II: Estandarización, Normalización y Distribuciones (Módulo III - Clase 8)](#5-transformación-de-datos-ii-estandarización-normalización-y-distribuciones-módulo-iii---clase-8)
6. [Clase Práctica, Integración y Pre-entregas (Módulo III - Clase 9)](#6-clase-práctica-integración-y-pre-entregas-módulo-iii---clase-9)
7. [Matriz Comparativa de Técnicas de Preprocesamiento](#7-matriz-comparativa-de-técnicas-de-preprocesamiento)

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

## 5. Transformación de Datos II: Estandarización, Normalización y Distribuciones (Módulo III - Clase 8)
*Material de referencia: `clase8_Transformación de datos y evaluación de distribuciones 2026.pptx.pdf`*

### 5.1. Estandarización (`StandardScaler`)
Transforma las variables para que tengan **media $\mu = 0$** y **desviación estándar $\sigma = 1$**.

$$Z = \frac{X - \mu}{\sigma}$$

* **Efecto**: Preserva la forma de la distribución original pero iguala la escala de variabilidad entre características.
* **Algoritmos recomendados**:
  * Support Vector Machines (SVM)
  * Regresión Logística
  * Redes Neuronales

### 5.2. Normalización (`Normalizer`)
Escala las observaciones (vectores de fila) de modo que tengan una norma unitaria, o transforma las variables hacia una **Distribución Normal** en el rango $[0, 1]$.

* **Algoritmos recomendados**:
  * Regresión Lineal
  * Análisis Discriminante Lineal (LDA)
  * Clasificador Naive Bayesiano

### 5.3. Evaluación Visual Comparativa
Es crucial graficar distribuciones (con Seaborn/Matplotlib) **antes y después** de aplicar `StandardScaler`, `MinMaxScaler` o `Normalizer` para confirmar que las asunciones del modelo se satisfagan.

---

## 6. Clase Práctica, Integración y Pre-entregas (Módulo III - Clase 9)
*Material de referencia: `clase9_clase_practica_2026.pptx.pdf`*

### 6.1. Flujo Integrador de Trabajo en Notebooks
El curso sigue una secuencia pedagógica práctica reflejada en las notebooks de trabajo:
* **Notebooks 5 y 7**: Fundamentos de NumPy y Pandas (Base de la Primera Pre-entrega).
* **Notebook 6**: Funcionalidades avanzadas de Pandas.
* **Notebook 8**: Descripción estática y visualización con Matplotlib/Seaborn.
* **Notebook 9**: Limpieza de datos faltantes e identificación de Outliers.
* **Notebook 10**: Exploración guiada e integradora de datasets.
* **Notebook 11**: Feature Engineering (Variables derivadas, LabelEncoder, OneHotEncoder).
* **Notebook 12**: Estandarización (`StandardScaler`), Normalización y Re-escalado (`MinMaxScaler`).
* **Notebook 13**: Consolidación y preparación completa de datos para Machine Learning (Módulo IV).

### 6.2. Hitos de Evaluación / Pre-entregas
* **Primera Pre-entrega**: Entrega de repositorios en GitHub con las Notebooks 5 y 7 resueltas.
* **Segunda Pre-entrega (Clase 10)**:
  1. Elección de dataset y tema de trabajo en grupo.
  2. Definición del objetivo del proyecto.
  3. Exploración, limpieza y manejo de faltantes/outliers con Pandas.
  4. Feature Engineering, estandarización y transformación de variables.
  5. Entregables: Código en repositorio público de GitHub y presentación del avance.

---

## 7. Matriz Comparativa de Técnicas de Preprocesamiento

| Técnica | Fórmula / Método | Rango Resultante | Propiedades Principales | Algoritmos Típicos |
| :--- | :--- | :--- | :--- | :--- |
| **MinMaxScaler** | $\frac{X - X_{\min}}{X_{\max} - X_{\min}}$ | $[0, 1]$ | Sensible a outliers; conserva distancias relativas. | KNN, SVM, Redes Neuronales |
| **StandardScaler** | $\frac{X - \mu}{\sigma}$ | Sin límites (centrado en $0$, $\sigma=1$) | Mantiene forma de distribución; robusto ante varianzas desiguales. | Regresión Logística, SVM, PCA, Redes Neuronales |
| **Normalizer** | Norma del vector $L_1$ o $L_2$ | Rescalado vectorial / $[0, 1]$ | Transforma cada muestra individual a norma unitaria. | Regresión Lineal, LDA, Naive Bayes, Text Mining |
| **LabelEncoder** | Categoría $\to \{0, 1, \dots, k-1\}$ | Enteros $[0, k-1]$ | Compacto; introduce orden implícito (usar con precaución). | Árboles de Decisión, Random Forest, XGBoost |
| **OneHotEncoder / get_dummies** | Categoría $\to$ Vector binario | $\{0, 1\}$ | Elimina orden artificial; aumenta cardinalidad/columnas. | Regresión Lineal/Logística, Redes Neuronales, SVM |

---
*Resumen generado para el curso de Data Science - Presentaciones Módulos II y III.*
