
## 01 · Portada 

> Les voy a contar un poco qué hicimos en el proyecto final del curso de Data Science con Juan Ignacio y Gabriel. Le pusimos **VeredaSegura**.
>
> Básicamente agarramos un dataset real de árboles de Nueva York y lo usamos para recorrer todo el proceso del curso: limpiar datos, explorar patrones, entrenar modelos que predicen, probar algoritmos sin etiquetas y dejarlo andando como un servicio chiquito. La idea es contarles qué fuimos haciendo, qué funcionó y dónde nos chocamos con los límites de los datos.

---

## 02 · El problema 

> La pregunta de partida era bastante concreta: mirando solo las características de un árbol en la calle —qué tan grueso es, si tiene problemas en las raíces, qué especie es, en qué barrio está—, ¿podemos estimar si la vereda de al lado figura como dañada?
>
> En la práctica, algo así serviría para que un municipio ordene por dónde empezar a inspeccionar, en vez de salir a recorrer cuadras al azar o esperar a que alguien reclame.
>
> Eso sí, desde el principio pusimos un límite claro para no vender humo: los datos son una foto de 2015. Así que el proyecto clasifica daño que ya se observó en ese momento; no predice cuándo se va a romper una vereda a futuro ni demuestra que el árbol sea el culpable.

---

## 03 · Los datos 

> El dataset viene del portal de datos abiertos de Nueva York: es un censo de 2015 hecho con voluntarios que tiene más de **683 mil árboles** registrados.
>
> De ahí filtramos solo árboles vivos con el estado de la vereda anotado y armamos una muestra reproducible de **100 mil árboles**. Y acá hicimos algo clave antes de tocar cualquier modelo: separamos **80 mil para entrenar** y guardamos **20 mil bajo llave** para el examen final. Si el modelo practica con los mismos datos con los que después lo evaluás, te miente porque memoriza.
>
> De cada árbol tenemos tamaño, conflictos visibles en raíces, tronco y ramas, salud, si algún vecino lo cuida, especie y barrio. Lo que **no** tenemos es información de la vereda: no sabemos de qué material es, cuándo se hizo ni si pasan camiones. Ese faltante después se nota mucho.

---

## 04 · Exploración y limpieza 

> Antes de entrenar nada, pasamos bastante tiempo revisando los datos —el famoso EDA— y salieron cuatro cosas concretas que definieron la limpieza.
>
> Primero, había muchísimos campos vacíos: casi un **88%** sin protector y un **75%** sin cuidador vecinal. Al mirarlo con cuidado nos dimos cuenta de que no eran datos perdidos, sino que significaban literalmente *"ninguno"*. En vez de borrar filas, los pasamos a una categoría propia.
>
> Segundo, vimos que las clases están desbalanceadas: **71% de veredas sanas contra 29% dañadas**. Eso ya nos prendió una alarma sobre cómo íbamos a medir después.
>
> Tercero, limpiamos errores de carga evidentes, como diámetros en cero o árboles de más de 60 pulgadas. Y cuarto, resumimos nueve columnas sueltas de problemas en **tres puntajes simples** —raíces, tronco y ramas— y agrupamos las especies poco frecuentes para no llenar el modelo de ruido.

---

## 05 · Contrastando la sospecha 

> Antes de meter machine learning, quisimos ver si nuestra intuición básica aparecía a simple vista en los gráficos.
>
> A la izquierda está el porcentaje de veredas dañadas según el grosor del tronco. La línea roja punteada es el promedio general, casi 29%. En árboles chicos, el daño está en **15%**, y en los más grandes llega casi al **49%**.
>
> A la derecha pasa algo todavía más marcado con los conflictos en raíces, como piedras o rejillas apretando la base: sin conflictos, el daño es del **20%**; con un solo conflicto salta al **55%**, casi el triple, y con dos pasa el **62%**.
>
> Con eso confirmamos que había señal real en los datos y que tenía sentido seguir adelante.

---

## 06 · La trampa del 71%

> Acá vino una de las cosas más útiles que aprendimos en el curso: cómo medir bien un modelo.
>
> Antes de entrenar algo sofisticado, armamos un modelo tonto de referencia —un *Dummy*— que siempre responde *"esta vereda está sana"*. En el gráfico, en gris, se ve que ese modelo acierta el **71%** de las veces simplemente porque la mayoría de las veredas están sanas. Si mirabas solo el Accuracy parecía decente, pero en realidad encontró **cero** veredas rotas.
>
> Por eso elegimos otra métrica principal: el **F1 de la clase con daño**, que combina dos cosas: cuántas veredas rotas reales logra atrapar —el Recall— y cuántas de las alertas que levanta son ciertas —la Precisión. En violeta está nuestro modelo final: en Accuracy da casi igual que el Dummy, pero pasa de 0 a **0,548 en F1** y atrapa más del 61% del daño.
>
> Con esa regla puesta, pusimos a competir tres familias de modelos —Regresión Logística, Árbol de Decisión y Random Forest— usando validación cruzada sobre los 80 mil datos de entrenamiento.

---

## 07 · Modelo supervisado y explicabilidad 

> Al final nos quedamos con **Random Forest**, que básicamente entrena muchos árboles de decisión y decide por mayoría, lo que lo hace más estable.
>
> En el examen final con los 20 mil árboles que nunca había visto, de todas las veredas realmente rotas detectó **3.539** y se le escaparon **2.216**: agarra más o menos **6 de cada 10**. Y cuando levanta una alerta, acierta cerca de la mitad de las veces, contra casi 3 de cada 10 si salieras al azar. Además, la diferencia entre entrenamiento y test fue mínima, así que no memorizó.
>
> Después abrimos el modelo para ver en qué se fijaba. La técnica consiste en desordenar una columna por vez y medir cuánto se cae el rendimiento. El gráfico de la izquierda es clarísimo: lo que domina por lejos son los **conflictos en raíces**, seguido por el **diámetro del tronco**.
>
> Y ahí apareció un detalle re interesante con el diámetro: cuando miramos la curva de riesgo, sube fuerte entre 5 y 15 pulgadas, pero a partir de las **20 pulgadas se aplana en una meseta**. Tiene toda la lógica física: una vez que el árbol creció y levantó la baldosa, por más que siga engordando no la rompe dos veces. Ese quiebre es justamente lo que un modelo lineal no ve y la razón por la que elegimos Random Forest aunque la Regresión Logística quedó cerquita.

---

## 08 · Parte 2: Aprendizaje no supervisado 

> En la segunda parte del curso cambiamos completamente el enfoque: le sacamos al algoritmo la columna que dice si la vereda está rota o no, y le pedimos que agrupe 20 mil árboles solo por parecido usando seis variables numéricas y de estado.
>
> Con **K-Means** encontramos **cuatro perfiles** bastante claros: un grupo grande de jóvenes sanos sin cuidado vecinal (42%), un grupo de **árboles grandes y maduros** (23%) que tienen el triple de diámetro y el triple de conflictos de raíz, otro de jóvenes cuidados por vecinos (19%) y un último grupo de árboles con mala salud (16%).
>
> Recién al final volvimos a traer la etiqueta real de la vereda para ver qué había pasado adentro de cada grupo. Y ahí está el gráfico de la derecha: tres grupos rondan el 22–26% de daño, mientras que el de **árboles grandes y maduros salta al 46,3%**. Sin haber visto nunca la respuesta, el algoritmo aisló por su cuenta el mismo patrón que había encontrado el modelo supervisado.
>
> También probamos **PCA** —que mostró que las 6 variables casi no se pueden comprimir— y **DBSCAN**, que no sirvió para separar grupos grandes pero sí funcionó bien como detector de anomalías: marcó un **8%** de árboles raros, en general grandes, enfermos y con problemas por todos lados.

---

## 09 · De notebook a servicio 

> Para que el proyecto no quedara encerrado en una notebook, armamos el flujo completo en scripts y lo expusimos como una **API chiquita en FastAPI**. Le mandás los datos de un árbol y el pipeline ya entrenado te devuelve la clase estimada y una probabilidad.
>
> Ahí hay dos pruebas concretas sobre árboles de Queens en buena salud: uno de **22 pulgadas con piedras en la raíz** devuelve **76% de probabilidad de daño**; uno de **5 pulgadas sin conflictos** baja al **30%**.
>
> Y cada respuesta devuelve automáticamente una advertencia aclarando que está basado en el censo de 2015 y que no sirve para automatizar decisiones de obra.

---

## 10 · Obstáculos y límites reales 

> Varias cosas no salieron como uno fantasea al principio, y la verdad es que fueron de lo que más dejó el proyecto.
>
> La primera es que los tres modelos supervisados que probamos —uno simple y otros más complejos— clavaron todos un **F1 alrededor de 0,54–0,55**. Eso nos mostró el techo real de los datos: si no sabés de qué material es la vereda ni cuántos años tiene, ningún algoritmo va a hacer magia.
>
> La segunda es que en la parte no supervisada la separación entre grupos es moderada (Silhouette 0,33) y no replica la división con/sin daño; sirve para entender perfiles del arbolado y detectar casos raros, no para reemplazar al clasificador.
>
> Y la tercera es de cuidado con el uso: el modelo usa el **barrio**, y si en ciertos barrios históricamente se mantuvo menos la vereda, automatizar prioridades solo con eso podría reforzar desigualdades. Por eso en la tabla de la derecha resumimos qué sí hace y qué no: sirve para orientar inspecciones y aprender el método, nunca para decidir obras sin gente mirando en la calle.

---

## 11 · Cierre

> Para cerrar, si tuviera que resumir lo que me dejó hacer este proyecto de punta a punta en el curso, son tres cosas:
>
> Primero, que **plantear bien la pregunta y sus límites** ahorra un montón de vueltas después.
>
> Segundo, **desconfiar de la primera métrica linda** que aparece: un 71% de aciertos puede ser un modelo que no sirve para nada.
>
> Y tercero, **respetar el techo de los datos**: entender qué información te falta suele aportar mucho más que seguir complicando el algoritmo.
>
> Eso fue VeredaSegura. Si quieren ver alguna parte más en detalle o tienen alguna duda, charlamos.

---

## Glosario rápido 

- **Examen final (test):** 20.000 árboles guardados aparte desde el inicio para medir si el modelo generaliza con datos nuevos.
- **Dummy (línea base):** modelo ingenuo que siempre predice la clase mayoritaria; sirve como piso mínimo a superar.
- **Recall / Precisión / F1:** Recall = de las veredas rotas reales, qué porcentaje encontramos; Precisión = de las alertas que tiramos, cuántas eran ciertas; F1 = el balance entre las dos.
- **Random Forest:** ensamble de muchos árboles de decisión que votan en conjunto.
- **Importancia por permutación:** desordenar una columna a la vez en test y ver cuánto cae el F1; si cae mucho, esa variable pesaba mucho.
- **K-Means / Silhouette:** K-Means agrupa observaciones parecidas sin usar etiquetas; Silhouette mide qué tan bien separados quedaron los grupos (de -1 a 1).
- **DBSCAN:** agrupa por densidad y marca como ruido (`-1`) los casos aislados o atípicos.
