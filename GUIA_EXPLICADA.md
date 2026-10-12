# Guía del ejercicio: pronóstico, experimentación y MLflow

Esta guía explica las decisiones del proyecto y cómo comprobarlas. El objetivo es poder responder qué se entrenó, con qué datos, cómo se evaluó y si otra ejecución reproduce sus resultados.

## 1. Qué problema estamos resolviendo

La variable objetivo es `Total orders`: demanda total de una observación diaria del dataset [Daily Demand Forecasting Orders de UCI](https://archive.ics.uci.edu/dataset/409/daily+demand+forecasting+orders). La copia local conserva 60 registros y el orden original.

El horizonte acordado es **la siguiente observación de negocio**, después de conocer la demanda de la observación anterior. No afirmamos que siempre sea el siguiente día calendario: faltan fechas completas y hay saltos en la secuencia de códigos de día. Suponemos que el día de la semana de la observación que vamos a pronosticar se conoce al emitir el pronóstico.

Esta definición decide qué información podemos utilizar. No se usan cantidades del mismo día que estamos prediciendo. En el análisis anterior comprobamos que los pedidos A, B y C suman la demanda total; incluirlos contemporáneamente permitiría reconstruir la respuesta sin demostrar anticipación. Si la empresa conociera anticipadamente alguna cantidad, habría que documentar su disponibilidad antes de incorporarla.

## 2. Por qué usamos Random Forest

Random Forest ya se entrenó en el notebook anterior. Conservamos esa familia para estudiar tres hiperparámetros sustantivos mediante una interfaz de línea de comandos. Aquí cambian las variables y el procedimiento de evaluación: usamos información histórica y reajustamos el modelo para cada pronóstico.

Un bosque promedia las predicciones de varios árboles. Cada árbol aprende reglas de separación a partir de sus ejemplos de entrenamiento. La documentación oficial define el estimador y sus hiperparámetros: [RandomForestRegressor de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). Esa flexibilidad puede ayudar, pero con tan pocos datos también puede aprender variaciones que no se repiten. El proyecto conserva una referencia Dummy para comprobar si esa complejidad aporta algo.

Un **parámetro aprendido** es una separación concreta que el árbol obtiene de los datos. Un **hiperparámetro** es una configuración que elegimos antes de entrenar, como la profundidad máxima. Ajustar hiperparámetros significa comparar esas configuraciones bajo el mismo procedimiento de evaluación.

## 3. Preparar el entorno y ejecutar

Abre PowerShell dentro de la carpeta del proyecto. Los comandos siguientes usan explícitamente el Python del entorno virtual; no es necesario activar el entorno ni cambiar la política de ejecución de PowerShell.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` fija las versiones de las dependencias directas. Conservarlo reduce cambios involuntarios entre ejecuciones. La semilla, las versiones y los datos también influyen en la repetibilidad; ninguna de esas piezas sustituye a las demás.

Para consultar la interfaz y ejecutar la configuración predeterminada:

```powershell
.venv\Scripts\python.exe train.py --help
.venv\Scripts\python.exe train.py
```

El programa imprime el identificador de ejecución, parámetros, métricas, URI del modelo y carpeta de resultados. También registra esa información en MLflow. Los valores por defecto permiten ejecutar el ejercicio sin editar Python.

Una configuración explícita equivalente a la base es:

```powershell
.venv\Scripts\python.exe train.py --n-estimators 100 --max-depth 3 --min-samples-leaf 2 --seed 42 --run-name base_manual
```

Cada llamada crea una ejecución nueva. Si proporcionas `--output-dir`, utiliza una carpeta nueva para conservar los resultados anteriores.

| Argumento | Valor predeterminado | Qué controla |
|---|---:|---|
| `--n-estimators` | 100 | Número de árboles. Aumentarlo implica entrenar más árboles; no crea más observaciones. |
| `--max-depth` | 3 | Profundidad máxima de cada árbol. Una profundidad mayor permite reglas más específicas. |
| `--min-samples-leaf` | 2 | Mínimo de ejemplos por hoja. Un mínimo mayor evita hojas sustentadas por muy pocos casos. |
| `--seed` | 42 | Semilla de Python, NumPy y el bosque. Permite repetir la aleatoriedad en el entorno evaluado. |

También puedes indicar `--data-path`, `--tracking-uri`, `--experiment-name`, `--run-name` y `--output-dir`. `--help` muestra sus valores y propósito. Los tres hiperparámetros principales deben ser enteros positivos; la semilla debe estar entre 0 y `2**32 - 1`.

## 4. Qué significan las cuatro variables de entrada

| Variable | Información para pronosticar la fila t |
|---|---|
| `Day of the week` | Código del día objetivo, supuesto conocido anticipadamente. |
| `lag_1` | Demanda de la observación inmediatamente anterior. |
| `mean_past_5` | Promedio de las cinco demandas observadas anteriormente. |
| `last_same_weekday` | Demanda de la observación anterior más reciente con el mismo código de día. |

La última variable no equivale necesariamente a un retraso fijo de cinco filas. Si faltan días, buscamos el último código coincidente. Si todavía no existe uno, el código utiliza la observación anterior como respaldo.

La expresión `shift(1)` desplaza los valores para que la demanda actual no entre en sus propias variables. El promedio móvil se calcula después de ese desplazamiento. Para el historial del mismo día se usa un desplazamiento dentro de cada grupo de códigos.

Las primeras cinco filas sirven de **historial inicial o warmup**. No se eliminan del archivo: aportan información para formar variables posteriores. El bosque empieza a tener ejemplos completos en la fila 6. El código de día se convierte en indicadores mediante One-Hot Encoding dentro del pipeline; las tres cantidades históricas se pasan directamente al bosque.

## 5. Cómo se evalúa sin utilizar el resultado futuro

La comparación calcula 24 pronósticos sobre las filas 25–48:

| Pronóstico | Demandas disponibles | Ejemplos completos utilizados por el bosque |
|---|---|---|
| Fila 25 | Filas 1–24 | Filas 6–24: 19 ejemplos |
| Fila 26 | Filas 1–25 | Filas 6–25: 20 ejemplos |
| … | Se incorpora cada resultado después de pronosticarlo | El entrenamiento aumenta progresivamente |
| Fila 48 | Filas 1–47 | Filas 6–47: 42 ejemplos |

Por tanto, utilizar la demanda real de la fila 25 para pronosticar la 26 es válido bajo el horizonte acordado. Utilizarla para pronosticar la propia fila 25 sería una fuga de información.

El Dummy de referencia predice la media de **todos** los resultados disponibles hasta ese momento. Puede aprovechar las primeras cinco demandas, aunque el bosque las necesite para iniciar sus variables. La comparación es justa porque ambos métodos conocen el mismo historial y se evalúan en las mismas fechas de observación.

Los hiperparámetros de una configuración permanecen fijos a lo largo de esos 24 pronósticos. Los árboles aprendidos se vuelven a ajustar en cada origen. Este proyecto compara configuraciones sobre los mismos orígenes de desarrollo; no debe confundirse con el ajuste interno y externo de Ridge realizado en otra etapa del notebook.

## 6. Ejecutar la comparación de hiperparámetros

```powershell
.venv\Scripts\python.exe scripts/run_comparison.py
```

El diseño se guarda antes de ejecutar las configuraciones:

| Configuración | Árboles | Profundidad | Mínimo por hoja | Semilla |
|---|---:|---:|---:|---:|
| `base` | 100 | 3 | 2 | 42 |
| `trees_300` | 300 | 3 | 2 | 42 |
| `depth_1` | 100 | 1 | 2 | 42 |
| `depth_6` | 100 | 6 | 2 | 42 |
| `leaf_1` | 100 | 3 | 1 | 42 |
| `leaf_4` | 100 | 3 | 4 | 42 |
| `base_repeat` | 100 | 3 | 2 | 42 |

Las primeras seis son alternativas distintas. Cada alternativa cambia un solo hiperparámetro respecto a la base. La séptima repite exactamente la base para comprobar resultados; se excluye de la selección.

Este diseño se conoce como **un factor a la vez**. Ayuda a observar qué sucede al cambiar un ajuste manteniendo los demás constantes. No explora todas las combinaciones ni permite concluir cuáles serían las mejores interacciones entre profundidad y tamaño de hoja.

La configuración elegida es la de menor `validation_mae`. En un empate exacto se conserva el orden de la lista. El archivo `paired_sensitivity.csv` muestra la diferencia de cada alternativa respecto a la base: una diferencia de MAE negativa indica menos error; una positiva, más error.

La selección usa esos mismos resultados de desarrollo. Su mejor puntuación es un dato para elegir entre configuraciones, no una estimación independiente e imparcial del desempeño futuro.

## 7. Leer las métricas

| Métrica | Interpretación |
|---|---|
| `validation_mae` | Promedio de los errores absolutos. Menor es mejor. |
| `validation_rmse` | Raíz del error cuadrático medio; penaliza más los errores grandes. Menor es mejor. |
| `validation_r2` | Comparación del error cuadrático con predecir la media del bloque evaluado. Puede ser negativo. |
| `validation_mean_signed_error` | Promedio de predicción menos realidad. Positivo: sobreestimación; negativo: subestimación. |
| `validation_dummy_mae` | MAE de la media acumulada en los mismos 24 pronósticos. |
| `validation_skill_vs_dummy` | `1 − MAE_modelo / MAE_Dummy`. Positivo: mejora relativa; negativo: empeora. |
| `validation_n_predictions` | Cantidad de pronósticos evaluados; aquí debe ser 24. |

**MAE y RMSE son cantidades en las unidades originales de la demanda, no porcentajes ni exactitud.** Un MAE de 60 significaría un error absoluto promedio de 60 unidades, no 60% de error. No se ha confirmado la escala física de esas unidades.

Por ejemplo, un `skill_vs_dummy` de 0.05 significaría reducir el MAE un 5% frente al Dummy en ese bloque. No significa 95% de exactitud. Para decidir si el error es aceptable operativamente, se necesitan costos de sobreestimar y subestimar la demanda.

## 8. Qué es cada elemento de MLflow

MLflow Tracking permite registrar ejecuciones y después consultar sus parámetros, métricas y archivos. Los conceptos siguientes se basan en la [documentación oficial de MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/), aplicados a este proyecto.

| Elemento | En este proyecto |
|---|---|
| **Experimento** | Agrupación `TC5061_Experimentacion_CLI_Demanda` que reúne ejecuciones comparables. |
| **Run o ejecución** | Una llamada a `train.py`, con un identificador propio. No equivale a un único ajuste: contiene los reajustes progresivos de su evaluación. |
| **Parámetros** | Configuración utilizada: número de árboles, profundidad, hoja mínima, semilla, protocolo y huellas de archivos. |
| **Métricas** | Resultados numéricos como MAE y RMSE. |
| **Artefactos** | Archivos: CSV de predicciones, gráfica, JSON de resultados, datos y fuentes de entrenamiento. |
| **Modelo guardado** | Pipeline ajustado al terminar la fila 48, con firma de entrada, ejemplo y dependencias. |

La base SQLite guarda el seguimiento y las referencias a artefactos. Los archivos de modelos y resultados se almacenan aparte. Por eso copiar solo `mlruns.sqlite` no basta para transportar todos los artefactos a otra computadora. El proyecto permite regenerarlos al ejecutar el código.

La entrega separa dos tipos de archivos:

| Ubicación | Qué contiene y cómo se utiliza |
|---|---|
| [`docs/results/`](docs/results/) | Copias portables de tablas, JSON, predicciones y comprobaciones de la entrega. Se pueden revisar en GitHub y versionar junto con el código, sin iniciar MLflow. |
| `mlruns.sqlite`, `artifacts/`, `results/` | Seguimiento, modelos y salidas de ejecución locales. Se generan al entrenar; no se necesita publicar la base ni las carpetas de modelos para revisar las tablas entregadas. |

Los identificadores archivados en `docs/results/` describen la ejecución original. Una ejecución en otra computadora crea identificadores nuevos: utiliza su propio `summary.json` para recuperar el modelo de esa nueva base local. Los JSON portables no contienen por sí mismos los modelos entrenados.

## 9. Abrir MLflow y comparar las ejecuciones

```powershell
.venv\Scripts\python.exe scripts/start_mlflow.py
```

El servidor local se consulta en **http://127.0.0.1:5001**. Mantén abierta esa terminal y detén el servidor con `Ctrl+C` al terminar. El proyecto utiliza el puerto 5001 para separar esta interfaz de la sesión anterior del notebook, que utilizaba el 5000.

Abre el experimento `TC5061_Experimentacion_CLI_Demanda`, entra en **Runs** y pega este filtro en el campo de búsqueda para mostrar las seis alternativas de la entrega:

```text
tags.comparison_group = 'entrega-final' AND tags.is_reproducibility_repeat = 'false'
```

La etiqueta `comparison_group` toma el nombre de la carpeta de comparación. Para tu propia reproducción, sustituye `entrega-final` por el nombre de la carpeta que utilizaste. Las ejecuciones auxiliares y la evaluación descriptiva no forman parte de estas seis alternativas.

1. En **Columns**, muestra `validation_mae`, `validation_rmse`, `max_depth`, `min_samples_leaf`, `n_estimators` y `seed`. Ordena `validation_mae` de menor a mayor. La [captura de la tabla](docs/evidence/comparison_table.png) muestra esta vista.
2. Marca las casillas de `base`, `trees_300`, `depth_1`, `depth_6`, `leaf_1` y `leaf_4`, y pulsa **Compare**. La página resultante indica **Comparing 6 Runs from 1 Experiment**.
3. En **Visualizations → Parallel Coordinates Plot**, selecciona los tres hiperparámetros en **Parameters** y `validation_mae` en **Metrics**. Cada línea representa una configuración, como muestra la [captura de comparación](docs/evidence/comparison_view.png).
4. Revisa también las tablas de parámetros y métricas. Todas las alternativas deben compartir semilla y hashes de datos y protocolo. Consulta MAE, RMSE, referencia Dummy y sesgo; una mejora pequeña no demuestra superioridad general.
5. Regresa a **Runs**, abre una ejecución y entra en la pestaña **Artifacts**. Expande `results` y selecciona `validation_predictions.csv`, la gráfica o los JSON. La [captura de artefactos](docs/evidence/run_artifacts.png) muestra las carpetas de la configuración elegida; `comparison` conserva además los resultados de la comparación.
6. Busca el modelo guardado en el detalle de la ejecución o la sección **Models**. Guardarlo no equivale a desplegarlo ni aprobarlo para producción.

Para investigar reproducibilidad, elimina la condición `tags.is_reproducibility_repeat = 'false'` del filtro y compara `base` con `base_repeat`. Sus identificadores y horas deben ser diferentes, pero sus parámetros, hashes y resultados deben coincidir dentro de la tolerancia verificada por el código.

## 10. Qué demuestra la reproducibilidad

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas cubren que modificar resultados actuales o futuros no altere un pronóstico anterior, que el historial del mismo día funcione con saltos, que el ajuste reciba solo su prefijo, que la actualización diaria sea causal, que las métricas tengan el signo correcto y que la misma semilla reproduzca las predicciones.

La comparación completa añade una comprobación de extremo a extremo entre `base` y `base_repeat`. Compara parámetros, hashes, orígenes, predicciones y métricas con tolerancias relativas y absolutas de `1e-12`. Guarda las comprobaciones en `reproducibility.json`; si fallan, la comparación se interrumpe.

Una huella **SHA-256** es un resumen del contenido de un archivo. El proyecto guarda huellas del CSV, del protocolo y de las fuentes de entrenamiento, incluido `requirements.txt`. Dos archivos con el mismo nombre podrían tener contenido diferente; las huellas ayudan a detectar ese cambio. No prueban que los datos sean correctos, solo permiten identificar la versión utilizada.

Esta evidencia demuestra repetibilidad en el entorno probado. No garantiza resultados idénticos con cualquier versión, sistema operativo o cambio de datos. Las dependencias del modelo también quedan registradas para facilitar su recuperación.

## 11. Por qué el modelo guardado no representa todos los reajustes

Al terminar la evaluación se ajusta un pipeline usando el historial hasta la fila 48. Tiene 43 ejemplos completos, de las filas 6–48. Ese estado se guarda en MLflow y se recarga para comprobar que produce la misma predicción para las variables disponibles de la fila 49, sin consultar su demanda real.

Las métricas de validación corresponden a 24 estados anteriores del modelo, cada uno entrenado con su propio prefijo. Un pipeline congelado después de la fila 48 no debe utilizarse para recalcular retrospectivamente esos errores: habría aprendido respuestas que entonces eran futuras.

Para reproducir toda la evaluación debes ejecutar el ciclo de pronósticos con sus datos y protocolo. Para recuperar una única predicción del estado guardado, puedes usar el `model_uri` del `summary.json` de una ejecución. Este ejemplo se ejecuta desde la raíz del proyecto y requiere sustituir la ruta del JSON:

```python
import json
import sys
from pathlib import Path
import mlflow

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
from demand_forecasting import build_features, load_data

summary_path = root / "results" / "RUTA_DE_TU_EJECUCION" / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
mlflow.set_tracking_uri("sqlite:///" + (root / "mlruns.sqlite").as_posix())
model = mlflow.pyfunc.load_model(summary["model_uri"])
raw = load_data(root / "data" / "daily_demand.csv")
x_next = build_features(raw).iloc[[48]].reset_index(drop=True)
print(model.predict(x_next))
```

En una comparación, la ruta de resultados incluye la carpeta de comparación y el nombre de la configuración. El modelo recibe las cuatro variables históricas ya construidas; no recibe directamente las trece columnas del archivo original.

## 12. Evaluación opcional de las últimas observaciones

Las filas 49–60 se observaron en el ejercicio anterior. El comando predeterminado no las evalúa para comparar configuraciones.

Si deseas una comparación descriptiva posterior a la selección:

```powershell
.venv\Scripts\python.exe scripts/run_comparison.py --evaluate-best-test
```

Este comando vuelve a ejecutar la comparación completa, guarda primero la selección y añade una ejecución de la configuración elegida con las métricas descriptivas. No cambia el ganador en función de ese resultado. También existe `train.py --evaluate-test` para una configuración explícita, pero revisar repetidamente ese bloque para ajustar parámetros lo convertiría en otro conjunto de desarrollo.

Ni siquiera un resultado favorable en ese bloque sería una validación nueva e independiente. Harían falta observaciones futuras que no se hayan usado para decidir variables, modelos o ajustes.

## 13. Resultados verificados de esta entrega

La comparación entregada se ejecutó con el comando siguiente, expresado desde la raíz del proyecto y usando el Python del entorno instalado:

```powershell
python scripts/run_comparison.py --output-dir results/entrega-final --evaluate-best-test
```

Si utilizas el entorno virtual de esta guía, sustituye `python` por `.venv\Scripts\python.exe`. La carpeta `results/entrega-final` conserva la ejecución documentada. Para una repetición tuya, utiliza otra carpeta vacía, por ejemplo `--output-dir results/mi-reproduccion`, porque el programa protege los resultados existentes.

| Configuración | MAE de validación ↓ | RMSE de validación ↓ | Diferencia de MAE respecto a base |
|---|---:|---:|---:|
| `base` | 77.13 | 106.18 | 0.00 |
| `trees_300` | 74.20 | 102.96 | −2.93 |
| **`depth_1`** | **65.84** | **93.91** | **−11.29** |
| `depth_6` | 77.27 | 106.02 | +0.14 |
| `leaf_1` | 86.34 | 115.46 | +9.21 |
| `leaf_4` | 71.34 | 101.13 | −5.79 |
| `base_repeat` | 77.13 | 106.18 | 0.00; control excluido de selección |

La referencia Dummy tiene **MAE 74.39** en los mismos 24 pronósticos. La configuración elegida reduce ese MAE aproximadamente **11.49%**. Su R² es 0.1561 y su sesgo medio es −1.89 unidades. Es la mejor configuración según el criterio fijado dentro de este diseño; no demuestra que sea la mejor posible para el problema.

Entre los cambios concretos probados, reducir la profundidad de 3 a 1 produjo la mayor variación absoluta del MAE: **11.29 unidades de mejora**. Reducir la hoja mínima de 2 a 1 empeoró el MAE en 9.21 unidades, mientras aumentarla a 4 lo mejoró en 5.79. Pasar de 100 a 300 árboles mejoró 2.93 unidades. Estas comparaciones se realizan respecto a la misma base, manteniendo las demás opciones constantes.

Por eso, en este conjunto pequeño la mayor flexibilidad no fue ventajosa: los árboles de profundidad 1 funcionaron mejor que los de profundidad 3 o 6 evaluados. Es compatible con la utilidad de limitar la complejidad, pero no prueba por sí solo la causa de los errores ni una superioridad general de los árboles más simples. Tampoco permite ordenar universalmente la importancia de los hiperparámetros: los valores y tamaños de cambio fueron diferentes.

La selección quedó guardada antes de ejecutar el bloque final descriptivo:

| Evaluación descriptiva de las filas 49–60 | Resultado |
|---|---:|
| MAE de `depth_1` | 52.92 |
| MAE de Dummy | 50.13 |
| RMSE de `depth_1` | 62.32 |
| R² de `depth_1` | −0.0699 |
| Sesgo medio de `depth_1` | +13.80 |

El MAE del bosque resultó **5.57% mayor** que el de Dummy en ese bloque. Por tanto, la ventaja de validación no se confirmó bajo el criterio de MAE. No cambiamos el ganador de la búsqueda después de observar ese resultado: diferenciamos la selección en desarrollo de la evidencia descriptiva posterior. Las doce observaciones ya habían sido examinadas y no constituyen una prueba independiente nueva.

La repetición pasó las **seis comprobaciones**: parámetros, hashes, nombres de métricas, valores de métricas, orígenes y predicciones. La tolerancia numérica fue `rtol=1e-12` y `atol=1e-12`. Los identificadores permiten localizar la evidencia:

| Evidencia | Identificador de ejecución |
|---|---|
| Base | `4999674e00ae47a6963a642791943356` |
| Repetición de la base | `b41c4aef93a74e4c8f88a655ce454569` |
| Seleccionada por validación | `613f6c1c9395459391b7a27914d4de64` |
| Evaluación descriptiva de la seleccionada | `7fa0f43edf3b4256bfab4ca9b40d6bb1` |

Así se completaron **ocho ejecuciones** para la entrega: seis alternativas, una repetición y una evaluación descriptiva adicional. Las copias portables de [comparison.csv](docs/results/comparison.csv), [paired_sensitivity.csv](docs/results/paired_sensitivity.csv), [reproducibility.json](docs/results/reproducibility.json) y [selected_descriptive_test.json](docs/results/selected_descriptive_test.json) permiten comprobar estas cifras con mayor precisión sin iniciar MLflow. Al volver a ejecutar el proyecto se generan identificadores nuevos.

Los resultados anteriores de Ridge o del Random Forest con solo calendario pertenecen a otros experimentos. No deben presentarse como resultados de este proyecto, que tiene variables y un procedimiento distintos.

## 14. Cómo se relaciona el proyecto con la entrega

| Requisito | Implementación o evidencia que se debe revisar |
|---|---|
| Entrenamiento mediante funciones y CLI | `train.py` y `src/demand_forecasting/forecasting.py`. |
| Tres hiperparámetros, semilla, valores por defecto y ayuda | Argumentos de `train.py`; consultar `--help`. |
| Registro de métricas y modelo | Ejecuciones MLflow, MAE/RMSE, modelo y `reload_verification.json`. |
| Seis configuraciones comparables | `scripts/run_comparison.py`, `frozen_design.json` y `comparison.csv`. |
| Reproducibilidad | Semilla, versiones, hashes, `base_repeat`, `reproducibility.json` y pruebas. |
| Explicación del efecto de los cambios | `paired_sensitivity.csv` y análisis de resultados. |
| Instrucciones breves de ejecución | `README.md`. |
| Informe y capturas | [Informe PDF](docs/TC5061_Experimentacion_MLFlow_SalgadoSaul.pdf), [tabla](docs/evidence/comparison_table.png), [comparación](docs/evidence/comparison_view.png) y [artefactos](docs/evidence/run_artifacts.png). |
| Repositorio público | [Drezneck-RM/tc5061-mlflow-experimentacion](https://github.com/Drezneck-RM/tc5061-mlflow-experimentacion). |

## 15. Límites y próximos pasos razonables

Este ejercicio permite comparar configuraciones bajo condiciones explícitas y revisar su trazabilidad. Continúan pendientes las fechas completas, la escala física de la demanda, la disponibilidad operativa de otros predictores y los costos de error del negocio.

Con 60 registros, una diferencia pequeña puede depender de unas pocas observaciones. Más árboles, más combinaciones o más ejecuciones no sustituyen más información independiente. Antes de desplegar un modelo, convendría obtener historia adicional verificable, evaluar el procedimiento en observaciones nuevas y definir criterios de aceptación y monitoreo. Un resultado que no supera al Dummy también es una conclusión útil cuando está correctamente medido y documentado.
