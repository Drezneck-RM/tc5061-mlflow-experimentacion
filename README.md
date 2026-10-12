# Experimentación MLflow

Proyecto de TC5061 de **Saul Salgado Ortega**. Materia: **Operaciones de aprendizaje automático (Gpo 10)**. Período: **Trimestral Sep - Dic de 2026**. Actividad Individual | Código de experimentación base con MLFlow.

Predice la demanda de la siguiente observación de negocio del dataset [UCI 409](https://archive.ics.uci.edu/dataset/409/daily+demand+forecasting+orders), con 60 registros. Continúa la familia Random Forest utilizada en el notebook y añade información histórica disponible al emitir el pronóstico.

## Ejecutar en Windows / PowerShell

Desde la carpeta del proyecto, con Python 3.13 instalado:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train.py --help
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe scripts/run_comparison.py
.venv\Scripts\python.exe scripts/start_mlflow.py
```

El último comando mantiene MLflow disponible en **http://127.0.0.1:5001** hasta detenerlo con `Ctrl+C`. En otra terminal, ejecuta las pruebas con `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

## Ejecutar como MLflow Project

Después de instalar el entorno anterior, desde la raíz del repositorio:

```powershell
$projectPython = (Resolve-Path ".venv\Scripts\python.exe").Path
$env:PATH = (Split-Path $projectPython) + [IO.Path]::PathSeparator + $env:PATH
$env:MLFLOW_TRACKING_URI = 'sqlite:///' + ((Join-Path (Get-Location) 'mlruns.sqlite') -replace '\\', '/')
$env:MLFLOW_EXPERIMENT_NAME = 'TC5061_Experimentacion_CLI_Demanda'
& $projectPython -m mlflow run . --env-manager local -P n_estimators=100 -P max_depth=1 -P min_samples_leaf=2 -P seed=42
```

`MLproject` declara los parámetros y `python_env.yaml` referencia las versiones. `--env-manager local` reutiliza el entorno instalado; el paso de `PATH` asegura que el subproceso utilice su Python. El adaptador de `scripts/train_project.py` reutiliza la corrida creada por Projects en el mismo almacén y experimento. Esta interfaz tiene una ejecución de integración verificada y no modifica el protocolo de pronóstico.

## Qué se compara

`train.py` expone `--n-estimators` (100), `--max-depth` (3), `--min-samples-leaf` (2) y `--seed` (42). La comparación ejecuta seis configuraciones que cambian un hiperparámetro a la vez, más una repetición de la base. Cada ejecución registra parámetros, MAE, RMSE, otras métricas, predicciones, datos, código y un modelo cuya recarga se comprueba.

Se selecciona por MAE de **24 pronósticos progresivos, filas 25–48**. El historial se actualiza después de cada resultado; se compara contra la media acumulada Dummy. Las filas 49–60 ya fueron examinadas y solo se evalúan opcionalmente con `--evaluate-test`; no intervienen en la selección automática. Faltan fechas completas y unidades físicas confirmadas: los resultados no demuestran preparación para producción.

## Resultados y documentación

Ganó `depth_1` (100 árboles, profundidad 1, hoja mínima 2, semilla 42): **MAE 65.84**, frente a 74.39 de Dummy en validación. En el bloque final descriptivo obtuvo MAE 52.92 frente a 50.13 de Dummy; no confirmó una ventaja en MAE. La repetición de la base superó las seis comprobaciones de reproducibilidad.

- [Guía explicada](GUIA_EXPLICADA.md): comandos, metodología e interpretación de MLflow.
- [Revisión de buenas prácticas y ejemplos](docs/REVISION_BUENAS_PRACTICAS.md): correspondencia de estructura, validaciones y límites.
- [Informe PDF](docs/TC5061_Experimentacion_MLFlow_SalgadoSaul.pdf), [capturas](docs/evidence/) y [resultados portables](docs/results/).
- [Repositorio público en GitHub](https://github.com/Drezneck-RM/tc5061-mlflow-experimentacion).

`docs/results/` conserva evidencia versionable. La base local `mlruns.sqlite`, los modelos en `artifacts/` y las salidas en `results/` se regeneran al ejecutar. Repetir un comando crea nuevas ejecuciones; no sustituye las anteriores.
