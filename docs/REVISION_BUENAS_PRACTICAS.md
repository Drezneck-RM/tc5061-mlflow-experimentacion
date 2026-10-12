# Revisión de estructura y buenas prácticas de experimentación

Se contrastó el repositorio con los ejemplos académicos `ITESM_MNA_2026_MLFlow_Projects.zip` y `Refactored_notebooks.zip`. La revisión se limita a experimentación reproducible para UCI 409; no certifica preparación para producción.

## Correspondencia con los ejemplos

| Elemento de los ejemplos | Implementación del proyecto | Decisión |
|---|---|---|
| `MLproject` y `python_env.yaml` | Ambos archivos en la raíz; parámetros tipados y valores por defecto | Se añadió la interfaz MLflow Projects, reutilizando el entrenamiento existente. |
| `wine/modeling/train.py` y `wine/workflow.py` | `src/demand_forecasting/forecasting.py`, `train.py` y `scripts/run_comparison.py` | Las responsabilidades ya están separadas; no es necesario copiar nombres del problema de vino. |
| Notebook de exploración y código refactorizado | `notebooks/01_eda.ipynb` para EDA; funciones en `src/` | El entrenamiento no depende del orden de ejecución de celdas. |
| Datos, dependencias y pruebas | `data/`, `requirements.txt`, `tests/` | Se conserva un CSV original pequeño con procedencia; no se duplican datos para crear carpetas vacías. |
| Documentación de ejecución | `README.md`, `GUIA_EXPLICADA.md`, informe y evidencias en `docs/` | Los comandos y resultados se pueden revisar desde GitHub. |

Estructura principal:

```text
tc5061-mlflow-experimentacion/
  MLproject
  python_env.yaml
  requirements.txt
  train.py
  data/
  notebooks/
  src/demand_forecasting/
  scripts/
    run_comparison.py
    start_mlflow.py
    train_project.py
  tests/
  docs/
    evidence/
    results/
  README.md
  GUIA_EXPLICADA.md
```

`MLproject` describe una corrida individual; la comparación de varias configuraciones continúa en su script específico. El adaptador resuelve el almacén y el experimento de la corrida creada por Projects y la reanuda. Esto evita crear otra corrida o intentar registrar parámetros de varios modelos en un único identificador. Los archivos de la interfaz quedan registrados como artefactos en `project_source/`.

## Prácticas verificadas

| Práctica | Evidencia y alcance |
|---|---|
| Evitar información futura | Variables desplazadas y ajuste con filas anteriores al origen. La suite prueba que cambiar resultados actuales o futuros no altera un pronóstico pasado. |
| Preprocesamiento dentro del entrenamiento | `Pipeline` y `ColumnTransformer` se ajustan sobre cada prefijo de entrenamiento. |
| Referencia simple | Dummy usa la media de todas las demandas conocidas; comparte los orígenes de evaluación con el bosque. |
| Comparación consistente | Seis configuraciones, misma semilla, datos, protocolo y 24 pronósticos; selección por MAE. |
| Repetibilidad | Dependencias directas fijadas, semilla, hashes y repetición base con tolerancia numérica definida. |
| Trazabilidad | Parámetros, métricas, datos, fuentes, predicciones, entorno y modelos en MLflow. |
| Recuperación del modelo | Recarga del pipeline y comprobación de su predicción. |
| Validación del nuevo punto de entrada | 10 pruebas: 6 del pronóstico y 4 de integración/configuración de Projects; además una ejecución real con 13 comprobaciones. |
| Separación de archivos públicos y locales | `.gitignore` excluye entornos, base de MLflow, modelos generados y credenciales; `docs/results/` conserva evidencia portable. |

La ejecución de integración `aea512bc032c4681a867c3a3ee3328ea` obtuvo MAE **77.12967000405226**, idéntico al de la base de la entrega. Conservó las 24 predicciones y las huellas del código central, los datos y el protocolo. Se comprobó que Projects y el entrenamiento usaran el mismo identificador de corrida. La evidencia está en [mlproject_verification.json](results/mlproject_verification.json).

La prueba de Projects es adicional a las ocho ejecuciones de la entrega original. Un intento inicial falló al resolver el intérprete en el entorno de ejecución y se conservó como fallido; la repetición con el entorno correcto terminó satisfactoriamente. El README fija `PATH` explícitamente porque `--env-manager local` invoca `python` sin crear un entorno aislado. El modo de creación automática de entornos no forma parte de esta verificación.

## Adaptaciones necesarias para este dataset

Los ejemplos de clasificación de vino utilizan particiones aleatorias. Aquí se conserva el orden del archivo y se evalúa un registro adelante, porque barajar filas puede permitir aprender del futuro. El orden cronológico real continúa sin poder confirmarse al faltar fechas completas.

En `Wine_Refactored_V1.ipynb`, el escalado se ajusta antes de dividir los datos; esa secuencia puede contaminar la evaluación. La versión V2 incorpora el preprocesamiento a un pipeline. Nuestro código aplica el ajuste dentro de cada prefijo de entrenamiento. Se toma de los ejemplos la separación de responsabilidades y el formato de Projects, sin trasladar decisiones que no corresponden al pronóstico.

No se añadió un paquete instalable con `pyproject.toml`, más capas de clases ni duplicados `data/raw` y `data/processed`: la entrada CLI y el módulo actual bastan para este ejercicio pequeño. Pueden incorporarse si el proyecto crece y requiere distribución como biblioteca.

## Límites que permanecen

Con 60 registros, selección sobre una ventana compartida, fechas incompletas y un bloque final ya examinado, no existe una estimación nueva e independiente de generalización. El bosque ganador en validación no supera a Dummy en MAE en el bloque descriptivo. La estructura y MLflow no corrigen esas restricciones estadísticas.

Las versiones directas están fijadas; no se incluye un archivo de bloqueo de todas las dependencias transitivas ni se ha validado cada sistema operativo. Para producción se necesitarían datos futuros, definición de unidades y costos de error, controles de calidad continuos, criterios de aceptación y monitoreo. Estos puntos son trabajo futuro, no capacidades implementadas en esta entrega.

## Fuentes técnicas

- [MLflow Projects](https://mlflow.org/docs/latest/ml/projects/): descriptor del proyecto, parámetros y entornos.
- [Buenas prácticas y errores comunes de scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html): prevención de fuga de información y uso de pipelines.
