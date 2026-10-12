# Datos públicos: Daily Demand Forecasting Orders

`daily_demand.csv` es una copia local del conjunto **Daily Demand Forecasting Orders**, identificador **409** en UCI Machine Learning Repository. Permite ejecutar el proyecto sin descargar datos durante el análisis.

- Fuente: https://archive.ics.uci.edu/dataset/409/daily+demand+forecasting+orders
- CSV público consultado: https://archive.ics.uci.edu/static/public/409/data.csv
- DOI: https://doi.org/10.24432/C5BC8T
- Atribución: Ferreira, R., Martiniano, A., Ferreira, A., Ferreira, A., & Sassi, R. (2016). *Daily Demand Forecasting Orders* [Dataset]. UCI Machine Learning Repository.
- Licencia declarada por UCI: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, https://creativecommons.org/licenses/by/4.0/

## Contenido

La fuente describe datos de una empresa logística brasileña recopilados durante 60 días: **60 registros, 12 variables predictoras y un objetivo (`Total orders`)**, con 13 columnas en total. La copia incluida no contiene valores faltantes ni filas duplicadas completas. Estos conteos se comprueban en `notebooks/01_eda.ipynb`.

`Week of the month` usa códigos 1–5; `Day of the week` usa 2–6 para lunes–viernes. Los demás campos son cantidades, incluidas las columnas independientes para tipos A/B/C, sectores y grupos bancarios. La documentación no define con detalle esos tipos o grupos ni aclara completamente sus unidades.

## Procedencia y transformaciones

La copia del proyecto procede del archivo local de trabajo `work/uci409.csv`, obtenido previamente de la fuente pública. La única transformación previa es la serialización/exportación de los valores numéricos a CSV; los números de punto flotante pueden tener una representación textual distinta (por ejemplo, `223.270` se exporta como `223.27`). Se preservaron los valores numéricos, los nombres de columnas y el orden de las filas. Se incluyen separador coma, punto decimal y encabezado, sin índice adicional.

No se imputaron valores, no se eliminaron filas o columnas, no se normalizaron unidades, no se corrigió el sector fiscal y no se fabricaron fechas. Las agrupaciones y columnas auxiliares del EDA se calculan en memoria y no sobrescriben este archivo.

## Límites de interpretación

- **Unidades inciertas:** conservar la escala publicada; no asumir que los decimales representan fracciones físicas de pedidos ni convertirlos automáticamente a miles. La columna fiscal presenta discrepancias compatibles con una posible diferencia de escala; esa hipótesis requiere confirmación.
- **Ausencia de fechas:** no hay fecha completa, año ni mes. Se conserva el orden del archivo, pero su carácter cronológico no está verificado. Semana del mes y día de la semana no bastan para reconstruir fechas únicas.
- **Componentes del total:** en esta copia, `Order type A + Order type B + Order type C = Total orders` en las 60 filas, salvo precisión de punto flotante. Su uso para pronosticar demanda futura exige verificar si estaban disponibles en el instante de predicción; conocer los componentes realizados del mismo día puede introducir fuga de información.
- **Tamaño de la muestra:** las relaciones del EDA son descriptivas y no establecen causalidad ni garantizan estabilidad futura.
