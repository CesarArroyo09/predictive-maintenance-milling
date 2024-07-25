# Predictive Maintenance Dataset (AI4I 2020)
Este repositorio estudia diferentes técnicas de clasificación para mantenimiento predictivo para el dataset Predictive Maintenance Dataset (AI4I 2020)

# Orden de los notebooks y scripts
Los notebooks y scripts fueron desarrollados en el siguiente orden:
1. `EDA.ipynb`: Análisis exploratorio de los datos.
2. `scripts/profile_report.py`: Sirve para generar un reporte de perfilamiento de los datos con `ydata-profiling`.
3. `preprocessing.ipynb`: Preprocesamiento de los datos. Sirve para determinar qué comportamiento necesitamos para el pipeline.
4. `mltools/preprocessing.py`: Este módulo encapsula la funcionalidad de preprocesamiento usando una clase.
5. `models.ipynb`: Exploramos varios modelos de clasificación.
6. `mltools/models.py`: Este módulo encapsula la funcionalidad de los modelos de clasificación usando una clase.
