# Sistema de Clasificación con Regresión Logística

Este repositorio contiene un sistema de machine learning diseñado por `Juan Andres Escobar` y `Mariana Celis` para tareas de clasificación utilizando Regresión Logística. El sistema está implementado en Python e incluye scripts para el procesamiento de datos, entrenamiento y evaluación de modelos, y realización de predicciones sobre nuevos datos. Está estructurado para ser reutilizable y adaptable a tareas de clasificación similares.

## Contenido

- **`system.py`:** Contiene la clase principal `ML_System_Classification`, que gestiona el flujo completo de trabajo: carga de datos, procesamiento de características, entrenamiento y evaluación de modelos, y predicciones sobre nuevos datos.
- **`test_system.py`:** Contiene pruebas unitarias para validar la funcionalidad de `ML_System_Classification`. Las pruebas aseguran que el sistema funcione correctamente y que el modelo alcance una precisión de al menos el 70%.
- **`system.ipynb`:** Una versión en Jupyter Notebook de `system.py`, que permite ejecutar y modificar el código de manera interactiva, facilitando la experimentación y visualización de resultados.

## Características

- **Procesamiento de Datos:** 
  - Maneja la carga y preprocesamiento de conjuntos de datos, incluyendo la escala de características y el manejo de variables categóricas.
  
- **Entrenamiento y Evaluación del Modelo:**
  - Utiliza `LogisticRegression` y `GridSearchCV` para entrenar el modelo con ajuste de hiperparámetros.
  - Evalúa el rendimiento del modelo utilizando métricas de precisión.

- **Flujo de Trabajo Robusto:**
  - Implementa un flujo de trabajo completo de ML con manejo de errores.
  - Soporta fácil reentrenamiento y validación.

- **Pruebas Unitarias:**
  - Garantiza la fiabilidad del sistema mediante pruebas automatizadas.
  - Verifica que el modelo cumpla con los criterios de rendimiento.

## Codigo

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/tu-repo.git
   cd tu-repo

1. **Ejecuta las pruebas:**
   ```bash
   python test_system.py
