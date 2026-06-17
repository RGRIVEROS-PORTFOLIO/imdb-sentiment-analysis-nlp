# Clasificación de Sentimiento en Reseñas de Películas

## Red Neuronal TF-IDF vs DistilBERT

## Descripción General

Este proyecto implementa un pipeline completo de **Procesamiento de Lenguaje Natural (NLP)** y **Deep Learning** para la clasificación automática de sentimiento en reseñas de películas utilizando el dataset IMDb.

El objetivo consiste en clasificar cada reseña como **Positiva** o **Negativa**, aplicando técnicas de limpieza textual, vectorización mediante TF-IDF, modelado con redes neuronales en PyTorch y comparación contra un modelo Transformer preentrenado (DistilBERT).

El trabajo fue desarrollado como proyecto final de la asignatura **Data Science III: NLP & Deep Learning** de Coderhouse, con foco tanto en la implementación práctica como en la justificación metodológica de cada decisión tomada.

---

# Objetivos

* Construir un pipeline completo de NLP de punta a punta.
* Analizar y preprocesar texto utilizando técnicas modernas.
* Comparar herramientas de lematización (NLTK y spaCy).
* Representar texto mediante TF-IDF.
* Desarrollar un clasificador de sentimiento utilizando PyTorch.
* Aplicar técnicas de regularización para controlar el sobreajuste.
* Comparar el desempeño de un modelo propio frente a DistilBERT.
* Analizar fortalezas, limitaciones y posibles mejoras futuras.

---

# Dataset

## IMDb Movie Reviews Dataset

**Fuente:**

* Hugging Face Datasets
* Dataset: IMDb

### Características

| Característica   |                 Valor |
| ---------------- | --------------------: |
| Total de reseñas |                50.000 |
| Positivas        |                25.000 |
| Negativas        |                25.000 |
| Tipo de problema | Clasificación binaria |

El dataset se encuentra perfectamente balanceado, lo que evita sesgos asociados a clases desproporcionadas.

Para optimizar tiempos de procesamiento y entrenamiento se utilizó una muestra estratificada de 10.000 registros.

### División utilizada

| Conjunto      | Registros |
| ------------- | --------: |
| Entrenamiento |     7.000 |
| Validación    |     1.500 |
| Test          |     1.500 |

---

# Tecnologías Utilizadas

## Procesamiento de Datos

* Python
* Pandas
* NumPy

## Procesamiento de Lenguaje Natural

* NLTK
* spaCy
* Expresiones Regulares (Regex)
* TF-IDF

## Machine Learning y Deep Learning

* Scikit-Learn
* PyTorch
* Hugging Face Transformers

## Visualización

* Matplotlib

---

# Pipeline del Proyecto

```text
Dataset IMDb
        ↓
Análisis Exploratorio (EDA)
        ↓
Limpieza de Texto (Regex)
        ↓
Tokenización
        ↓
Lematización
        ↓
Comparación NLTK vs spaCy
        ↓
Vectorización TF-IDF
        ↓
Red Neuronal en PyTorch
        ↓
Entrenamiento y Regularización
        ↓
Evaluación
        ↓
Comparación con DistilBERT
```

---

# Preprocesamiento de Texto

El pipeline de preprocesamiento incluyó:

* Conversión a minúsculas
* Eliminación de etiquetas HTML
* Eliminación de caracteres especiales
* Tokenización
* Lematización
* Eliminación de stopwords

Además, se realizó una comparación entre **NLTK** y **spaCy** para evaluar precisión y calidad lingüística.

## Herramienta seleccionada

Se optó por **spaCy** debido a:

* Mejor lematización contextual.
* Mayor precisión lingüística.
* Mejor rendimiento en procesamiento por lotes.

---

# Representación Numérica

## TF-IDF

Configuración utilizada:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=3,
    max_df=0.90,
    sublinear_tf=True
)
```

### ¿Por qué TF-IDF?

A diferencia del simple conteo de palabras, TF-IDF:

* Reduce el peso de términos excesivamente frecuentes.
* Resalta palabras discriminativas.
* Mejora la calidad de las características utilizadas por el modelo.
* Genera representaciones más adecuadas para clasificación de sentimiento.

---

# Arquitectura de Deep Learning

Se desarrolló un modelo propio utilizando PyTorch.

## TextClassifier

```text
Entrada (10.000 Features TF-IDF)
        ↓
Linear (512)
        ↓
BatchNorm
        ↓
Dropout (0.5)
        ↓
Linear (256)
        ↓
BatchNorm
        ↓
Dropout (0.5)
        ↓
Linear (128)
        ↓
BatchNorm
        ↓
Dropout (0.5)
        ↓
Salida (2 Clases)
```

### Configuración de Entrenamiento

* Optimizador: Adam
* Función de pérdida: CrossEntropyLoss
* Early Stopping (paciencia = 5)
* Batch Normalization
* Dropout = 0.5

### Parámetros entrenables

```text
5.286.786
```

---

# Resultados de Entrenamiento

Mejor desempeño obtenido durante validación:

| Métrica             | Resultado |
| ------------------- | --------: |
| Mejor época         |         1 |
| Validation Loss     |    0.3019 |
| Validation Accuracy |    0.8793 |

El mecanismo de Early Stopping detuvo automáticamente el entrenamiento cuando la pérdida de validación dejó de mejorar.

---

# Comparación de Modelos

Evaluación realizada sobre la misma muestra de prueba.

| Métrica  | TextClassifier | DistilBERT |
| -------- | -------------: | ---------: |
| Accuracy |         0.8700 |     0.8250 |
| F1-Score |         0.8632 |     0.8108 |

## Observación

La combinación de TF-IDF y red neuronal logró superar ligeramente al modelo DistilBERT preentrenado utilizado como referencia.

Este resultado demuestra la importancia de:

* Un buen preprocesamiento.
* Una correcta representación de características.
* Técnicas adecuadas de regularización.
* Selección apropiada del modelo para el problema.

---

# Principales Aprendizajes

Durante el desarrollo del proyecto se profundizó en conceptos fundamentales de NLP y Deep Learning:

* El preprocesamiento textual impacta directamente en el rendimiento final.
* spaCy ofrece ventajas significativas respecto a NLTK en tareas de lematización.
* TF-IDF continúa siendo una técnica altamente competitiva para clasificación de texto.
* La regularización es fundamental para controlar el sobreajuste.
* Modelos más complejos no garantizan necesariamente mejores resultados.
* Los Transformers representan una herramienta poderosa, pero modelos más simples pueden competir eficazmente en determinados contextos.

---

# Mejoras Futuras

Algunas líneas de trabajo futuras incluyen:

* Entrenamiento utilizando los 50.000 registros completos.
* Optimización sistemática de hiperparámetros.
* Implementación de Word Embeddings (Word2Vec, GloVe o FastText).
* Fine-Tuning de DistilBERT.
* Análisis detallado de errores de clasificación.
* Incorporación de técnicas de interpretabilidad y explicabilidad.

---

# Estructura del Repositorio

```text
imdb-sentiment-analysis-nlp/

├── notebooks/
│   └── imdb_sentiment_analysis.ipynb
│
├── reports/
│   └── Informe_Final_IMDB_NLP.pdf
│
├── docs/
│   └── Informe_Final_IMDB_NLP.docx
│
├── assets/
│   └── images/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Autor

**Rodolfo Riveros**

Data Science | Customer Experience | Gestión de Calidad

---

# Licencia

Este repositorio tiene fines educativos, de aprendizaje y portfolio profesional.
