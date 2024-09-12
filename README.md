INFORME SOBRE LA APLICACIÓN DEL FRAMEWORK

Estudiante: Jhamil Crespo Rejas
Carrera: Ingeniería en Ciencias de la Computación

1. Preparación de los datos
El código inicia con la carga y preprocesamiento de imágenes, pero debido a las limitaciones de memoria RAM, se decidió trabajar con un conjunto de datos reducido. Para evitar problemas de rendimiento, se tomaron las siguientes decisiones:
Reducción de las imágenes: Las imágenes originales de 800x800 píxeles se modificaron manualmente a 128x128 píxeles fuera del algoritmo para que el procesamiento fuera manejable. Esto pudo haber influido negativamente en la capacidad del modelo para extraer detalles importantes.
Tamaño reducido del dataset: Se utilizaron 2000 imágenes, divididas en 1600 para entrenamiento y 400 para validación.
Se utilizaron librerías como Flux, Images, Colors, y otras para manejar la manipulación de imágenes y la construcción del modelo.
2. Explicación del framework Flux
Flux es un framework para redes neuronales escrito en Julia, diseñado para ser flexible, simple y eficiente, aprovechando las ventajas de Julia en computación numérica de alto rendimiento.
Características principales de Flux:
Modelo basado en cadenas (Chains):
Las redes neuronales en Flux se definen mediante el uso de la función Chain, que permite apilar capas de manera secuencial. Cada capa se define como un objeto que toma la entrada y produce una salida transformada.
Las capas que se pueden utilizar incluyen capas densas (Dense), de convolución, recurrentes, activaciones como ReLU, y regularizaciones como Dropout.
Optimizadores:
Flux ofrece varios algoritmos de optimización, como SGD, ADAM, y otros. Estos optimizadores actualizan los parámetros del modelo para minimizar la función de pérdida.
En este caso, se utilizó el optimizador ADAM, que ajusta los pesos del modelo adaptativamente para mejorar el rendimiento del entrenamiento.
Diferenciación automática:
Una de las ventajas más notables de Flux es su capacidad de realizar diferenciación automática, lo que significa que puede calcular los gradientes de manera eficiente sin que el usuario tenga que preocuparse por la implementación matemática de las derivadas.
Esto se consigue mediante la función Flux.train!, que toma la función de pérdida, los parámetros del modelo y los datos para ajustar los pesos de manera automática.
Compatibilidad con GPU:
Flux tiene soporte integrado para el uso de GPU mediante la librería CUDA.jl, lo que acelera el entrenamiento en hardware compatible.
Facilidad de integración con otras librerías:
Flux puede integrarse fácilmente con librerías numéricas y científicas de Julia, lo que permite una personalización y optimización del flujo de trabajo de entrenamiento.
3. Arquitectura del Modelo
El modelo implementado es una red neuronal profunda (MLP), construida en Flux con las siguientes capas:
Dense: Capa conectada con 512 neuronas y activación ReLU.
Dropout: Una capa de regularización con 50% de dropout.
Dense: Otra capa con 256 neuronas y activación ReLU.
Dense: Capa de salida con 5 neuronas, correspondiente a las 5 clases.
Softmax: Función de activación utilizada para obtener probabilidades de clase.
4. Entrenamiento del Modelo
Función de pérdida: Se utilizó la cross entropy (Flux.crossentropy) para evaluar la pérdida.
Optimizador: El optimizador utilizado fue ADAM con una tasa de aprendizaje inicial de 0.001.
El entrenamiento se realizó durante 20 épocas, y la tasa de aprendizaje se redujo cada 5 épocas. Los resultados obtenidos fueron:
La pérdida de entrenamiento bajó de 1.57 a 1.22.
La pérdida de validación se redujo de 1.58 a 1.29.

5. Resultados
Precisión en el conjunto de validación: La precisión del modelo en el conjunto de validación fue del 39%, lo que sugiere que aún tiene dificultades para generalizar bien.
Predicción de prueba: Al probar con una imagen de la clase AmorSeco, el modelo predijo incorrectamente la clase Perejil.
6. Limitaciones
El rendimiento del modelo podría estar afectado por:
Reducción de imágenes: Las imágenes fueron reducidas manualmente de 800x800 a 128x128, lo que puede haber hecho que se perdieran detalles importantes.
Tamaño del conjunto de datos: Se utilizó un conjunto limitado de 2000 imágenes, lo que podría haber afectado la capacidad del modelo para generalizar correctamente.
7. Posibles mejoras
Mayor cantidad de datos: Aumentar el tamaño del dataset podría mejorar la capacidad de generalización.
Ajustes en la arquitectura: Añadir más neuronas o capas puede mejorar la capacidad del modelo para aprender características más complejas, aunque es importante controlar el sobreajuste.
