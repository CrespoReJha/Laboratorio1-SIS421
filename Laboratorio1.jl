# Importar las librerías necesarias
using Flux          # Para definir y entrenar redes neuronales
using Images        # Para cargar y procesar imágenes
using Colors        # Para manejar colores en las imágenes (convertirlas a escala de grises)
using FileIO        # Para trabajar con el sistema de archivos
using Glob          # Para trabajar con patrones de archivos (no se usa directamente en este código)
using Statistics    # Para funciones estadísticas como `mean`
using Random        # Para generar números aleatorios (para mezclar los índices de datos)

# Definir una función para cargar las imágenes y sus etiquetas
function load_images_and_labels(dataset_dir::String, classes::Vector{String})
    image_files = []  # Inicializar lista para almacenar las imágenes cargadas
    labels = []       # Inicializar lista para almacenar las etiquetas correspondientes

    # Iterar sobre cada clase
    for class in classes
        class_dir = joinpath(dataset_dir, class)  # Obtener la ruta del directorio de cada clase
        for file in readdir(class_dir, join=true) # Leer todos los archivos en ese directorio
            if endswith(file, ".jpg")  # Si el archivo es una imagen con extensión ".jpg"
                img = load(file)  # Cargar la imagen
                img_gray = Gray.(img)  # Convertir la imagen a escala de grises
                img_resized = imresize(img_gray, (128, 128))  # Redimensionar la imagen a 128x128 píxeles
                img_array = Float32.(img_resized) / 255.0  # Normalizar la imagen (valores entre 0 y 1)
                push!(image_files, img_array)  # Almacenar la imagen procesada
                push!(labels, class)  # Almacenar la etiqueta correspondiente
            end
        end
    end

    num_images = length(image_files)  # Número total de imágenes cargadas
    img_height, img_width = size(image_files[1])  # Obtener el tamaño de una imagen
    # Aplanar las imágenes (convertirlas en vectores) y almacenarlas en una matriz
    images = hcat([reshape(img, img_height * img_width) for img in image_files]...)
    # Convertir las etiquetas en formato one-hot (binario)
    labels_onehot = Flux.onehotbatch(labels, classes)

    return images, labels_onehot  # Retornar las imágenes y sus etiquetas en formato one-hot
end

# Directorio que contiene el dataset de imágenes
dataset_dir = "C:\\Users\\Jhamil\\Desktop\\Dataset2mil - copia"

# Clases de las plantas en el dataset
classes = ["AmorSeco", "Boldo", "Charanguillo", "Ortiga", "Perejil"]

# Cargar las imágenes y etiquetas usando la función definida
train_x, train_y = load_images_and_labels(dataset_dir, classes)

# Crear un modelo MLP (Perceptrón Multicapa) con Flux
model = Chain(
    Dense(128*128, 512, relu),  # Primera capa densa: 128*128 entradas, 512 neuronas, activación ReLU
    Dropout(0.5),               # Capa de dropout: apaga aleatoriamente el 50% de las neuronas
    Dense(512, 256, relu),      # Segunda capa densa: 512 entradas, 256 neuronas, activación ReLU
    Dropout(0.5),               # Otra capa de dropout (50%)
    Dense(256, length(classes)), # Capa de salida: 256 entradas, tantas salidas como clases (5)
    softmax                     # Función softmax para obtener probabilidades de clasificación
) |> gpu  # Utilizar la GPU si está disponible para acelerar el procesamiento

# Definir la función de pérdida utilizando entropía cruzada
function loss(x, y)
    pred = model(x)  # Realizar la predicción
    Flux.crossentropy(pred, y)  # Calcular la entropía cruzada entre predicción y etiqueta real
end

# Definir el optimizador ADAM con una tasa de aprendizaje de 0.001
optimizer = Flux.Optimise.ADAM(0.001)

# Dividir los datos en conjunto de entrenamiento y validación
num_samples = size(train_x, 2)  # Número total de muestras

# Mezclar los índices aleatoriamente
indices = shuffle(1:num_samples)
split_idx = round(Int, 0.8 * num_samples)  # Dividir el 80% de los datos para entrenamiento
train_indices = indices[1:split_idx]  # Índices de entrenamiento
val_indices = indices[split_idx+1:end]  # Índices de validación

# Seleccionar los datos de entrenamiento y validación
train_x_data = train_x[:, train_indices]  # Imágenes de entrenamiento
train_y_data = train_y[:, train_indices]  # Etiquetas de entrenamiento
val_x_data = train_x[:, val_indices]      # Imágenes de validación
val_y_data = train_y[:, val_indices]      # Etiquetas de validación

# Preparar los datos para el entrenamiento
train_data = [(train_x_data[:, i], train_y_data[:, i]) for i in 1:size(train_x_data, 2)]  # Crear pares de datos de entrada y etiqueta

# Definir el optimizador inicial
initial_lr = 0.001  # Tasa de aprendizaje inicial
optimizer = Flux.Optimise.ADAM(initial_lr)  # Usar el optimizador ADAM con esta tasa de aprendizaje

# Entrenar el modelo durante 20 épocas
for epoch in 1:20
    if epoch % 5 == 0  # Cada 5 épocas
        current_lr = initial_lr * (0.5 ^ (epoch ÷ 10))  # Reducir la tasa de aprendizaje a la mitad cada 10 épocas
        optimizer = Flux.Optimise.ADAM(current_lr)  # Actualizar el optimizador con la nueva tasa de aprendizaje
    end

    # Entrenar el modelo utilizando los datos de entrenamiento y el optimizador
    Flux.train!(loss, Flux.params(model), train_data, optimizer)
    
    # Calcular la pérdida para el conjunto de entrenamiento y validación
    train_loss = loss(train_x_data, train_y_data)
    val_loss = loss(val_x_data, val_y_data)
    
    # Imprimir el número de la época y las pérdidas correspondientes
    println("Epoch: $epoch - Train Loss: $train_loss - Validation Loss: $val_loss")
end

# Evaluar el modelo en el conjunto de validación
function accuracy(x, y)
    predictions = Flux.onecold(model(x))  # Obtener las predicciones del modelo (índices de la clase más probable)
    labels = Flux.onecold(y)  # Convertir las etiquetas one-hot a índices
    mean(predictions .== labels)  # Calcular la precisión (porcentaje de predicciones correctas)
end

# Imprimir la precisión en el conjunto de validación
println("Validation Accuracy: ", accuracy(val_x_data, val_y_data))

# Definir una función para predecir la clase de una imagen
function predict_image(img_path::String)
    # Verificar si el archivo de imagen existe
    if !isfile(img_path)
        error("No file exists at given path: $img_path")
    end

    # Cargar y procesar la imagen
    img = load(img_path)  # Cargar la imagen
    img_gray = Gray.(img)  # Convertirla a escala de grises
    img_resized = imresize(img_gray, (128, 128))  # Redimensionar la imagen a 128x128 píxeles
    img_array = Float32.(img_resized) / 255.0  # Normalizar los valores de la imagen
    img_vector = reshape(img_array, 128*128)  # Convertir la imagen a un vector para pasarla al modelo

    # Realizar la predicción utilizando el modelo
    prediction = model(img_vector)
    predicted_class = classes[Flux.onecold(prediction)]  # Convertir la predicción a una clase
    return predicted_class  # Devolver la clase predicha
end

# Ruta de una imagen de prueba
test_img_path = "C:\\Users\\Jhamil\\Desktop\\Dataset2mil - copia\\AmorSeco\\1bd631a03aff5b854b6a3b1edf0f26efbda6ee55_center_mirror_vertical.jpg"

# Realizar la predicción de la imagen de prueba e imprimir la clase predicha
println("Predicted Class: ", predict_image(test_img_path))
