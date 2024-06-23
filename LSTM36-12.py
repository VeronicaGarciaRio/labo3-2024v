import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Lista para acumular las filas de resultados
resultados_por_producto = []

# Elegir un número de pasos de tiempo
n_steps_top = 36  # Ventana de tiempo de 36 meses para productos top
n_steps_rest = 12  # Ventana de tiempo de 12 meses para productos restantes
n_features = 1  # Cambia esto si tienes más características

# Calcular las ventas totales por producto y obtener los 70 productos con mayores ventas
ventas_totales_por_producto = sell_in_completo.groupby("product_id")["tn"].sum()
top_70_productos = ventas_totales_por_producto.nlargest(70).index

# Función para crear secuencias de tiempo
def crear_secuencias(datos, n_steps, step_ahead=1):
    X, y = [], []
    for i in range(len(datos)):
        end_ix = i + n_steps
        out_end_ix = end_ix + step_ahead - 1
        if out_end_ix > len(datos) - 1:
            break
        seq_x, seq_y = datos[i:end_ix], datos[out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Inicializar scaler
scaler = MinMaxScaler()

# Definir la función para construir el modelo para productos top
def build_top_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(units=128, activation='relu'), input_shape=(n_steps_top, n_features)))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model

# Definir la función para construir el modelo para productos restantes
def build_rest_model():
    model = Sequential()
    model.add(LSTM(units=16, activation='relu', input_shape=(n_steps_rest, n_features)))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model

# Inicializar EarlyStopping para los modelos top
early_stopping_top = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Inicializar EarlyStopping para los modelos restantes
early_stopping_rest = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Función para entrenar y predecir para un producto específico con el modelo top
def entrenar_y_predecir_top(aux, producto):
    try:
        # Normalizar los datos
        aux_normalizado = scaler.fit_transform(aux)

        # Crear secuencias de tiempo
        X, y = crear_secuencias(aux_normalizado, n_steps_top, step_ahead=2)
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        # Construir el modelo
        model = build_top_model()

        # Entrenar el modelo con todos los datos
        model.fit(X, y, epochs=100, callbacks=[early_stopping_top], verbose=0)

        # Preparar los datos para la predicción
        ultima_secuencia = aux_normalizado[-(n_steps_top + 1):-1].reshape((1, n_steps_top, n_features))

        # Predecir las ventas para el próximo período
        prediccion_normalizada = model.predict(ultima_secuencia)

        # Desnormalizar la predicción
        prediccion = scaler.inverse_transform(prediccion_normalizada)

        # Agregar las predicciones a la lista de resultados
        resultados_por_producto.append({'product_id': producto, 'prediccion_1': prediccion[0][0]})
    except Exception as e:
        print(f"Error al procesar el producto {producto}: {str(e)}")
        resultados_por_producto.append({'product_id': producto, 'prediccion_1': 0})

# Función para entrenar el modelo único para productos restantes
def entrenar_modelo_rest(productos_restantes):
    try:
        # Agrupar datos de todos los productos restantes
        aux_combined = []
        for producto in productos_restantes:
            aux = sell_in_completo[sell_in_completo["product_id"] == producto].drop(columns=["product_id"]).values
            aux_normalizado = scaler.fit_transform(aux)
            aux_combined.append(aux_normalizado)

        aux_combined = np.concatenate(aux_combined, axis=0)

        # Crear secuencias de tiempo
        X, y = crear_secuencias(aux_combined, n_steps_rest, step_ahead=2)
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        # Construir el modelo
        model = build_rest_model()

        # Entrenar el modelo con todos los datos
        model.fit(X, y, epochs=10, callbacks=[early_stopping_rest], verbose=0)

        return model
    except Exception as e:
        print(f"Error al entrenar el modelo para los productos restantes: {str(e)}")
        return None

# Función para predecir para un producto específico con el modelo de productos restantes
def predecir_rest(model, aux, producto):
    try:
        # Normalizar los datos
        aux_normalizado = scaler.fit_transform(aux)

        # Preparar los datos para la predicción
        ultima_secuencia = aux_normalizado[-(n_steps_rest + 1):-1].reshape((1, n_steps_rest, n_features))

        # Predecir las ventas para el próximo período
        prediccion_normalizada = model.predict(ultima_secuencia)

        # Desnormalizar la predicción
        prediccion = scaler.inverse_transform(prediccion_normalizada)

        # Agregar las predicciones a la lista de resultados
        resultados_por_producto.append({'product_id': producto, 'prediccion_1': prediccion[0][0]})
    except Exception as e:
        print(f"Error al procesar el producto {producto}: {str(e)}")
        resultados_por_producto.append({'product_id': producto, 'prediccion_1': 0})

# Procesar los productos top 30
for producto in top_70_productos:
    aux = sell_in_completo[sell_in_completo["product_id"] == producto].drop(columns=["product_id"]).values
    entrenar_y_predecir_top(aux, producto)

# Procesar los productos restantes
productos_restantes = sell_in_completo[~sell_in_completo["product_id"].isin(top_70_productos)]["product_id"].unique()

# Entrenar el modelo único para los productos restantes
modelo_rest = entrenar_modelo_rest(productos_restantes)

# Realizar predicciones individuales para cada producto restante con el modelo único
if modelo_rest is not None:
    for producto in productos_restantes:
        aux = sell_in_completo[sell_in_completo["product_id"] == producto].drop(columns=["product_id"]).values
        predecir_rest(modelo_rest, aux, producto)

# Convertir la lista de resultados en un DataFrame
predicciones_por_producto = pd.DataFrame(resultados_por_producto)

# Mostrar el DataFrame con las predicciones por producto
print(predicciones_por_producto)

# Guardar el DataFrame con las predicciones por producto en un archivo CSV
predicciones_por_producto.to_csv("C:/Users/vgarciario/desktop/UA MASTER/labo3/lstm36-12m.csv", index=False)




