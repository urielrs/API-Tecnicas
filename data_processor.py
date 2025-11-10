import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json
import os
import io # <--- IMPORTANTE: Nueva librería para manejar buffers de texto

# --- Configuración y Carga de Datos ---
# ASEGÚRATE DE QUE ESTA RUTA ES CORRECTA para tu archivo CSV
DATA_FILE = 'TotalFeatures-ISCXFlowMeter.csv' 
RESULTS_DIR = 'api/results' # Directorio donde guardaremos los JSON

# Función auxiliar para eliminar etiquetas
def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

# Crear el directorio de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Cargar el DataFrame
try:
    df = pd.read_csv(DATA_FILE)
    print("DataFrame cargado exitosamente.")
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo de datos en {DATA_FILE}. Por favor, verifica la ruta.")
    exit()

# 2. Preprocesamiento mínimo
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Separar características (X) y objetivo (y). 'calss' es el nombre de tu columna objetivo.
X, y = remove_labels(df, 'calss')

# Dividir para obtener datos de entrenamiento (necesario para el modelo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Entrenamiento del modelo para obtener feature_importances_
clf_rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_rnd.fit(X_train, y_train)


# --- Generación de Resultados Solicitados y Guardado en JSON ---

# Función para guardar un objeto de Pandas como JSON
def save_to_json(data, filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    if isinstance(data, pd.DataFrame):
        # Guardar DataFrames usando orient='split' para facilitar la lectura en el frontend
        data.to_json(filepath, orient='split', index=True) 
    elif isinstance(data, pd.Series):
        # Guardar Series como JSON simple
        data.to_json(filepath, orient='index')
    else:
        # Para listas, diccionarios o texto
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4) 
        
    print(f"Resultado guardado en: {filepath}")

# 1. df.head(10)
save_to_json(df.head(10), 'df_head_10.json')

# 2. df.info() - CÓDIGO CORREGIDO AQUÍ
info_buffer = io.StringIO()
df.info(buf=info_buffer)
info_output_string = info_buffer.getvalue()
save_to_json(info_output_string, 'df_info.json')

# 3. df['calss'].value_counts()
save_to_json(df['calss'].value_counts(), 'calss_value_counts.json')

# 4. df.describe()
save_to_json(df.describe(), 'df_describe.json')

# 5. corr_matrix['calss'].sort_values(ascending=False)
y_coded = y.astype('category').cat.codes
calss_corr = X.apply(lambda x: x.corr(y_coded))
save_to_json(calss_corr.sort_values(ascending=False), 'calss_corr_sorted.json')

# 6. X.corr()
save_to_json(X.corr(), 'X_corr_matrix.json')

# 7. clf_rnd.feature_importances_
feature_importances = clf_rnd.feature_importances_

# 8. feature_importances_sorted.head(20)
feature_importances_series = pd.Series(feature_importances, index=X_train.columns)
feature_importances_sorted = feature_importances_series.sort_values(ascending=False)
save_to_json(feature_importances_sorted.head(20), 'feature_importances_top_20.json')

# 9. columns (Top 10 features)
columns_top_10 = list(feature_importances_sorted.head(10).index)
save_to_json(columns_top_10, 'columns_top_10.json')

# 10. X_train_reduced.head(10) (Usando solo las 10 columnas seleccionadas)
X_train_reduced = X_train[columns_top_10]
save_to_json(X_train_reduced.head(10), 'X_train_reduced_head_10.json')

print("\n¡Procesamiento de datos finalizado! Los resultados están en 'api/results'.")