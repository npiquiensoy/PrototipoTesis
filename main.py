#Manejo de información
import streamlit as st
import pandas as pd
import numpy as np
import joblib
# import pickle as pkl

#Visualizacion de datos
import seaborn as sns
import matplotlib.pyplot as plt

#Modelos
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Función para cargar los datos
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/npiquiensoy/PrototipoTesis/main/DatasetClima.csv?token=GHSAT0AAAAAACSRYUNRLLZLERCYRKZIUH5WZSL5CZA') #Cargar datos desde un csv
    # df['Altitud'] = df['Altitud']
    df = df.dropna()

# Cargar los datos
df = load_data()

# Título de la aplicación
st.title('Estimación de Biomasa de Cacao')

# Seleccionar el tipo de modelo
model_type = st.selectbox('Seleccione el tipo de modelo', ['Choose', 'Decision Tree', 'Random Forest', 'Linear Regression', 'SVR', 'Neural Network'])
if model_type == 'Decision Tree':
    model = joblib.load('arbol_decision.pkl') 
    print("ok")
elif model_type == 'Random Forest':
    print()
elif model_type == 'Linear Regression':
    print()
elif model_type =='SVR':
    print()
elif model_type == 'Neural Network':
    print()

# Ingresar los parámetros del modelo
st.sidebar.header('Parámetros del Modelo')
ALLSKY_SFC_SW_DWN = st.sidebar.number_input('ALLSKY_SFC_SW_DWN', min_value=0.0, max_value=1000.0, value=200.0)
CLRSKY_SFC_PAR_TOT = st.sidebar.number_input('CLRSKY_SFC_PAR_TOT', min_value=0.0, max_value=1000.0, value=200.0)
PRECTOTCORR = st.sidebar.number_input('PRECTOTCORR', min_value=0.0, max_value=1000.0, value=100.0)
RH2M = st.sidebar.number_input('RH2M', min_value=0.0, max_value=100.0, value=50.0)
WS2M = st.sidebar.number_input('WS2M', min_value=0.0, max_value=100.0, value=5.0)
T2M_MAX = st.sidebar.number_input('T2M_MAX', min_value=0.0, max_value=50.0, value=25.0)
T2M_MIN = st.sidebar.number_input('T2M_MIN', min_value=0.0, max_value=50.0, value=15.0)
Altitud = st.sidebar.number_input('Altitud', min_value=0.0, max_value=10000.0, value=1000.0)

# Preprocesamiento de los datos
# X = df[['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_PAR_TOT', 'PRECTOTCORR', 'RH2M', 'WS2M', 'T2M_MAX', 'T2M_MIN', 'Altitud']]
# y = df['Biomass']

# # Dividir los datos en conjuntos de entrenamiento y prueba
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1952)

# # Entrenar el modelo
# model, scaler = train_model(model_type, X_train, y_train)

# # Realizar predicciones
# input_data = np.array([[ALLSKY_SFC_SW_DWN, CLRSKY_SFC_PAR_TOT, PRECTOTCORR, RH2M, WS2M, T2M_MAX, T2M_MIN, Altitud]])
# if model_type == 'SVR':
#     input_data = scaler.transform(input_data)
# biomass_prediction = model.predict(input_data)

# # Mostrar la estimación de biomasa
# st.subheader('Estimación de Biomasa')
# st.write(f'La biomasa estimada es: {biomass_prediction[0]:.2f}')

# # Visualizar gráficos de aptitud en función de las variables climatológicas
# st.subheader('Gráficas de Aptitud en Función de Variables Climatológicas')

# # Seleccionar variable a visualizar
# variable = st.selectbox('Seleccione la variable a visualizar', X.columns)

# # Visualización de gráficos

# plt.figure(figsize=(18, 9))
# sns.boxplot(x='Aptitud', y=variable, data=df)
# sns.stripplot(x='Aptitud', y=variable, data=df, color='red', alpha=0.5)
# plt.title(f'Boxplot de {variable} según Aptitud')
# plt.xlabel('Aptitud')
# plt.ylabel(variable)
# st.pyplot(plt)