import streamlit as st
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tensorflow.keras.models import load_model as keras_load_model

# Suppress sklearn's UserWarning
warnings.filterwarnings('ignore', category=UserWarning)

# New data loading caching using st.cache_data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/npiquiensoy/PrototipoTesis/main/DatasetClima.csv')
        df = df.dropna()
        df = df.drop(columns='Unnamed: 0')
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error
    
# New model caching using st.cache_resource
@st.cache_resource
def load_model(model_filename, model_type):
    try:
        if model_type in ['Decision Tree', 'Random Forest', 'Linear Regression', 'SVR']:
            model = joblib.load(model_filename)
        elif model_type == 'Neural Network':
            model = keras_load_model(model_filename)
        else:
            raise ValueError("Invalid model type")
        return model
    except Exception as e:
        st.error(f"Failed to load model {model_filename}: {e}")
        return None

def plot_predictions(y_test, y_pred, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    sns.lineplot(x=y_test, y=y_test, color='red')  # Identity line
    plt.title(title)
    plt.xlabel('Actual Biomass Values')
    plt.ylabel('Predicted Biomass Values')
    st.pyplot()


# Session state for page management
state = st.session_state
if 'page' not in state:
    state.page = 'home'
if 'subpage' not in state:
    state.subpage = 'home'

# Set the current page
def set_page(page_name):
    state.page = page_name
    state.subpage = 'home'  # Reset subpage when changing the main page

# Sidebar for main navigation
st.sidebar.title("Navegación Principal")
st.sidebar.button("¿Qué es la biomasa de cacao?", on_click=set_page, args=('biomasa',))
st.sidebar.button("Estimación de la biomasa de cacao", on_click=set_page, args=('estimacion',))
st.sidebar.button("Diccionario de variables", on_click=set_page, args=('variables',))
st.sidebar.button("El modelo simple", on_click=set_page, args=('modelo_simple',))
st.sidebar.button("Resultados de calibración", on_click=set_page, args=('calibracion',))
st.sidebar.button("Acerca de", on_click=set_page, args=('acerca_de',))

# Home page only shows a title
def home():
    st.title("Cacao - BEST")
    st.subheader("Cacao Biomass Estimation and Simulation Tool")
    st.markdown("""

**CacaoBEST** es un prototipo web desarrollado para la estimación de biomasa de cacao en la región de Santander, Colombia, utilizando técnicas de aprendizaje automático. La aplicación ha sido implementada utilizando el paquete Streamlit de Python para crear una interfaz de usuario interactiva y amigable.

## Objetivo

El objetivo de este prototipo es proporcionar una herramienta que permita a los agricultores y técnicos agrícolas estimar la biomasa de los cultivos de cacao mediante la utilización de modelos de aprendizaje automático basados en datos recolectados en campo.

## Alcance

Este prototipo está diseñado para:

- Integrar y procesar datos recolectados en plantaciones de cacao.
- Aplicar modelos de aprendizaje automático para estimar la biomasa.
- Presentar resultados de manera interactiva y visual a los usuarios finales.

## Tecnologías Utilizadas

### Lenguaje de Programación
- **Python**: Utilizado para el desarrollo de todos los componentes del prototipo.

### Paquetes y Librerías
- **Streamlit**: Paquete utilizado para crear la interfaz web interactiva.
- **Pandas**: Para la manipulación y análisis de datos.
- **Scikit-learn**: Para la implementación de modelos de aprendizaje automático.
- **TensorFlow/Keras**: Para la creación de redes neuronales.
- **Matplotlib/Seaborn**: Para la visualización de datos y resultados.

## Funcionalidades

- **Carga y Procesamiento de Datos**: Permite la carga de datos de plantaciones de cacao y su preprocesamiento para el análisis.
- **Selección de Modelos**: Ofrece la posibilidad de seleccionar entre diferentes modelos de aprendizaje automático (árboles de decisión, bosques aleatorios, regresión lineal, SVR, redes neuronales).
- **Estimación de Biomasa**: Utiliza los modelos seleccionados para realizar estimaciones de biomasa basadas en los datos ingresados.
- **Visualización Interactiva**: Presenta los resultados de la estimación de biomasa de manera visual e interactiva, facilitando la interpretación de los resultados.
- **Configuración de Parámetros del Modelo**: Permite a los usuarios ajustar los parámetros de entrada para refinar las estimaciones.

## Uso

Para utilizar **CacaoBEST**, sigue estos pasos:

1. **Carga de Datos**: Sube los datos de la plantación de cacao en formato CSV.
2. **Selección del Modelo**: Elige el modelo de aprendizaje automático deseado desde el menú lateral.
3. **Configuración de Parámetros**: Ajusta los parámetros del modelo si es necesario.
4. **Estimación**: Ejecuta la estimación de biomasa y visualiza los resultados de manera interactiva.
""")
    
    st.image("/workspaces/PrototipoTesis/Logo_1.png")

# Estimation page with the original code
def page_estimacion():
    st.title('Estimación de Biomasa de Cacao')
    st.write("En esta página, podrás cargar tus datos y seleccionar diferentes modelos de aprendizaje automático para estimar la biomasa. Usamos técnicas avanzadas como árboles de decisión, bosques aleatorios, regresión lineal, SVR y redes neuronales para proporcionar predicciones precisas. Ajusta los parámetros según tus necesidades y obtén una estimación de la biomasa esperada.")
    df = load_data()

    if df.empty:
        st.error("No data available.")
        return
    
    model_options = {
        'Decision Tree': 'arbol_decision (1).pkl',
        'Random Forest': 'bosque_aleatorio (1).pkl',
        'Linear Regression': 'modelo_lineal (1).pkl',
        'SVR': 'svr (1).pkl',
        'Neural Network': 'red_neuronal_model.h5'  # Actualiza el nombre del archivo
    }

    choose_model = st.selectbox('Seleccione el tipo de modelo', options=list(model_options.keys()), index=0)

    model_descriptions = {
        'Decision Tree': 'Un árbol de regresión es un modelo de aprendizaje supervisado que divide el conjunto de datos en subconjuntos más pequeños, basándose en características predictoras, para predecir un valor continuo. Se entrena dividiendo los datos en subconjuntos basados en condiciones de las variables predictoras.\n' 
            'Fortalezas y limitaciones: Su interpretación es sencilla y puede manejar datos no lineales. Sin embargo, puede ser propenso al sobreajuste y no es muy robusto a pequeñas variaciones en los datos de entrenamiento.',
        'Random Forest': 'Un bosque aleatorio es un conjunto de árboles de decisión que se entrenan con diferentes subconjuntos de datos y características, y luego combinan sus predicciones para mejorar la precisión y evitar el sobreajuste. Se entrena mediante la construcción de múltiples árboles de decisión con diferentes subconjuntos de datos y características.\n'
            'Fortalezas y limitaciones: Suele tener un rendimiento robusto y puede manejar conjuntos de datos grandes con muchas características. Sin embargo, puede ser computacionalmente intensivo y difícil de interpretar en comparación con un solo árbol de decisión.',
        'Linear Regression': 'La regresión lineal es un modelo que busca encontrar la mejor línea de ajuste para predecir una variable dependiente a partir de una o más variables independientes.\n'
            'Fortalezas y limitaciones: Es simple y fácil de interpretar, pero asume una relación lineal entre las variables y puede no capturar relaciones no lineales entre las características',
        'SVR': 'SVR es una variante de las Máquinas de Vectores de Soporte (SVM) utilizada para problemas de regresión, que busca encontrar una función de regresión que esté dentro de un margen de tolerancia especificado por un parámetro epsilon.\n'
            'Fortalezas y limitaciones: Puede manejar eficazmente conjuntos de datos de alta dimensionalidad y es robusto frente al sobreajuste. Sin embargo, puede ser sensible a la elección de hiperparámetros y puede ser computacionalmente costoso.',
        'Neural Network': 'Una red neuronal es un modelo inspirado en el cerebro humano que consta de capas de nodos interconectados, donde cada nodo realiza operaciones matemáticas en los datos de entrada y pasa el resultado a la siguiente capa.\n'
            'Fortalezas y limitaciones: Es capaz de modelar relaciones complejas entre las características y puede manejar grandes volúmenes de datos. Sin embargo, puede requerir grandes cantidades de datos de entrenamiento y ser susceptible al sobreajuste, especialmente con arquitecturas muy profundas.'
    }

    if choose_model:
        st.write(model_descriptions[choose_model])
        model_path = model_options[choose_model]
        model = load_model(model_path, choose_model)

        if model is None:
            st.error("Model could not be loaded.")
            return

        st.sidebar.header('Parámetros del Modelo')

        # Crear un diccionario para almacenar los valores de entrada con los valores por defecto proporcionados
        default_values = {
            'ALLSKY_SFC_SW_DWN': 3315.68,
            'CLRSKY_SFC_PAR_TOT': 24222.03,
            'PRECTOTCORR': 462.93,
            'RH2M': 13510.22,
            'WS2M': 133.8,
            'T2M_MAX': 4659.51,
            'T2M_MIN': 2795.93
        }

        inputs = {}
        for key, default_value in default_values.items():
            inputs[key] = st.sidebar.number_input(key, min_value=0.0, max_value=100000.0, value=default_value)


        input_values = np.array([list(inputs.values())])  # Convertir el diccionario a una matriz numpy

        # Convertir los valores a strings formateados para eliminar ceros innecesarios y redondear a 2 decimales
        input_values_str = np.vectorize(lambda x: f'{x:.2f}'.rstrip('0').rstrip('.'))(input_values)

        # Calcular anchos dinámicos para cada columna basado en la longitud de los valores
        column_widths = {key: max(100, len(str(value)) * 10) for key, value in zip(inputs.keys(), input_values_str[0])}

        # Imprimir los valores de entrada y estilizar las columnas
        st.write("Valores de entrada:")
        df_input = pd.DataFrame([input_values_str[0]], columns=list(inputs.keys()))

        styles = []
        for i, key in enumerate(inputs.keys()):
            styles.append({'selector': f'th.col_heading.level0.col{i}', 'props': [('width', f'{column_widths[key]}px')]})

        df_input_style = df_input.style.set_table_styles(styles)
        st.write(df_input_style.to_html(), unsafe_allow_html=True)

        # Separar el botón de la tabla
        st.write("\n")
        if st.button('Realiza una estimación'):
            scaler_filename = 'scaler.pkl'
            scaler = joblib.load(scaler_filename)  # Cargar el scaler guardado
            scaled_inputs = scaler.transform(input_values)

            prediction = model.predict(scaled_inputs)
            st.write(f'Biomasa esperada: {prediction[0]}')

# Additional pages content
def page_variables():
    st.title("Diccionario de variables")

    st.write("https://power.larc.nasa.gov/api/system/manager/system/parameters")
    variables_info = {
        'ALLSKY_SFC_SW_DWN': {
            'name': 'Radiación solar descendente total',
            'units': 'Watts por metro cuadrado (W/m²)',
            'description': 'Esta variable mide la cantidad de radiación solar que llega a la superficie de la Tierra bajo condiciones de cielo completamente despejado.',
            'average': 'Media diaria: 18.09 W/m²'
        },
        'CLRSKY_SFC_PAR_TOT': {
            'name': 'Radiación fotosintéticamente activa bajo cielo claro',
            'units': 'Micro moles por metro cuadrado por segundo (W/m²)',
            'description': 'Representa la cantidad de radiación fotosintéticamente activa (PAR) que alcanza la superficie durante condiciones de cielo claro.',
            'average': 'Media diaria: 135.08 W/m²'
        },
        'PRECTOTCORR': {
            'name': 'Precipitación total corregida',
            'units': 'Milímetros (mm)',
            'description': 'Mide la cantidad total de precipitación, incluyendo lluvia y precipitación equivalente de nieve, ajustada por posibles errores de medición.',
            'average': 'Media diaria: 3.49 mm'
        },
        'RH2M': {
            'name': 'Humedad relativa a 2 metros sobre la superficie',
            'units': 'Porcentaje (%)',
            'description': 'Indica el porcentaje de humedad relativa del aire a dos metros sobre la superficie.',
            'average': 'Media diaria: 76.99%.'
        },
        'WS2M': {
            'name': 'Velocidad del viento a 2 metros sobre la superficie',
            'units': 'Metros por segundo (m/s)',
            'description': 'Esta variable registra la velocidad del viento a una altura de dos metros sobre el suelo.',
            'average': 'Media diaria: 0.76 m/s.'
        },
        'T2M_MAX': {
            'name': 'Temperatura máxima a 2 metros sobre la superficie',
            'units': 'Grados Celsius (°C)',
            'description': 'Mide la temperatura máxima diaria a dos metros sobre la superficie.',
            'average': 'Media diaria: 25.66°C.'
        },
        'T2M_MIN': {
            'name': 'Temperatura mínima a 2 metros sobre la superficie',
            'units': 'Grados Celsius (°C)',
            'description': 'Registra la temperatura mínima diaria a dos metros sobre la superficie.',
            'average': 'Media diaria: 16.07°C.'
        }
    }

    for var, info in variables_info.items():
        st.subheader(f"{var} ({info['name']})")
        st.write(f"**Unidades:** {info['units']}")
        st.write(f"**Descripción:** {info['description']}")
        if info['average']:
            st.write(f"**Promedio:** {info['average']}")
        st.write("\n")

def page_biomasa():
    st.title("¿Qué es la biomasa de cacao?")
    st.write("Definicion segun la RAE: Materia orgánica originada en un proceso biológico, espontáneo o provocado, utilizable como fuente de energía. ")
    st.write("https://dle.rae.es/biomasa")
    st.write("En el caso de la biomasa de cacao es la cantidad total de materia orgánica producida por las plantas de cacao en una determinada área, en este proyecto usaremos g/m². Incluye todas las partes de la planta, como hojas, ramas, raíces y frutos. Medir la biomasa de cacao es importante porque ayuda a entender el crecimiento y la salud de los cultivos, lo que puede mejorar la productividad y la sostenibilidad de las plantaciones de cacao.")
    st.image("/workspaces/PrototipoTesis/ImagenBiomasa.jpeg")

def page_modelo_simple():
    st.title("El modelo simple")
    st.markdown("""

    El uso del modelo de cultivo SIMPLE una herramienta diseñada inicialmente para evaluar el impacto del cambio climático en la producción de cultivos. Dada la escasez de modelos específicos para numerosos cultivos, el modelo SIMPLE se desarrolló como un sistema genérico adaptable a cualquier tipo de cultivo para simular su desarrollo, crecimiento y rendimiento. Requiere entradas comúnmente disponibles como datos meteorológicos diarios, manejo del cultivo y parámetros de retención de agua del suelo.

    La biomasa estimada del cacao se utilizó como variable dependiente en modelos de aprendizaje automático, que fueron diseñados para identificar qué variables o factores ambientales influyen predominantemente en la producción del cacao. A través de estos modelos, se evaluó cómo diferentes condiciones ambientales afectan la producción de cacao, proporcionando datos cruciales que pueden apoyar la toma de decisiones relacionadas con tratamientos agrícolas específicos.

    ### Modelo Simple como Alternativa para Estimar la Producción de Biomasa 

    Un modelo simple para estimar la producción de biomasa es una herramienta accesible y fácil de interpretar que permite evaluar la cantidad de biomasa generada por una planta o un cultivo sin necesidad de recurrir a técnicas complejas o costosas. Estos modelos simplificados buscan capturar las relaciones esenciales entre un conjunto reducido de variables y la producción de biomasa, facilitando su uso en diversas aplicaciones prácticas.
    """)

def page_calibracion():
    st.title("Resultados de calibración")
    st.markdown("""
    ## ¿Qué son MAD, R² y RMSE?

    **MAD (Mean Absolute Deviation)**: El MAD es la desviación absoluta media, que mide el promedio de las diferencias absolutas entre los valores predichos por el modelo y los valores observados. Es un indicador de la precisión de un modelo; un MAD más bajo indica un modelo más preciso.

    **R² (Coeficiente de Determinación)**: El R² es una medida que indica qué tan bien se ajustan los datos al modelo. Va de 0 a 1, donde 1 significa un ajuste perfecto y 0 significa que el modelo no explica ninguna de las variaciones en los datos de respuesta. Un R² más alto indica un mejor ajuste del modelo a los datos.

    **RMSE (Root Mean Squared Error)**: El RMSE es la raíz cuadrada del error cuadrático medio, que mide la diferencia entre los valores predichos por el modelo y los valores observados. Es una medida de la precisión del modelo y se expresa en las mismas unidades que los datos originales. Un RMSE más bajo indica un modelo más preciso.

    ## ¿Cómo se obtienen estos valores?
    """)

    st.markdown("**MAD** se calcula como el promedio de las diferencias absolutas entre los valores predichos y observados:")
    st.latex(r'''
    \text{MAD} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
    ''')

    st.markdown("**R²** se calcula como 1 menos la suma de los cuadrados de los residuos dividida por la suma de los cuadrados totales:")
    st.latex(r'''
    R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
    ''')

    st.markdown("**RMSE** se calcula como la raíz cuadrada de la media de los cuadrados de las diferencias entre los valores predichos y observados:")
    st.latex(r'''
    \text{RMSE} = \sqrt{\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}}
    ''')

    st.markdown("""
    ## ¿Por qué son importantes?

    Estas métricas son cruciales para evaluar y comparar el rendimiento de los modelos de aprendizaje automático. Nos ayudan a entender qué tan bien un modelo puede predecir resultados basados en datos de entrada. Un buen modelo debe tener un MAD y RMSE bajos y un R² alto. Estas medidas proporcionan una evaluación cuantitativa de la precisión y la eficiencia del modelo, permitiendo seleccionar el modelo más adecuado para una tarea específica.

    En las imágenes mostradas, podemos observar cómo diferentes modelos de aprendizaje automático (Regresión Lineal, Árbol de Decisión, Bosque Aleatorio, Soporte Vectorial y Red Neuronal) se comparan en términos de estas métricas. Los resultados numéricos y gráficos proporcionan una visión clara de cuál modelo ofrece las predicciones más precisas para la estimación de la biomasa de cacao.
    """)

    # Mostrar las imágenes cargadas
    st.image("/workspaces/PrototipoTesis/R2, MAD y RMSE.png", caption='Resultados Numéricos de la Calibración')
    st.image("/workspaces/PrototipoTesis/Grafica comparatiava de R2, MAD y RMSE.png", caption='Comparación de Modelos')

    st.markdown("""
    ## Conclusión del Modelo Recomendado

    Según los resultados mostrados, el modelo de Árbol de Decisión y el Bosque Aleatorio presentan los mejores valores en términos de R², MAD y RMSE, con valores muy cercanos a 1 en R² y extremadamente bajos en MAD y RMSE. Esto indica que estos modelos son los más precisos para la estimación de la biomasa de cacao. En particular, el Bosque Aleatorio es recomendado por su balance óptimo entre precisión y capacidad de generalización.
    """)

def page_acerca_de():
    st.title("Acerca de")
    st.subheader("Descripción General del Proyecto")
    st.markdown("""
    **CacaoBEST** es una herramienta desarrollada para la estimación de la biomasa del cacao en la región de Santander, Colombia. Utiliza técnicas avanzadas de aprendizaje automático para analizar datos meteorológicos y otros factores ambientales con el fin de proporcionar estimaciones precisas y útiles para agricultores y técnicos agrícolas.
    """)

    st.subheader("Objetivos del Proyecto")
    st.markdown("""
    - Desarrollar un modelo preciso para estimar la biomasa del cacao.
    - Facilitar la toma de decisiones agrícolas mediante el uso de tecnologías avanzadas.
    - Proveer una plataforma interactiva y fácil de usar para los usuarios finales.
    """)

    st.subheader("Equipo del Proyecto")
    st.markdown("""
    **Juan Antonio Hernández Gómez** - Estudiante de tesis
    - Estudiante aplicando al titulo de Ingeniero en Sistemas.

    **Leonardo Hernán Talero Sarmiento** - Director de tesis
    - Ingeniero industrial (UIS).
    - Magister en ingeniería industrial (UIS).
    - Candidato a doctor en ingeniería (UNAB).

    **Feisar Enrique Moreno Corzo** - Co-director de tesis
    - Ingeniero de sistemas (UIS).
    - Magíster en Gestión, Aplicación y Desarrollo de Software (UNAB).
    """)

    st.subheader("Instituciones y Colaboradores")
    st.markdown("""
    Este proyecto fue deserollado para la Universidad Autonoma de Bucaramanga (UNAB).
    """)

    st.subheader("Tecnologías y Herramientas Utilizadas")
    st.markdown("""
    - **Python**: Lenguaje de programación principal utilizado para el desarrollo del proyecto. Puedes encontrar más información y descargar Python desde su [sitio web oficial](https://python.org).
    - **Streamlit**: Paquete utilizado para crear la interfaz web interactiva. Visita el [sitio oficial de Streamlit](https://streamlit.io) para más detalles.
    - **Scikit-learn**: Para la implementación de modelos de aprendizaje automático. La documentación oficial y recursos se encuentran en el [sitio web de Scikit-learn](https://scikit-learn.org).
    - **TensorFlow**: Para la creación y entrenamiento de redes neuronales. El sitio oficial de TensorFlow es [tensorflow.org](https://tensorflow.org).
    - **Pandas** y **NumPy**: Para la manipulación y análisis de datos. Puedes encontrar la documentación oficial de Pandas en [pandas.pydata.org](https://pandas.pydata.org) y de NumPy en [numpy.org](https://numpy.org).
    """)

    st.subheader("Contacto")
    st.markdown("""
    Para más información, puedes contactarnos a través de nuestro correo electrónico: **jhernandez296@unab.edu.co**.
    """)

    st.image("/workspaces/PrototipoTesis/Logo_1.png")

# Dictionary of pages
pages = {
    'home': home,
    'biomasa': page_biomasa,
    'estimacion': page_estimacion,
    'variables': page_variables,
    'modelo_simple': page_modelo_simple,
    'calibracion': page_calibracion,
    'acerca_de': page_acerca_de
}

# Execute the function of the selected page
if state.page in pages:
    pages[state.page]()
