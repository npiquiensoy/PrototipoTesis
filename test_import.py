import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('/content/drive/MyDrive/tesis/DatasetClima.xlsx')

# Imprimir el DataFrame antes de eliminar la columna 'Location'
print("DataFrame antes de eliminar 'Location':")
print(df)

# Eliminar la columna 'Location'
df = df.drop(columns='Location')

# Imprimir el DataFrame después de eliminar la columna 'Location'
print("DataFrame después de eliminar 'Location':")
print(df)

# Independent variables (X) and dependent variable (y)
X = df[['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_PAR_TOT', 'PRECTOTCORR', 'RH2M', 'WS2M', 'T2M_MAX', 'T2M_MIN']]
y = df['Biomass']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame with correct feature names for consistency in predictions
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, r2

# Linear Regression Model
linear_model = LinearRegression()
linear_y_pred, linear_mse, linear_r2 = evaluate_model(linear_model, X_train_scaled, X_test_scaled, y_train, y_test)
print("Linear Regression MSE:", linear_mse)
print("Linear Regression R^2:", linear_r2)

# Decision Tree Model
tree_model = DecisionTreeRegressor(random_state=42)
tree_y_pred, tree_mse, tree_r2 = evaluate_model(tree_model, X_train_scaled, X_test_scaled, y_train, y_test)
print("Decision Tree MSE:", tree_mse)
print("Decision Tree R^2:", tree_r2)

# Random Forest Model
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_y_pred, forest_mse, forest_r2 = evaluate_model(forest_model, X_train_scaled, X_test_scaled, y_train, y_test)
print("Random Forest MSE:", forest_mse)
print("Random Forest R^2:", forest_r2)

# Support Vector Machine Model
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
svr_y_pred = svr_model.predict(X_test_scaled)
svr_mse = mean_squared_error(y_test, svr_y_pred)
svr_r2 = r2_score(y_test, svr_y_pred)
print("SVR MSE:", svr_mse)
print("SVR R^2:", svr_r2)

# Neural Network Model
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(1))  # Output layer
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
nn_y_pred = nn_model.predict(X_test_scaled).flatten()
nn_mse = mean_squared_error(y_test, nn_y_pred)
nn_r2 = r2_score(y_test, nn_y_pred)
print("Neural Network MSE:", nn_mse)
print("Neural Network R^2:", nn_r2)

# Crear un DataFrame para comparar los modelos
comparison_df = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'Árbol de Decisión', 'Bosque Aleatorio', 'Soporte Vectorial', 'Red Neuronal'],
    'MSE': [linear_mse, tree_mse, forest_mse, svr_mse, nn_mse],
    'R^2': [linear_r2, tree_r2, forest_r2, svr_r2, nn_r2]
})

print(comparison_df)

# Función para crear scatter plots
def plot_predictions(y_test, y_pred, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    sns.lineplot(x=y_test, y=y_test, color='red')  # Línea de identidad
    plt.title(title)
    plt.xlabel('Valores Reales de Biomass')
    plt.ylabel('Valores Estimados de Biomass')
    plt.show()

# Crear gráficos de dispersión para cada modelo
plot_predictions(y_test, linear_y_pred, 'Valores Reales vs. Valores Estimados (Regresión Lineal)')
plot_predictions(y_test, tree_y_pred, 'Valores Reales vs. Valores Estimados (Árbol de Decisión)')
plot_predictions(y_test, forest_y_pred, 'Valores Reales vs. Valores Estimados (Bosque Aleatorio)')
plot_predictions(y_test, svr_y_pred, 'Valores Reales vs. Valores Estimados (Soporte Vectorial)')
plot_predictions(y_test, nn_y_pred, 'Valores Reales vs. Valores Estimados (Red Neuronal)')