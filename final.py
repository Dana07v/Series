from __future__ import print_function
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import Dash, html, dash_table, dcc
import pandas as pd
import numpy as np
import subprocess
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_pacf
from scikeras.wrappers import KerasRegressor
from statsmodels.tsa.stattools import pacf
rcParams['figure.figsize'] = 15, 10
from pandas.plotting import register_matplotlib_converters
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import statsmodels.api as sm
import datetime as dt
import plotly.express as px
from scipy.stats import boxcox
import scipy as sp
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
import sklearn
import matplotlib.pylab as plt
from pandas.plotting import register_matplotlib_converters
import heapq
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from pandas import Series
register_matplotlib_converters()
from itertools import product
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import os
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import scipy as sp
from pandas import DataFrame
import sklearn
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
rcParams['figure.figsize'] = 15, 6
from pandas.plotting import register_matplotlib_converters
import heapq
import tensorflow as tf
layers = tf.keras.layers
from datetime import timedelta
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from scikeras.wrappers import KerasRegressor

# ORGANIZACIÓN DE LOS DATOS
datos = pd.read_csv('c:\\Users\\Series\\Series-2\\Series-2\\AAPL.csv', delimiter=',', header = 0, usecols =[2,4],
                    names =["Fecha", "Cierre"], dtype={"Cierre": np.float64})

datos['Fecha'] = pd.to_datetime(datos['Fecha'])
d_apple = datos.set_index("Fecha")

t_apple = d_apple['Cierre']

###############################################################################################
# ESTABILIZACIÓN DE LA VARIANZA
Valor = d_apple['Cierre'].values
transformed_data, lambda_optimo = boxcox(Valor)

logCierre=sp.stats.boxcox(d_apple['Cierre'],lmbda=-0.3660743190568889)
data = datos.assign(logCierre=logCierre)

lapple=data.set_index('Fecha')
l_apple=  lapple['logCierre']

################################################################################################
# TENDENCIA
l_apple.index = pd.to_datetime(l_apple.index)

## Convertir las fechas a segundos desde epoc
X = l_apple.index.astype('int64') // 10**9
y = l_apple.values

from sklearn.linear_model import LinearRegression
X = np.array(X).reshape(-1, 1)

#  modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Predecir
trend = model.predict(X)

# DIFERENCIACIÓN
dl_apple=l_apple.diff(periods=1).dropna()
dl_apple1=l_apple.diff(periods=1)

###############################################################################################
# Relaciones no lineales
values = pd.DataFrame(dl_apple1.values)
lags = 8
columns = [values]
for i in range(1, lags + 1):
    columns.append(values.shift(i))
dataframe = pd.concat(columns, axis=1)

columns_names = ['t'] + ['t-' + str(i) for i in range(1, lags + 1)]
dataframe.columns = columns_names
###############################################################################################
# detección de estacionalidad
dl_apple1.index = pd.to_datetime(dl_apple1.index)
dl_apple1.index.to_series().apply(lambda x:x.strftime('%Y'))
dl_apple1.index.to_series().apply(lambda x:x.strftime('%m'))

acciones = dl_apple1.values.flatten()  # Asegúrate de que sea unidimensional
fechas = dl_apple1.index.to_series().tolist()  # Convertir a lista unidimensional
years = dl_apple1.index.year.tolist()  # Convertir a lista unidimensional
months = dl_apple1.index.month_name().tolist()  # Convertir a lista unidimensional

# Crear el diccionario
d = {
    'Accion': acciones,
    'Fecha': fechas,
    'year': years,
    'mes': months,
}

df_apple = pd.DataFrame(data=d)

df_apple['mes'] = pd.Categorical(df_apple['mes'],
                            categories=['January', 'February', 'March', 'April', 'May', 'June',                   'July', 'August', 'September', 'October', 'November', 'December'],       ordered=True)

mediaspormes_apple = df_apple.groupby('mes', observed=False)['Accion'].mean()

df_mediaspormes_apple = pd.DataFrame({
    'Accion': mediaspormes_apple.values,
    'mes': mediaspormes_apple.index
})

sdpormes_apple=df_apple.groupby('mes', observed=False)['Accion'].std()

df_sdpormes_apple = pd.DataFrame({
    'Accion': sdpormes_apple.values,
    'mes': sdpormes_apple.index
})

df_sdpormes_apple1=pd.DataFrame(data=df_sdpormes_apple)

from sorted_months_weekdays import *
from sort_dataframeby_monthorweek import *

Ordenado_medias_apple=Sort_Dataframeby_Month(df = df_mediaspormes_apple,  monthcolumnname='mes')
Ordenado_sd_apple=Sort_Dataframeby_Month(df = df_sdpormes_apple1,  monthcolumnname='mes')

df=dl_apple1.reset_index()

df['year'] = [d.year for d in df['Fecha']]
df['month'] = [d.strftime('%b') for d in df['Fecha']]
df['week'] = df['Fecha'].dt.isocalendar().week
years = df['year'].unique()
df.columns
df_nuevo=df.rename(columns={'Fecha':'Fecha',0:'Cierre'})
###############################################################################################
# Suavizamiento Exponencial
datos['Fecha'] = pd.to_datetime(datos['Fecha'])
datos.set_index('Fecha', inplace=True)

Valor = datos['Cierre'].values
transformed_data, lambda_optimo = sp.stats.boxcox(Valor)
datos['logCierre'] = transformed_data

# Preparar datos para Holt-Winters
b_apple = data['logCierre']

h = 1  # Horizon de pronóstico
train_size = int(len(b_apple) * 0.85)
test_size = len(b_apple) - train_size

# Crear conjuntos de entrenamiento y prueba
train = b_apple[:train_size]
test = b_apple[train_size:]

# Crear matriz para almacenar pronósticos
forecast_steps = np.zeros((test_size, h))

# Grilla de parámetros
alpha_values = np.arange(0.001, 1.0, 0.1)
beta_values = np.arange(0.001, 1.0, 0.1)
parameter_grid = product(alpha_values, beta_values)

def fit_model(train_data, alpha=None, beta=None):
    model = ExponentialSmoothing(
        train_data,
        trend='add'
    )
    return model.fit(
        smoothing_level=alpha,
        smoothing_trend=beta,
        optimized=False
    )

def forecast_with_model(model, steps):
    return model.forecast(steps=steps)

def calculate_rmse(forecast_steps, actual_values):
    errors = np.array([actual - forecast for actual, forecast in zip(actual_values, forecast_steps)])
    mse = np.mean(errors**2, axis=0)
    rmse = np.sqrt(mse)
    return rmse

best_params = None
best_rmse = float('inf')

for alpha, beta in parameter_grid:
    rolling_errors = []
    for i in range(test_size):
        rolling_train = train[:train_size + i]
        fitted_model = fit_model(rolling_train, alpha, beta)
        forecast = forecast_with_model(fitted_model, h)
        forecast_steps[i, :] = forecast
        actual = test[i:i+h]
        rolling_errors.append(actual - forecast_steps[i, :])

    rolling_errors = np.array(rolling_errors)
    forecast_original = forecast_steps  # Ya están en la escala de Box-Cox
    actual_original = test.values.reshape(-1, 1)

    rmseexpo = calculate_rmse(forecast_original, actual_original)

    if rmseexpo < best_rmse:
        best_rmse = rmseexpo
        best_params = (alpha, beta)

def inv_boxcox(transformed_data, lambda_boxcox):
    if lambda_boxcox == 0:
        return np.exp(transformed_data)
    else:
        return np.exp(np.log(1+(lambda_boxcox * transformed_data)) / lambda_boxcox)

# Aplicar la inversión de Box-Cox a los pronósticos y datos reales
forecast_original = inv_boxcox(forecast_steps, lambda_optimo)
actual_original = inv_boxcox(test.values, lambda_optimo)

mse_se = mean_squared_error(actual_original, forecast_original)

fit1 = ExponentialSmoothing(t_apple, trend='add', initialization_method="estimated", use_boxcox=True).fit(
    smoothing_level= best_params[0],  #alpha 0.91
    smoothing_trend= best_params[1],  #beta 0.91
)

alpha_sl = best_params[0]  #alpha 0.91
beta_sl = best_params[1]  #beta 0.91

fcast1 = fit1.forecast(20).rename("Pronósticos")
start_date = datos.index[-1] + pd.DateOffset(days=1)
end_date = datos.index[-1] + pd.DateOffset(days=20)
fcast_index = pd.date_range(start=start_date, end=end_date, freq='D')

#  índice  fcast1
fcast1.index = fcast_index
###############################################################################################
#####################Arboles de desición
t_dlapple_v = pd.DataFrame(dl_apple.values, index=dl_apple.index, columns=['logCierre']).dropna()

var_rez = pd.DataFrame()
for i in range(30, 0, -1):
    var_rez['t-' + str(i)] = t_dlapple_v.shift(i)

for i in range(180, 120, -1):
    var_rez['t-' + str(i)] = t_dlapple_v.shift(i)

var_rez['t'] = t_dlapple_v.values
var_rezl = var_rez[180:]

#división de los datos
dlapple_split = var_rezl.values
X1= dlapple_split[:, 0:-1]
y1 =dlapple_split[:,-1]

Y1 = y1

traintarget_size = int(len(Y1) * 0.70)
valtarget_size = int(len(Y1) * 0.10)# Set split
testtarget_size = int(len(Y1) * 0.20)# Set split

Y1 = y1
traintarget_size = int(len(Y1) * 0.70)
valtarget_size = int(len(Y1) * 0.10)+1# Set split
testtarget_size = int(len(Y1) * 0.20)+1# Set split
train_target, val_target,test_target = Y1[0:traintarget_size],Y1[(traintarget_size):(traintarget_size+valtarget_size)] ,Y1[(traintarget_size+valtarget_size):len(Y1)]

trainfeature_size = int(len(X1) * 0.70)
valfeature_size = int(len(X1) * 0.10)+1# Set split
testfeature_size = int(len(X1) * 0.20)# Set split
train_feature, val_feature,test_feature = X1[0:traintarget_size],X1[(traintarget_size):(traintarget_size+valtarget_size)] ,X1[(traintarget_size+valtarget_size):len(Y1)]

# Arboles 1
best_depth = None
min_rmse = float('inf')

for d in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    # Crear el árbol y ajustarlo
    decision_tree_PM25 = DecisionTreeRegressor(max_depth=d)
    decision_tree_PM25.fit(train_feature, train_target)

    # Calcular el RECM sobre el conjunto de validación
    rmse = sklearn.metrics.mean_squared_error(decision_tree_PM25.predict(val_feature), val_target, squared=False)

    # Verificar si este RECM es el más bajo encontrado
    if rmse < min_rmse:
        min_rmse = rmse
        best_depth = d

train_val_feature=np.concatenate((train_feature,val_feature),axis=0)
train_val_target=np.concatenate((train_target,val_target),axis=0)

# Mejor Profundidad
decision_tree_PM25 = DecisionTreeRegressor(max_depth=best_depth)
decision_tree_PM25.fit(train_val_feature, train_val_target)

# Predecir
train_val_prediction = decision_tree_PM25.predict(train_val_feature)
test_prediction = decision_tree_PM25.predict(test_feature)

# Contra valores
decision_tree_PM25_prun_mincost = DecisionTreeRegressor(max_depth=best_depth)
decision_tree_PM25_prun_mincost.fit(train_val_feature, train_val_target)

train_val_prediction_prun_mincost = decision_tree_PM25.predict(train_val_feature)
test_prediction_prun_mincost = decision_tree_PM25.predict(test_feature)

indx1 = train_val_prediction.size
indicetrian_val_test=var_rezl.index
indx2 = indicetrian_val_test.size

indicetrain_val=indicetrian_val_test[0:indx1]
indicetest=indicetrian_val_test[indx1:indx2]
indicetrian_val_test

targetjoint=np.concatenate((train_val_target,test_target))
predictionjoint=np.concatenate((train_val_prediction,test_prediction))

d = {'observado': targetjoint, 'Predicción': predictionjoint}
ObsvsPred=pd.DataFrame(data=d,index=indicetrian_val_test)
ObsvsPred.index

# Transformación
inicial = l_apple.loc[indicetrian_val_test[0]]

obs_reverted = ObsvsPred['observado'].cumsum() + inicial
pred_reverted = ObsvsPred['Predicción'].cumsum() + inicial

lambda_boxcox = -0.3660743190568889

def inv_boxcox_manual(transformed_data, lambda_boxcox):
    if lambda_boxcox < 0:
        return np.exp(np.log(1+(lambda_boxcox * transformed_data)) / lambda_boxcox)
    else:
        raise ValueError("This method is intended for negative lambda values.")

# Aplica la inversión manual
obs_reverted1 = inv_boxcox_manual(obs_reverted, lambda_boxcox)
pred_reverted1 = inv_boxcox_manual(pred_reverted, lambda_boxcox)

mse_a2 = mean_squared_error(obs_reverted1,pred_reverted1)
###############################################################################################
# Redes Neuronales Multicapa
d_dlapple_v = pd.DataFrame(datos['Cierre'].values,index=datos.index)
# Entrenamiento Validación y prueba
column_indices = {name: i for i, name in enumerate(d_dlapple_v.columns)}

n = len(d_dlapple_v)
train_apple = d_dlapple_v[0:int(n*0.7)]
val_apple = d_dlapple_v[int(n*0.7):int(n*0.9)]
test_apple = d_dlapple_v[int(n*0.9):]

num_features = d_dlapple_v.shape[1]

# Normalización
train_mean = train_apple.mean()
train_std = train_apple.std()

train_df = (train_apple - train_mean) / train_std
val_df = (val_apple - train_mean) / train_std
test_df = (test_apple - train_mean) / train_std

# Variables rezagadas
df1 = DataFrame()

for i in range(30,0,-1):
    df1[['t-'+str(i)]] = d_dlapple_v.shift(i)

for i in range(180,120,-1):
    df1[['t-'+str(i)]] = d_dlapple_v.shift(i)

df1['t'] = d_dlapple_v.values
df1_apple = df1[180:]

# Divisió de covariables
APsplit = df1_apple.values

X1= APsplit[:, 0:-1]
y1 =APsplit[:,-1]

X_train_full, X_test, y_train_full, y_test=train_test_split(X1,y1,test_size=0.1, train_size=0.9,shuffle=False)
X_train, X_val, y_train, y_val=train_test_split(X_train_full,y_train_full,test_size=0.2, train_size=0.8,shuffle=False)

# Normalización de Covariables
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_val=scaler.transform(X_val)
X_test=scaler.transform(X_test)

# funciones
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int("num_units", min_value=32, max_value=64, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"])))
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 2)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=64, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(1, activation="linear"))
    learning_rate = hp.Float("lr", min_value=1e-3, max_value=1e-2,step=0.003)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mean_squared_error"]
    )
    return model


build_model(kt.HyperParameters())

tuner_GridSearch_mlp = kt.GridSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=None,
    executions_per_trial=1,
    overwrite=True,
    directory="dirsalida",
    project_name="helloworld",
)

stop_early=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=2)

try:
    tuner_GridSearch_mlp.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[stop_early])
except UnicodeDecodeError as e:
    print(f"Error de codificación: {e}")
    print("Asegúrate de que todos los archivos y configuraciones estén usando la codificación correcta.")
finally:
    print("Continuando con la ejecución del resto del código...")

# Mejores 2 Modelos
# Obtener los 2 mejores modelos.
models_mlp = tuner_GridSearch_mlp.get_best_models(num_models=2)
best_model_mlp = models_mlp[0]

# No necesitamos llamar a `build` si el modelo ya está configurado.
# Verificamos si el modelo ya tiene una forma de entrada configurada.
if not hasattr(best_model_mlp, 'built') or not best_model_mlp.built:
    # Solo si el modelo no está construido, lo configuramos.
    best_model_mlp.build(input_shape=(32, 1, 90))

best_model_mlp.summary()

# Concatenar los conjuntos de entrenamiento y validación para entrenar con el mejor modelo.
x_all = np.concatenate((X_train, X_val))
y_all = np.concatenate((y_train, y_val))

# Crear una callback para detener el entrenamiento temprano si la pérdida no mejora.
callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

# Obtener los mejores hiperparámetros para construir el modelo.
best_hps_mlp = tuner_GridSearch_mlp.get_best_hyperparameters(2)

# Construir el modelo con los mejores hiperparámetros.
model_mlp = build_model(best_hps_mlp[0])

# Ajustar el modelo con todo el conjunto de datos.
model_mlp.fit(x_all, y_all, epochs=50, callbacks=[callback])

# Realizar predicciones sobre el conjunto de prueba.
prediction_test = model_mlp.predict(X_test, verbose=1)

# Asegurarse de que y_test tenga la forma correcta.
y_test = y_test.reshape((y_test.shape[0], 1))

errors_squared  = np.mean(np.square(y_test - prediction_test))
print ("ECM :", errors_squared )
# Asignar el ECM a una variable si lo necesitas para otros cálculos.
ecm_rnm = errors_squared

###############################################################################################
# red neuronal recurrente
use_features = ['Cierre']
target = ['Cierre']
n_steps_ahead = 1

# Modelo Autoregresivo
n_steps = 1

# Entrenamiento Validación y prueba
train_weight = 0.8
split = int(len(d_apple) * train_weight)

df_train = d_apple[use_features].iloc[:split]
df_test = d_apple[use_features].iloc[split:]

# Estandarización
mu = float(df_train.mean())
sigma = float(df_train.std())

stdize_input = lambda x: (x - mu) / sigma

df_train = df_train.apply(stdize_input)
df_test = df_test.apply(stdize_input)

# Definir Funciones
def get_lagged_features(df, n_steps, n_steps_ahead):
    """
    df: pandas DataFrame of time series to be lagged
    n_steps: number of lags, i.e. sequence length
    n_steps_ahead: forecasting horizon
    """
    lag_list = []

    for lag in range(n_steps + n_steps_ahead - 1, n_steps_ahead - 1, -1):
        lag_list.append(df.shift(lag))
    lag_array = np.dstack([i[n_steps+n_steps_ahead-1:] for i in lag_list])
    # We swap the last two dimensions so each slice along the first dimension
    # is the same shape as the corresponding segment of the input time series
    lag_array = np.swapaxes(lag_array, 1, -1)
    return lag_array

x_train = get_lagged_features(df_train, n_steps, n_steps_ahead)
Y_train =  df_train.values[n_steps + n_steps_ahead - 1:]
Y_train_timestamps = df_train.index[n_steps + n_steps_ahead - 1:]

x_test = get_lagged_features(df_test, n_steps, n_steps_ahead)
Y_test =  df_test.values[n_steps + n_steps_ahead - 1:]
Y_test_timestamps = df_test.index[n_steps + n_steps_ahead - 1:]

# Modelo SimpleRNN
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras

# Función del modelo SimpleRNN

def SimpleRNN_(n_units=10, l1_reg=0.0, seed=0):
    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(n_units,
        activation='tanh',
        kernel_initializer=keras.initializers.glorot_uniform(seed),
        bias_initializer=keras.initializers.glorot_uniform(seed),
        recurrent_initializer=keras.initializers.orthogonal(seed),
        kernel_regularizer=keras.regularizers.l1(l1_reg),
        input_shape=(x_train.shape[1], x_train.shape[-1]),
        unroll=True, stateful=False))
    model.add(keras.layers.Dense(1,
        kernel_initializer=keras.initializers.glorot_uniform(seed),
        bias_initializer=keras.initializers.glorot_uniform(seed),
        kernel_regularizer=keras.regularizers.l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def GRU_(n_units=10, l1_reg=0.0, seed=0):
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(n_units,
        activation='tanh',
        kernel_initializer=keras.initializers.glorot_uniform(seed),
        bias_initializer=keras.initializers.glorot_uniform(seed),
        recurrent_initializer=keras.initializers.orthogonal(seed),
        kernel_regularizer=keras.regularizers.l1(l1_reg),
        input_shape=(x_train.shape[1], x_train.shape[-1]),
        unroll=True, stateful=False))
    model.add(keras.layers.Dense(1,
        kernel_initializer=keras.initializers.glorot_uniform(seed),
        bias_initializer=keras.initializers.glorot_uniform(seed),
        kernel_regularizer=keras.regularizers.l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def LSTM_(n_units=10, l1_reg=0.0, seed=0):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(n_units,
        activation='tanh',
        kernel_initializer=keras.initializers.glorot_uniform(seed),
        bias_initializer=keras.initializers.glorot_uniform(seed),
        recurrent_initializer=keras.initializers.orthogonal(seed),
        kernel_regularizer=keras.regularizers.l1(l1_reg),
        input_shape=(x_train.shape[1], x_train.shape[-1]),
        unroll=True, stateful=False))
    model.add(keras.layers.Dense(1,
        kernel_initializer=keras.initializers.glorot_uniform(seed),
        bias_initializer=keras.initializers.glorot_uniform(seed),
        kernel_regularizer=keras.regularizers.l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Función para crear el KerasRegressor
def create_keras_regressor(model_function, n_units=10, l1_reg=0.0):
    return KerasRegressor(build_fn=model_function, n_units=n_units, l1_reg=l1_reg, epochs=100, batch_size=500)

es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10, min_delta=1e-7, restore_best_weights=True)

# Definición de parámetros
params = {
    'rnn': {'function': SimpleRNN_, 'H': 10, 'l1_reg': 0.01, 'label': 'RNN', 'color': 'blue'},
    'gru': {'function': GRU_, 'H': 10, 'l1_reg': 0.01, 'label': 'GRU', 'color': 'orange'},
    'lstm': {'function': LSTM_, 'H': 10, 'l1_reg': 0.01, 'label': 'LSTM', 'color': 'green'},
    # Agregar otros modelos según sea necesario...
}
# Validación cruzada
do_training = True
cross_val =False  # WARNING: Esto tomará muchas horas para ejecutarse

if do_training and cross_val:
    n_units = [5, 10, 20]
    l1_reg = [0, 0.001, 0.01, 0.1]

    param_grid = {'n_units': n_units, 'l1_reg': l1_reg}

    tscv = TimeSeriesSplit(n_splits=5)

    for key in params.keys():
        print('Realizando validación cruzada. Modelo:', key)

        # Crear el KerasRegressor aquí
        model = create_keras_regressor(params[key]['function'])

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=tscv, n_jobs=1, verbose=2)
        grid_result = grid.fit(x_train, Y_train, callbacks=[es])
        print("Mejor: %f usando %s" % (grid_result.best_score_, grid_result.best_params_))

# Validación cruzada entrenamiento
if do_training:
    for key in params.keys():
        tf.random.set_seed(0)
        print('Entrenando el modelo', key)

        # Crear el modelo con los mejores parámetros
        model = params[key]['function'](params[key]['H'], params[key]['l1_reg'])
        model.fit(x_train, Y_train, epochs=100, batch_size=500, callbacks=[es], shuffle=False)
# Predicción
for key in params.keys():
    # Crear el modelo con los mejores parámetros
    model = create_keras_regressor(params[key]['function'], params[key]['H'], params[key]['l1_reg'])

    # Entrenar el modelo
    model.fit(x_train, Y_train, epochs=100, batch_size=500, shuffle=False)

    # Almacenar el modelo en el diccionario
    params[key]['model'] = model


    # Predicción
    params[key]['pred_train'] = model.predict(x_train, verbose=1)
    params[key]['pred_test'] = model.predict(x_test, verbose=1)

    # Calcular MSE para entrenamiento y prueba
    mse_train = mean_squared_error(Y_train, params[key]['pred_train'])
    mse_test = mean_squared_error(Y_test, params[key]['pred_test'])

    # Almacenar los valores de MSE en el diccionario
    params[key]['MSE_train'] = mse_train
    params[key]['MSE_test'] = mse_test

###############################################################################################
# ARIMA

def read_order_value():
    with open('order_b.txt', 'r') as file:
        return file.read()

def encode_image(image_file):
    with open(image_file, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('ascii')
    return f"data:image/png;base64,{encoded}"

def read_coeftest_output():
    with open('coeftest_output.txt', 'r') as file:
        return file.read()

def get_image(image_path):
    with open(image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('ascii')
    return f"data:image/png;base64,{encoded_image}"  

def read_outlier_table():
    return pd.read_csv('outlier_table.txt')

def read_outlier_table2():
    return pd.read_csv('outlier_table2.txt')

def read_normalidad_output():
    with open('norma.txt', 'r') as file:
        return file.read()

def read_ecm_value():
    with open('ecm.txt', 'r') as file:
        return file.read()
    
mse_ar = 411221.5
###############################################################################################
#
r_script_path = r'C:\\Users\\VALERIA\\Documents\\DANA\\Universidad\\UNAL\\VIII SEMESTRE\\Series\\Series-2\\Corto.R'
command = [r"C:\\Program Files\\R\\R-4.3.1\\bin\\Rscript.exe", r_script_path] 

try:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    
    # Imprimir la salida de R (stdout)
    print("Salida del script R:")
    print(result.stdout)
    
    # Imprimir posibles errores (stderr)
    if result.stderr:
        print("Errores del script R:")
        print(result.stderr)
    
except subprocess.CalledProcessError as e:
    print(f"Error al ejecutar el script R: {e}")

coeftest_output = read_coeftest_output()
residuals_image = get_image('residuales3_plot.png')
outlier_table = read_outlier_table()
outlier_table2 = read_outlier_table2()
res3_image = get_image('resi_3.png')
res32_image = get_image('acf_resi2.png')
norma1 = read_normalidad_output()
pronosticos = get_image('pronost_plot.png')
###############################################################################################
# APLICACIÓN
app = dash.Dash(__name__, suppress_callback_exceptions=True)


# GRAFICAS
#################################################################################################
## Grafica 1 de la serie original
fig = px.line(x=datos.index, y=datos['Cierre'].values, title='Cierre acciones de Apple')
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(title_x=0.5,
                xaxis_title='Fecha',
                yaxis_title='Cierre')

#################################################################################################
## Grafica con box Cox
fig2 = px.line(l_apple, y='logCierre', title='Serie con Tranformación Box-Cox')
fig2.update_layout(title_x=0.5)

##################################################################################################
# Comparación grafica 1 y 2
fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Serie Original', 'Serie con Transformación Box-Cox'))

fig3.add_trace(
    go.Scatter(x=d_apple.index, y=d_apple['Cierre'], mode='lines', name='Serie Original', line=dict(color='red')),
    row=1, col=1
)

# Gráfico de la serie transformada
fig3.add_trace(
    go.Scatter(x=lapple.index, y=lapple['logCierre'], mode='lines', name='Serie Transformada'),
    row=1, col=2
)

fig3.update_layout(title_text="Comparación de Series", showlegend=False,
                title_x=0.5)
##################################################################################################
# Tendencia
fig4 =go.Figure()
fig4.add_trace(go.Scatter(x=l_apple.index, y=y, mode='lines', name='Cierre de la acción de APPLE'))
fig4.add_trace(go.Scatter(x=l_apple.index, y=trend, mode='lines', name='Tendencia (Regresión Lineal)', line=dict(color='red', dash='dash')))
fig4.update_layout(
    title='Regresión Lineal de cierre de acciones APPLE',
    xaxis_title='Fecha',
    yaxis_title='logCierre',
    legend=dict(y=1.0, x=1.0),
    title_x=0.5
)

fig5 = go.Figure([go.Scatter(x=dl_apple.index, y=dl_apple.values, mode='lines', name='Serie Diferenciada')])
fig5.update_layout(
    title='Serie diferenciada',
xaxis_title='Fecha',
yaxis_title='logCierre',
legend=dict(y=1.0, x=1.0),
title_x=0.5
)
##################################################################################################
# Relaciones no lineales
plt.figure(figsize=(12, 8))
for i in range(1, lags + 1):
    ax = plt.subplot(240 + i)
    ax.set_title('t vs t-' + str(i))
    sns.regplot(x=dataframe['t'].values, y=dataframe['t-' + str(i)].values, color="black", lowess=True,
                line_kws={"color": "magenta", "linewidth": 5})

# Convertir la figura en una imagen en base64
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img_nolin = base64.b64encode(buf.read()).decode('utf-8')
plt.close()
##################################################################################################
# ACF Y PACF
fig7, ax = plt.subplots()
dl_apple_values = dl_apple.values
plot_acf(dl_apple_values,adjusted=False,lags=48,title='ACF Serie Diferenciada')

buf1 = io.BytesIO()
plt.savefig(buf1, format='png')
buf1.seek(0)
plt.close(fig7)

img_acf = base64.b64encode(buf1.getvalue()).decode('utf-8')

fig8, ax = plt.subplots()
dl_apple_values = dl_apple.values
plot_pacf(dl_apple_values,lags=100,method='ywm',alpha=0.01,title='ACF Serie Diferenciada')

buf2 = io.BytesIO()
plt.savefig(buf2, format='png')
buf2.seek(0)
plt.close(fig8)

img_pacf = base64.b64encode(buf2.getvalue()).decode('utf-8')
##################################################################################################
# Suavizamiento Exponencial
fig11, axes = plt.subplots(4, 1, figsize=(10, 12))
axes[0].plot(t_apple)
axes[0].set_ylabel('Cierre Acciones')
axes[1].plot(fit1.level)
axes[1].set_ylabel('level')
axes[2].plot(fit1.trend)
axes[2].set_ylabel('trend')
axes[3].plot(fit1.resid)
axes[3].set_ylabel('resid')

# Guardar el gráfico en un buffer de memoria
buf3 = io.BytesIO()
plt.savefig(buf3, format='png')
plt.close(fig11)
buf3.seek(0)

# Convertir la imagen a base64
holwint = base64.b64encode(buf3.getvalue()).decode('utf-8')

#### Pronosticos
fig12 = go.Figure()

# Línea de t_apple
fig12.add_trace(go.Scatter(
    x=t_apple.index,
    y=t_apple,
    mode='lines+markers',
    name='Cierre',
    line=dict(color='black'),
    marker=dict(symbol='diamond', size=3)
))

# Pronóstico fcast1
fig12.add_trace(go.Scatter(
    x=fcast1.index,
    y=fcast1,
    mode='lines+markers',
    name='Pronóstico',
    line=dict(color='blue'),
    marker=dict(symbol='diamond', size=3)
))

# Configuración del diseño
fig12.update_layout(
    title="Pronósticos usando Holt-Winters",
    xaxis_title="Fecha",
    yaxis_title="Valores",
    legend=dict(x=0.01, y=0.99, traceorder='normal'),
    showlegend=True
)

# Agregar cuadrícula
fig12.update_xaxes(showgrid=True)
fig12.update_yaxes(showgrid=True)

##################################################################################################
# Arboles

fig9 = go.Figure()
fig9.add_trace(go.Scatter(x=ObsvsPred.index, y=ObsvsPred['observado'], mode='lines', name='Observado'))
fig9.add_trace(go.Scatter(x=ObsvsPred.index, y=ObsvsPred['Predicción'], mode='lines', name='Predicción'))

fig9.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Valores"
)

# Transformada
fig10 = go.Figure()

fig10.add_trace(go.Scatter(x=ObsvsPred.index, y=obs_reverted1, mode='lines+markers',
                        name='Observado',marker=dict(symbol='diamond', size=3)))
fig10.add_trace(go.Scatter(x=ObsvsPred.index, y=pred_reverted1, mode='lines+markers', name='Predicción',marker=dict(symbol='diamond', size=3)))

fig10.update_layout(
    title="Observado vs Predicción (Escala Original)",
    xaxis_title="Fecha",
    yaxis_title="Valor",
    title_x = 0.5
)
##################################################################################################
# red multicapa
plt.figure()
plt.plot(y_test, label='Respuesta real')
plt.plot(prediction_test, label='Predicción de la Respuesta')
plt.legend(loc="best", fontsize=12)
plt.ylabel('Y y $\hat{Y}$ en conjunto de prueba', fontsize=14)

# Guardar la gráfica en un buffer en memoria
buf4 = io.BytesIO()
plt.savefig(buf4, format='png')
buf4.seek(0)

# Convertir la imagen a formato base64
red_pred = base64.b64encode(buf4.getvalue()).decode('utf-8')
buf4.close()

# grafico del error
plt.figure()
plt.plot(prediction_test - y_test, label='Error de predicción')
plt.legend(loc="best", fontsize=12)
plt.ylabel('$\hat{e}$ en conjunto de prueba', fontsize=14)

# Guardar la gráfica en un buffer en memoria
buf5 = io.BytesIO()
plt.savefig(buf5, format='png')
buf5.seek(0)

# Convertir la imagen a formato base64
redm_err = base64.b64encode(buf5.getvalue()).decode('utf-8')
buf5.close()

##################################################################################################
# rendimiento
max_pts = 10**4
compare = params.keys()  # Por ejemplo: ['rnn', 'alpharnn'] o ['lstm']
l, u = (None, None)  # Índices inferiores y superiores del rango a graficar
ds = max(1, len(Y_train[l:u])//max_pts)  # Proporción de submuestreo

# Crear la figura con Matplotlib
fig13 = plt.figure(figsize=(15, 8))
x_vals = Y_test_timestamps[l:u:ds]

for key in compare:
    y_vals = params[key]['pred_test'][l:u:ds]
    label = params[key]['label'] + ' (test MSE: %.2e)' % params[key]['MSE_test']
    #label = params[key]['label'] + ' (test MSE: %.2e)' % params[key]['MSE_test']
    plt.plot(x_vals, y_vals, c=params[key]['color'], label=label, lw=1)

plt.plot(x_vals, Y_test[l:u:ds], c="black", label="Observed", lw=1)
start, end = x_vals.min(), x_vals.max()
xticks = [start.date() + timedelta(days=(1 + i)) for i in range(1 + (end - start).days)]
xticks = xticks[::max(1, len(xticks)//30)]
for t in xticks:
    plt.axvline(x=t, c='gray', linewidth=0.5, zorder=0)

plt.xticks(xticks, rotation=70)
plt.xlim(start, end)
plt.ylabel('$\hat{Y}$', rotation=0, fontsize=14)
plt.legend(loc="best", fontsize=12)
plt.title('Observed vs Model Outputs (Testing)', fontsize=16)

# Convertir el gráfico a una imagen en formato base64
img_bytes = io.BytesIO()
plt.savefig(img_bytes, format='png')
img_bytes.seek(0)
rnr_pred = base64.b64encode(img_bytes.read()).decode('utf-8')
plt.close(fig13)

mse_results = {}
for key in compare:
    mse_results[key] = params[key]['MSE_test']

mse_rn1 = mse_results['rnn']
mse_rn2 = mse_results['gru']
mse_rn3 = mse_results['lstm']

###############################################################################################
# errores

dat_err = {
    'Categoría': ['Suavizamiento Exponencial', 'Arboles de Decisión', 'Red Multicapa', 'RNN','GRU','LSTM',"ARIMA"],
    'Valor': [mse_se, mse_a2, ecm_rnm, mse_rn1, mse_rn2, mse_rn3, mse_ar]
}

df_dater = pd.DataFrame(dat_err)

table1 = dash_table.DataTable(
    data=df_dater.to_dict('records'),
    columns=[{"name": i, "id": i} for i in df_dater.columns],
    style_table={'width': '50%'},
    style_cell={'textAlign': 'center'},
)

fig14 = px.bar(df_dater, x='Categoría', y='Valor', title='Error Cuadratico Medio')
##################################################################################################
# Variables a texto
lambda_optimo_str = f"{lambda_optimo:.4f}"
best_depth_str =f"{best_depth:4f}"
mse_a2_str = f"{mse_a2:4f}"
mse_se_str = f"{mse_se:4f}"
ecm_rnm_str = f"{ecm_rnm:4f}"
mse_rn1_str = f"{mse_rn1:4f}"
mse_rn2_str = f"{mse_rn2:4f}"
mse_rn3_str = f"{mse_rn3:4f}"
alpha_sl_str = f"{alpha_sl:4f}"
beta_sl_str = f"{beta_sl:4f}"
################################################################################################
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='cat1', children=[
        dcc.Tab(label='Análisis descriptivo', value='cat1'),
        dcc.Tab(label='Suavizamiento exponencial', value='cat2'),
        dcc.Tab(label='Arboles de decisión', value='cat3'),
        dcc.Tab(label='Redes Neuronales', value='cat4'),
        dcc.Tab(label='ARIMA', value='cat6'),
        dcc.Tab(label='ECM de los Modelos', value='cat5')
    ]),
    html.Div(id='sub-category-container')
])

@app.callback(
    Output('sub-category-container', 'children'),
    Input('tabs', 'value')
)
def update_sub_category(selected_tab):
    if selected_tab == 'cat1':
        return dcc.Tabs([
            dcc.Tab(label='Serie de Acciones', children=html.Div([
                html.P('''Cierre de las acciones de APPLE desde el 27 de Mayo de 2015 (2015-05-27) 
                    al 10 de Mayo de 2020 (2020-05-10), la serie no tiene observaciones los
                    fines de semana. Del grafico podemos observar que se tiene tendencia que
                    podría ser lineal además de que su varianza va aumentando a traves del tiempo,
                    no parece tener estacionalidad.'''),
                dcc.Graph( id='example-graph1',
            figure=fig
                )
                                                                ])),
            dcc.Tab(label='Estabilización de Varianza', children=html.Div([
                html.P(f'''Para estabilizar la varianza tenemos en cuenta el valor de lambda: {lambda_optimo_str}. 
                    Dado que no es 1, se decide aplicar la transformación de Box-Cox.'''),
                dcc.Graph(
                id='example-graph2',
                figure=fig2
                ),
                html.P('''Tras haber hecho la transformación de BoxCox, parece haber una reducción de los
                    picos más altos y las caidas más grandes de la serie además del cambio
                    significativo en su escala'''),
                dcc.Graph(
                id='example-graph3',
                figure=fig3
                )
            ])),
            dcc.Tab(label='Tendencia', children=html.Div([
                html.P('''Con la serie transformada podemos
                        ver una tendencia lineal'''),
                dcc.Graph( id='Example-grahp4',   figure=fig4
                ),
            html.P('''Para eliminar la tendencia haremos uso de la diferencia ordinaria'''),
            dcc.Graph(
                id='Example-grahp5',
                figure=fig5
            )
            ])),
            dcc.Tab(label='Relaciones no lineales', children=html.Div([
                html.P('''Dado los primeros 8 retardos no se ven relaciones significativas respecto a la primera observación.'''),
                html.Img(src=f'data:image/png;base64,{img_nolin}', style={'width': '90%', 'height': 'auto','display': 'block', 'margin': 'auto'})
            ])),
            dcc.Tab(label='Autocorrelación y Autocorrelación Parcial', children=html.Div([
                html.P('''En cuanto a la grafica de autocorrelación no se evidencia algún patrón marcado, podemos observar que un par de observaciones con algún grado de correlación, pero no parece ser demasiado significativo. '''),
html.Img(src=f'data:image/png;base64,{img_acf}', style={'width': '80%', 'height': 'auto','display': 'block', 'margin': 'auto'}), html.Img(src=f'data:image/png;base64,{img_pacf}', style={'width': '80%', 'height': 'auto','display': 'block', 'margin': 'auto'})
            ])),
            dcc.Tab(label='Detección de estacionalidad', children=html.Div([
                html.P('''Realizamos un acercamiento a la detección de la estacionalidad.
                    El grafico muestra el comportamiento medio de las acciones por mes y año.'''),
                dcc.Dropdown(
                    id='graph-dropdown',
                    options=[
                        {'label': 'Box Plot por Año', 'value': 'year'},
                        {'label': 'Box Plot por Semana', 'value': 'week'},
                        {'label': 'Box Plot por Mes', 'value': 'month'}
                    ],
                    value='year'  # Valor por defecto
                ),
                dcc.Graph(id='boxplot-graph'),
                html.P('''De los anteriores graficos se observa que no hay una diferencia significativa de
medias en ninguno de los casos mencionados.''')
            ]))
        ])
    elif selected_tab == 'cat2':
        return dcc.Tabs([
            html.Div([
                html.P('''En el caso de Suavizamiento exponencial vamos a hacer uso de Holt Winter.
                Además teniendo en cuenta que no encontramos alguna componente estacional, la
                componente "seasonal" no será tenida en cuenta'''),
                html.Img(src=f'data:image/png;base64,{holwint}', style={'display': 'block', 'margin': 'auto'}),
                html.P(f'''Haciendo Rolling para encontrar los mejores parametros del modelo anterior
                    tenemos un error cuadratico medio de:{mse_se_str} con parametros de tendencia y nivel de
                    :{alpha_sl_str} y {beta_sl_str}'''),
                html.P('''Se hace el pronostico a 20 días'''),
                dcc.Graph(
                    id='example-graph12',
                    figure=fig12
                )
            ])
        ])
    elif selected_tab == 'cat3':
        return dcc.Tabs([
            html.Div([
                html.P('''Para trabajar con arboles de decisión se va usar para el modelamiento la serie
                    diferenciada, posteriormente la serie volverá a su escala original'''),
                dcc.Graph(
                id='Example-grahp5',
                figure=fig5
            ),
            html.P('''Para hallar los retardos nos fijaremos principarlmente de las funciones de Autocorrelación
                y autocoreelación parcial. Debido a que la serie no tiene una componente estacional se
                evaluaron diversos rezagos hasta dar con los más optimos que incluyen el primer mes y
                el periodo dentro de 4 y 6 meses'''),
            html.P(f'''Ajustanto un modelo de arboles de decisión, dada la mejor profundidad de {best_depth_str},
                haciendo predicción un paso adelante tenemos las siguientes predicciones'''),
                dcc.Graph(
                id='Example-grahp9',
                figure=fig9
            ),
            html.P('''Se realiza una nueva transformación para utilizar la escala original de la serie.
                Así, los valores predichos por el modelo de arboles se muestran en el grafico'''),
                dcc.Graph(
                    id='Example-grahp10',
                    figure=fig10
                ),
            html.P(f'''El modelo de arboles presentado tiene un error cuadratico medio de:
                y {mse_a2_str}''')
            ])
        ])
    elif selected_tab == 'cat4':
        return dcc.Tabs([
            dcc.Tab(label='Red Neuronal Multicapa', children=html.Div([
                html.P(children=['''Se realiza una busqueda de hiperparametros para encontrar la red
                    con mejor predicción. De la busqueda de mejores parametros, nos quedamos
                    con un modelo con función de activación ''', html.I("relu"), ''' con 2 capas ocultas, con 32 nodos
                    la primera y 64 la segunda''']),
                html.Img(src=f'data:image/png;base64,{red_pred}', style={'width': '90%', 'height': 'auto','display': 'block', 'margin': 'auto'}),
                html.P(f'''Donde se tiene un error cuadratico medio de:{ecm_rnm_str}'''),
                html.Img(src=f'data:image/png;base64,{redm_err}', style={'width': '90%', 'height': 'auto','display': 'block', 'margin': 'auto'})
            ])
        ),
            dcc.Tab(label='Red Neuronales Recurrentes', children=html.Div([
                html.P(''' Del conjunto de datos para entrenamiento Se usa el 80%, mientras por otro lado se usa el 
                20% para prueba, tenga en cuenta que no es una base de datos muy extensa'''),
                html.Img(src=f'data:image/png;base64,{rnr_pred}', style={'width': '90%', 'height': 'auto','display': 'block', 'margin': 'auto'}),
                html.P(f'''Para los modelos mostrados se tienen los siguientes errores cuadraticos medios:
                        Para RNN: {mse_rn1_str}, GRU:{mse_rn2_str}, LSTM:{mse_rn3_str}''')
            ]))
        ])
    elif selected_tab == 'cat6':
        return dcc.Tabs([
            html.Div([
                html.H1("Ajuste del Modelo"),
                html.P('''Para determinar si se va a diferenciar la serie se trabaja con la serie
                transformada con BoxCox y se toman de numero de rezagos:'''),
                html.P(read_order_value()),
                html.P('''Por resultados de la prueba de Raiz unitaria, es necesario dierenciar. A partir
                       de esto se tienen los graficos de ACF y PACF tras diferenciación'''),
                html.Img(src=encode_image('acf_plot.png')),
                html.Img(src=encode_image('pacf_plot.png')),
                html.P('''Se requiere un MA(q) de orden 9 o hasta máximo 22, o puede ser un autoregresivo (p) de orden 8 - 37. Si bien 
                       se hizo una nueva revisión de prueba de raiz unitaria donde se requeria una nueva diferenciación, por 
                       sugerencia y ajuste del modelo automatico para ambas series diferenciadas 1 y 2 
                       veces por criterios de información, se va a trabajar con la serie de una solo una diferencia.'''),
                html.P('''Por el ajuste automatico se llegó a ARIMA(3,0,7). El ajuste automatico fue realizado 
                       con la serie diferenciada'''),
                html.P('''Luego se lleva acabo un ajuste y refinamiento del modelo ARIMA(3,1,7), tal que se tiene'''),
                html.Pre(coeftest_output, style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}),
                html.H1("Análisis de Outliers"),
                html.P('''Tener en cuenta que son demasiados outliers si superan el 5 o 10 porciento del total de la serie,
                       en este caso 63 o 126 outliers respectivamente'''),
                html.Img(src=residuals_image, style={'display': 'block', 'margin': 'auto'}),
                dash_table.DataTable(
            data=outlier_table.to_dict('records'),  # Mostrar la tabla en Dash
            columns=[{'name': i, 'id': i} for i in outlier_table.columns],
            style_table={'width': '50%'}
        ),
                html.P('''Detecta 67 outliers con 26 de tipo aditivo, 25 cambio de media y 16 de cambio transitorio. Dado que 
                       los outliers son menores al 10 porciento del total de la serie se hace un ajuste'''),
                html.P('''Tras realizar un reajuste ajustes con variables regresoras se vuelven a buscar outliers'''),
                dash_table.DataTable(
            data=outlier_table2.to_dict('records'),  # Mostrar la tabla en Dash
            columns=[{'name': i, 'id': i} for i in outlier_table2.columns],
            style_table={'width': '50%'}
        ),
                html.P('''Podriamos seguir modelando pero parece plausible dejar de modelar hasta este punto, si se siguen 
                       buscando outliers estos seguiran apareciendo cada vez en mayor cantidad.'''),
                html.H1("Análisis de Residuales"),
                html.Img(src=res3_image, style={'display': 'block', 'margin': 'auto'}),
                html.P('''Los residuales parecen tener media 0 y no tienen algún patrón de comportamiento notorio'''),
                html.H2("Autocorrelación"),
                html.Img(src=encode_image('acf_resi.png')),
                html.Img(src=encode_image('pacf_resi.png')),
                html.P(''' No se observan valores fuera de las bandas de confianza para los graficos'''),
                html.Img(src=res32_image, style={'display': 'block', 'margin': 'auto'}),
                html.P('''Los residuales están altamente correlacionados, lo que indica que se tiene una 
                       varianza marginal no constante, dado lo anterior, notamos que la serie es muy volatil'''),
                html.H2('''Normalidad'''),
                html.Pre(norma1, style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}),
                html.P('''Los datos no siguen una distribución normal'''),
                html.H2("CUSUM y CUSUMq"),
                html.Img(src=encode_image('cusum_plot.png')),
                html.Img(src=encode_image('cusumq_plot.png')),
                html.H1("Pronostico"),
                html.P('''Se realizó un conjunto del entrenamiento y prueba para calcular su ECM, 
                       para el modelo es de:'''),
                html.P(read_ecm_value()),
                html.P('''Se realiza un pronostico para 20 días'''),
                html.Img(src=pronosticos, style={'display': 'block', 'margin': 'auto'})
            ])
        ])
    elif selected_tab =='cat5':
        return dcc.Tabs([
            html.Div([
                html.H1("Tabla y Diagrama de ECM", style={'textAlign': 'center'}),
                table1,
                dcc.Graph(
                    id='example-grapg14',
                    figure=fig14)
            ])
        ])
#########################################################################################################
# CALL BACKS
@app.callback(
    Output('boxplot-graph', 'figure'),
    Input('graph-dropdown', 'value')
)
def update_graph(selected_value):
    if selected_value == 'year':
        fig = px.box(df_nuevo, x='year', y='logCierre',
                    title='Box Plot por Año \n(The Trend)',
                    labels={'year': 'Año', 'logCierre': 'Log Cierre de Acción'})
    elif selected_value == 'week':
        fig = px.box(df_nuevo, x='week', y='logCierre',
                    title='Box Plot por Semana \n(The Seasonality)',
                    labels={'week': 'Semana del Año', 'logCierre': 'Log Cierre de Acción'})
    elif selected_value == 'month':
        fig = px.box(df_nuevo, x='month', y='logCierre',
                    title='Box Plot por Mes \n(The Seasonality)',
                    labels={'month': 'Mes', 'logCierre': 'Log Cierre de Acción'})

    fig.update_layout(
        xaxis_title='',
        yaxis_title='',
        xaxis_tickangle=-45,
        title_x=0.5
    )

    return fig
if __name__ == '__main__':
    app.run_server(debug=True)
