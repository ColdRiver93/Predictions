#https://www.youtube.com/watch?v=n9tCB90GtbQ
#https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-prophet-in-python-3
#https://facebook.github.io/prophet/docs/quick_start.html

import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
import itertools
import warnings

import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Funciones
def norm(x): #Normalizar
  return (x - train_stats['mean']) / train_stats['std']
  
def build_model(): #Crear modelo(red neuronal)
  model = keras.Sequential([
    layers.Dense(10, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mae', 'mse', 'mape'])
  return model

#Mostrar avance del entrenamiento
class av_puntos(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# Import
print('Ejecutando...')
data = pd.read_csv(r'C:\Users\Juan Riofrio\Desktop\Clean Data\IPC-General.csv',delimiter = ';', header = 0, engine = 'python')
data = data.transpose()


#Dividing data
period = data.copy()
period = period.reset_index(drop = False)
len(period)
period['index'] = pd.DatetimeIndex(start= '2005-01-01', periods =len(period), freq = 'MS' )
period.columns = ['ds', 'y']
print(period)

#Plot
ndata = period.set_index('ds')
plt.figure(figsize=(12,5), dpi=100)
plt.plot(ndata, label = 'IPC')
plt.title('IPC Ecuador')
plt.xlabel('Fecha')
plt.ylabel('IPC')
plt.legend()
plt.show()

#Separar en Train y Test
split = int(len(period)*0.8)
train_dataset = period[:split]
test_dataset = period[split:]

#Descripcion de Train dataset
train_stats = train_dataset.describe()
train_stats.pop('y')
train_stats = train_stats.transpose()

#Elegir el target(Prediccion)
train_labels = train_dataset.pop('y')
test_labels = test_dataset.pop('y')

#Normalizar los datos  
normed_train_data = train_dataset
normed_test_data = test_dataset


#Constuir el modelo y descripcion
model = build_model()
model.summary()
EPOCHS = 1000

#Parar si ya no se mejora, patience es la cant de epocas a chequear por mejora  
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


#Entrenar el modelo
print('Entrenando el modelo...')
print('')
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, av_puntos()])


#Evaluar el modelo e imprimir errores                
# loss, mae, mse, mape = model.evaluate(normed_test_data, test_labels, verbose=0)
# print('')
# print('Errores: ')
# print("Testing set Mean Square Error: {:5.2f}".format(mse))
# print("Testing set Mean Abs Error: {:5.2f} peso".format(mae))
# print("Testing set Mean Abs Perc Error: {:5.2f} %".format(mape))


#Prediccion de los datos
test_predictions = model.predict(normed_test_data).flatten()
prediction_series = pd.Series(test_predictions, index=test_dataset.index)

# Plot Actual - Prediccion
plt.figure(figsize=(12,5), dpi=100)
plt.plot(test_labels, label = 'Actual Values', color = 'blue')
plt.plot(prediction_series, color='darkgreen', label = 'Predicted Values')
plt.legend()
plt.title("Predicción peso bebe")
plt.show()

# Plot Actual - Prediccion 2
plt.scatter(test_labels, test_predictions)
plt.xlabel('Valores [peso]')
plt.ylabel('Predicciones [peso]')
plt.axis('equal')
plt.axis('square')
plt.xlim([test_labels.min()-100,plt.xlim()[1]])
plt.ylim([test_predictions.min()-100,plt.ylim()[1]])
_ = plt.plot([0, test_labels.max()+test_predictions.max()], [0, test_labels.max()+test_predictions.max()])
plt.show()