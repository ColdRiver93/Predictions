# ----> Creado por Juan Riofrio
# ----> Entrenamiento de una red neuronal de tensorflow
# ----> Datos de nacimientos en Ecuador 2017
# ----> Prediccion de pesos de los nacidos de acuerdo a (sexo, talla, sem_gest tipo_part, edad_mad)


#Librerias
import matplotlib.pyplot as plt
import pandas as pd
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


#Importar data
##          ---MODIFICAR----
data = pd.read_csv(r'C:\Users\Juan Riofrio\Desktop\ABD\Datos_abiertos_ENV_2017\clean_data.csv', header=0)
naci_data1= data.copy()

#Descripcion dataset completo
print('Data exportada exitosamente')
ds = naci_data1.describe().transpose()
print('Descripcion de los datos: ')
print(ds)
print('')

#Reducir cantidad de datos para TRAIN-TEST
tnds = naci_data1.sample(frac=0.01,random_state=0)
ttds = naci_data1.sample(frac=0.0002,random_state=0)
size = len(tnds)


#Crear dataset TRAIN-TEST
dataset = tnds.append(ttds)
dataset = dataset.reset_index(drop=True)


#Descripcion y graficos de dataset
data_stats = dataset.describe().transpose()
print('Descripcion de los datos: ')
print(data_stats)
print('')

#Boxplots de Peso, Talla
plt.boxplot(dataset['peso'], labels = ['Peso'], meanline = True)
plt.show()

plt.boxplot(dataset['talla'], labels = ['Talla'], meanline = True)
plt.show()

#Histogramas Peso, Talla, Sexo, Edad madre.
plt.hist(dataset['peso'], bins = 25)
plt.xlabel("Pesos recien nacidos")
_ = plt.ylabel("Count")
plt.show()

plt.hist(dataset['talla'], bins = 25)
plt.xlabel("Talla recien nacidos")
_ = plt.ylabel("Count")
plt.show()

plt.hist(dataset['sexo'], bins = 25)
plt.xlabel("Sexo recien nacidos(H=0, M=1)")
_ = plt.ylabel("Count")
plt.show()

plt.hist(dataset['edad_mad'], bins = 25)
plt.xlabel("Edad madre")
_ = plt.ylabel("Count")
plt.show()

sns.pairplot(dataset, diag_kind='kde',kind='reg', height = 3)
plt.show()


#Separar en Train y Test
train_dataset = dataset[:size]
test_dataset = dataset[size:]

#Descripcion de Train dataset
train_stats = train_dataset.describe()
train_stats.pop('peso')
train_stats = train_stats.transpose()


#Elegir el target(Prediccion)
train_labels = train_dataset.pop('peso')
test_labels = test_dataset.pop('peso')


#Normalizar los datos  
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

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
loss, mae, mse, mape = model.evaluate(normed_test_data, test_labels, verbose=0)
print('')
print('Errores: ')
print("Testing set Mean Square Error: {:5.2f}".format(mse))
print("Testing set Mean Abs Error: {:5.2f} peso".format(mae))
print("Testing set Mean Abs Perc Error: {:5.2f} %".format(mape))


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


#Distibucion del error
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Error Prediccion [Peso]")
_ = plt.ylabel("Count")
plt.show()