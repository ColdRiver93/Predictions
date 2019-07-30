#https://www.tensorflow.org/tutorials/keras/basic_regression
#https://keras.io/models/about-keras-models/
#https://keras.io/models/sequential/
#https://keras.io/getting-started/sequential-model-guide/
#https://www.kaggle.com/kaggleslayer/grocery-prediction-with-neural-network
#https://www.youtube.com/watch?v=Oi7qD5gAZ7w

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


data = pd.read_csv(r'C:\Users\Juan Riofrio\Desktop\Documentos Interesantes\Tesis\Pedidos11.csv', names=['value'], header=0)
sales= data.copy()



# Plot
sales.plot(title='Bottled Water Sales', figsize =(12,5))
plt.show()


#Decompose
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(sales, freq = 12)
decomp.plot()
plt.show()

#Dividing data
r_min = 0
train_r=28
test_r=8
r_max = r_min + train_r + test_r
period = sales[r_min:r_max]
values = sales['value'].values

date = pd.date_range(start='1/1/2016', periods=train_r+test_r, freq='1MS')
period = period.reindex(date)

month = range(1,37)
period.insert(loc=0, column = 'month', value = month)
period.pop('value')
period.insert(loc=1, column = 'value', value = values)
print(period)





train_dataset = period[0:train_r] # data as train data
test_dataset = period[train_r : (train_r+test_r)]   # data as test data


train_dataset_c =train_dataset.copy()
test_dataset_c =test_dataset.copy()
train_labels = train_dataset_c.pop('value')
test_labels = test_dataset_c.pop('value')

sns.pairplot(train_dataset, diag_kind='kde')
plt.show()

train_stats = train_dataset.describe().transpose()
print(train_stats)

def norm(x):
#  return (x - train_stats['mean']) / train_stats['std']
  return (x- train_stats['mean']) / train_stats['std']
  
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
normed_data = norm(period)
print('asdadsads')
print(period)
print('ppoiuio')
print(normed_data)
print(normed_train_data)


def build_model():
  model = keras.Sequential([
    layers.Dense(36, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(36, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model
  
model = build_model()
print(model.summary())

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)
print(normed_train_data[:10])

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1600

# history = model.fit(
  # normed_train_data, train_labels,
  # epochs=EPOCHS, validation_split = 0.2, verbose=0,
  # callbacks=[PrintDot()])

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  #plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  #plt.ylim([0,20])
  plt.legend()
  plt.show()


#plot_history(history)


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                   validation_split =0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
plt.show()

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

print(test_dataset.transpose())
print(test_predictions)
prediction_series = pd.Series(test_predictions, index=test_dataset.index)

# Plot TRAIN-TEST
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_dataset['value'], label = 'Train Values')
plt.plot(test_dataset['value'], label = 'Actual Values', color = 'blue')
plt.plot(prediction_series, color='darkgreen', label = 'Predicted Values')
plt.legend()
plt.title("Forecast of Bottled Water Sales")
plt.show()

plt.figure(figsize=(12,5), dpi=100)
plt.plot(test_dataset['value'], label = 'Actual Values', color = 'blue')
plt.plot(prediction_series, color='darkgreen', label = 'Predicted Values')
plt.legend()
plt.title("Forecast of Bottled Water Sales")
plt.show()



plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-2000, 2000], [-2000, 2000])

plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

plt.show()

tf_predictions = model.predict_on_batch(normed_data).flatten()



print(tf_predictions)
prediction_series = pd.Series(tf_predictions, index=normed_data.index)

# Plot TRAIN-TEST
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_dataset['value'], label = 'Train Values')
plt.plot(test_dataset['value'], label = 'Actual Values', color = 'blue')
plt.plot(prediction_series, color='darkgreen', label = 'Predicted Values')
plt.legend()
plt.title("Forecast of Bottled Water Sales")
plt.show()

plt.figure(figsize=(12,5), dpi=100)
plt.plot(test_dataset['value'], label = 'Actual Values', color = 'blue')
plt.plot(prediction_series, color='darkgreen', label = 'Predicted Values')
plt.legend()
plt.title("Forecast of Bottled Water Sales")
plt.show()