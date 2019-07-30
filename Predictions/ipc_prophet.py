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
from fbprophet import Prophet

# Import
print('ejecutando')
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

my_model = Prophet(interval_width=0.95)
my_model.fit(period)

future_dates = my_model.make_future_dataframe(periods=12, freq='MS')
forecast = my_model.predict(future_dates)
predictions = forecast[['ds','yhat']]

#Plot
pd.plotting.register_matplotlib_converters() #Problemas con fbprophet y panda

npred = predictions.set_index('ds')
plt.figure(figsize=(12,5), dpi=100)
plt.plot(ndata, label = 'Actual')
plt.plot(npred, label = 'Prediccion')
plt.title('Predicciones IPC')
plt.xlabel('Fecha')
plt.ylabel('IPC')
plt.legend()
plt.show()

future_dates = my_model.make_future_dataframe(periods=12, freq='MS')
future_dates = future_dates[-12:]

forecast = my_model.predict(future_dates)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
predictions = forecast[['ds','yhat']]

#Plot
pd.plotting.register_matplotlib_converters() #Problemas con fbprophet y panda

npred = predictions.set_index('ds')
plt.figure(figsize=(12,5), dpi=100)
plt.plot(ndata, label = 'Actual')
plt.plot(npred, label = 'Prediccion')
plt.title('Predicciones IPC')
plt.xlabel('Fecha')
plt.ylabel('IPC')
plt.legend()
plt.show()

