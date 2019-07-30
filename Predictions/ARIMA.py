#Time Series Predictions
#https://www.youtube.com/watch?v=D9y6dcy0xK8
#https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
#https://www.machinelearningplus.com/time-series/time-series-analysis-python/
#https://www.datacamp.com/community/tutorials/time-series-analysis-tutorial
#https://otexts.com/fpp2/stochastic-and-deterministic-trends.html


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
from statsmodels.tsa.arima_model import ARIMA
import itertools
import warnings



def parser(x): return datetime.strptime (x, '%Y-%m-%d')


def turn_list():
    for pred in fc:
        forecasted.append(pred)
    test_n = test.values
    for val in test_n:
        for idk in val:
            actual.append(idk)
    print('')
    print('Forecasted Values')
    print(forecasted)
    print('Actual values')
    print(actual)

#Errors
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(list(np.array(forecast) - np.array(actual)))/np.abs(actual))  # MAPE
    me = np.mean(list(np.array(forecast) - np.array(actual)))             # ME
    mae = np.mean(np.abs(list(np.array(forecast) - np.array(actual))))    # MAE
    mpe = np.mean((list(np.array(forecast) - np.array(actual)))/np.array(actual))   # MPE
    rmse = np.mean(list(np.array(list(np.array(forecast) - np.array(actual)))**2))**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
#    mins = np.amin(np.hstack([forecast[:,None], 
#                              actual[:,None]]), axis=1)
#    maxs = np.amax(np.hstack([forecast[:,None], 
#                              actual[:,None]]), axis=1)
#    minmax = 1 - np.mean(mins/maxs)             # minmax
#    acf1 = acf(pd.Series(list(np.array(forecast)-np.array(test)), index=test.index))[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse,'corr':corr})



#('https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv', names=['value'], header=0)
#('/users/Juan Riofrio/Downloads/daily-min-temperatures.csv', index_col = 0,parse_dates = [0], date_parser = parser,converters={'Daily minimum temperatures':float})
#Plot sales
t_file = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv', names=['value'], header=0)

r_min =0
train_r=85
test_r=15
size_d=train_r+test_r+r_min

#Dividing data
sales = t_file[r_min:size_d]
train = sales[0:train_r] #28 data as train data
test = sales[train_r:(train_r+test_r)]   #8 data as test data
predictionsAR = []
predictionsARIMA = []
forecasted= []
actual = []

#print(sales.head())
#print(sales.tail())
#print(sales.values)
#print(sales.size)
print(sales.describe().transpose())

plt.figure(figsize=(12,5), dpi=100)
plt.plot(sales)
plt.title('Sales per Date(month-year)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()


decomp = seasonal_decompose(sales, freq = 25)
decomp.plot()
plt.show()


#Autocorrelation - Partial Autocorrelation
plot_acf(sales)
plt.title('Sales Autocorrelation')
plt.show()

plot_pacf(sales, title = 'Sales Partial Autocorrelation', lags=50)
plt.show()


#Integrated - Converting to Stationary - Differencing
sales_diff = sales.diff(periods=1) #Integrated of order 1
sales_diff = sales_diff[1:]

sales_diff.plot()
plt.title('Sales(Integrated of order 1)')
plt.xlabel('Dates')
plt.show()

#Autocorrelation - Partial Autocorrelation DIFF
plot_acf(sales_diff)
plt.title('Diff_Sales Autocorrelation')
plt.show()

plot_pacf(sales_diff, title='Diff_Sales Partial Autocorrelation', lags=50, method='ywm')
plt.show()


#AutoRegressive Model
model_ar = AR(train)    #freq = 'D'
model_ar_fit = model_ar.fit()
predictionsAR = model_ar_fit.predict(start=train_r, end = size_d)

plt.plot(test, label = 'Real data')
plt.plot(predictionsAR, color = 'red', label ='AR prediction')
plt.title('Predictions AR')
plt.ylabel('Sales')
plt.legend()
plt.show()


#ARIMA model(AutoRegresive Integrated Moving Average)
#Parameters for ARIMA (p,d,q)
#p = periods taken for autoregressive model
#d = Integrated order, difference
#q = periods in moving average model
#AIC = Akaike Information Criteria (the less the better)

model_arimaX = ARIMA (sales, order = (3,2,1)) #freq = 'D'
model_arimaX_fit = model_arimaX.fit(disp = 0)
model_arimaX_fit.plot_predict(dynamic=False)
plt.show()


#ARIMA parameters (found the correct one)
#p=d=q=range(0,5)
#aux = None
#pdq = list(itertools.product(p,d,q))
#warnings.filterwarnings('ignore')

#for param in pdq:
#   try:
#      model_arima = ARIMA (train, order = param)
#      model_arima_fit = model_arima.fit(disp = 0)
#      if aux is None:
#         aux = [param, model_arima_fit.aic]
#      if model_arima_fit.aic < aux[1]:
#         aux = [param, model_arima_fit.aic]
#      print(aux)
#   except:
#      continue
#print(aux) #aic less the better

#model_arima = ARIMA(train, order = aux[0])    #Train with the best p,d,q
model_arima = ARIMA(train, order = (3,2,1))    #freq = 'D'
model_arima_fit = model_arima.fit(disp = 0)
fc, se , conf = model_arima_fit.forecast(test_r, alpha=0.05) #95%conf
print(fc)
print(test)
print(test)
turn_list()
x=forecast_accuracy(forecasted, actual)
print('')
print(x)
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0])
upper_series = pd.Series(conf[:, 1])


# Plot
print(model_arima_fit.summary())
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
print(fc_series.subtract(test))
plt.fill_between(test.index, lower_series, upper_series, color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

predictionsARIMA = model_arima_fit.forecast(steps=test_r)[0]
prediction_serie = pd.Series(predictionsARIMA, index=test.index)
plt.plot(test, label = 'Real data')
plt.plot(prediction_serie, color = 'red', label = 'ARIMA prediction')
plt.title('Predictions ARIMA')
plt.ylabel('Sales')
plt.legend()
plt.show()


# plot residual errors
residuals = DataFrame(model_arima_fit.resid)
residuals.plot()
plt.title('Residual Errors')
plt.show()
residuals.plot(kind='kde')
plt.title('Density Plot of Residual Errors')
plt.show()
print('Residual Errors description')
print(residuals.describe().transpose())

#import statsmodels.tsa.statespace.mlemodel.MLEResults.plot_diagnostics
#model_arima_fit.plot_diagnostics(figsize=(15,12))
#plt.show()


# Forecast
n_periods = 24
n_start=train_r+test_r
prediction_arima = ARIMA(sales, order = (3,2,1))    #freq = 'D'
prediction_arima_fit = prediction_arima.fit(disp = 0)
fc, se , confint = prediction_arima_fit.forecast(n_periods, alpha=0.05) #95%conf
index_of_fc = np.arange(n_start, n_start+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
print('------------------------------------------------------------------------')
print('')
print('Forecasted values')
print(fc_series)
print(prediction_arima_fit.summary())
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(sales)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage")
plt.show()