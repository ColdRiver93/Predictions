#https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
#https://github.com/yinniyu/timeseries_ARIMA
#https://www.kaggle.com/poiupoiu/how-to-use-sarimax

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
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import warnings

def to_list(values):
    list=[]
    for val in values:
        for v in val:
            list.append(v)
    return(list)
    
# Import
data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')
sales= data[:]

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(data[:], label='Original Series')
axes[0].plot(data[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)


# Seasonal Differencing
axes[1].plot(data[:], label='Original Series')
axes[1].plot(data[:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Drug Sales', fontsize=16)
plt.show()


#Decompose
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(sales, freq =12)
decomp.plot()
plt.show()

#Autocorrelation - Partial Autocorrelation
plot_acf(sales)
plt.title('Sales Autocorrelation')
plt.show()

plot_pacf(sales, title = 'Sales Partial Autocorrelation', lags=50)
plt.show()


#Integrated - Converting to Stationary - Differencing
sales_diff = sales.diff(periods=12) #Integrated of order 1
sales_diff = sales_diff[12:]

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

#Dividing data
#print(sales.size,len(sales)) #204
r_min = 0
train_r=76
test_r=24
r_max = r_min + train_r + test_r
period = data[r_min:r_max]
train = period[0:train_r] # data as train data
test = period[train_r : (train_r+test_r)]   # data as test data
predictionSAR = []

#VARIABLE EXOGENA
# multiplicative seasonal component
result_mul = seasonal_decompose(data['value'][-36:],   # 3 years
                                model='multiplicative', 
                                extrapolate_trend='freq')

seasonal_index = result_mul.seasonal[-12:].to_frame()
seasonal_index['month'] = pd.to_datetime(seasonal_index.index).month

# merge with the base data
data['month'] = data.index.month
df = pd.merge(data, seasonal_index, how='left', on='month')
df.columns = ['value', 'month', 'seasonal_index']
df.index = data.index  # reassign the index.


#SARIMAX TRAIN-TEST
print('  -- SARIMAX --   ')
model_SARIMA= SARIMAX(train,order=(3,0,0), seasonal_order=(0,1,1,12), exog= df[:76][['seasonal_index']])
model_SARIMA_fit=model_SARIMA.fit(disp=0)
predictionSAR = model_SARIMA_fit.forecast(test_r, exog =df[76:100][['seasonal_index']] )
forecast_SARIMA_conf = model_SARIMA_fit.get_forecast(test_r,exog =df[76:100][['seasonal_index']])

lower_val = forecast_SARIMA_conf.conf_int()[['lower value']].values
upper_val = forecast_SARIMA_conf.conf_int()[['upper value']].values
lower_series = pd.Series(to_list(lower_val), index=predictionSAR.index)
upper_series = pd.Series(to_list(upper_val), index=predictionSAR.index)


# Plot TRAIN-TEST
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label = 'Train Values')
plt.plot(test, label = 'Actual Values')
plt.plot(predictionSAR, color='darkgreen', label = 'Forecast Values')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)
plt.legend()
plt.title("Forecast of Drug Sales")
plt.show()

plt.figure(figsize=(12,5), dpi=100)
plt.plot(test, label = 'Actual Values')
plt.plot(predictionSAR, color='darkgreen', label = 'Forecast Values')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)
plt.legend()
plt.title("Forecast of Drug Sales")
plt.show()


#SARIMAX TRAIN-TEST Errors
print('Predicted Values')
print(predictionSAR)
print('Actual Values')
print(test)


#SARIMAX FINAL FORECASTING
n_periods =24
model_SARIMA= SARIMAX(sales,order=(3,0,0), seasonal_order=(0,1,1,12),exog= df[['seasonal_index']])
model_SARIMA_fit=model_SARIMA.fit(disp=0)
prediction_SARIMA = model_SARIMA_fit.forecast(n_periods, exog  = np.tile(seasonal_index.value, 2).reshape(-1,1))
forecast_SARIMA_conf = model_SARIMA_fit.get_forecast(n_periods, exog=np.tile(seasonal_index.value, 2).reshape(-1,1))

lower_val = forecast_SARIMA_conf.conf_int()[['lower value']].values
upper_val = forecast_SARIMA_conf.conf_int()[['upper value']].values

model_SARIMA_fit.plot_diagnostics(figsize=(15,12))
plt.show()

# make series for plotting purpose
print('------------------------------------------------------------------------')
print('')
print('Forecasted values')
print(prediction_SARIMA)
print(model_SARIMA_fit.summary())
lower_series = pd.Series(to_list(lower_val), index=prediction_SARIMA.index)
upper_series = pd.Series(to_list(upper_val), index=prediction_SARIMA.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(sales)
plt.plot(prediction_SARIMA, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of Drug Sales")
plt.show()


# plot residual errors
from pandas import DataFrame
residuals = DataFrame(model_SARIMA_fit.resid)
residuals.plot()
plt.title('Residual Errors')
plt.show()
residuals.plot(kind='kde')
plt.title('Density Plot of Residual Errors')
plt.show()
print(residuals.describe().transpose())


#FINDING THE BEST PARAMETERS SARIMAX

# p=q=range(0,5)
# d=[0,1]
# sp=sq=range(0,5)
# sd=[0,1]
# ss=[12]
# aux = None
# pdq = list(itertools.product(p,d,q))
# s_pdq = list(itertools.product(sp,sd,sq,ss))
# warnings.filterwarnings('ignore')

# print('Init')
# for param in pdq:
    # for s_param in s_pdq:
        # try:
            # try_SARIMA= SARIMAX(sales,order=param, seasonal_order=s_param)
            # try_SARIMA_fit=try_SARIMA.fit(disp=0)
            # if aux is None:
                # aux = [param, s_param, try_arima_fit.aic]
            # if try_arima_fit.aic < aux[2]:
                # aux = [param, s_param, try_arima_fit.aic]
            # print(aux)
        # except:
            # print('Error',param, s_param )
            # continue
# print(aux) #aic less the better