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

def to_list_f(forecasted):
    list=[]
    for val in forecasted:
        list.append(val)
    return(list)

def to_list_ac(actual):
    list=[]
    for val in actual:
        for v in val:
            list.append(v)
    return(list)
    

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(list(np.array(forecast) - np.array(actual)))/np.abs(actual))  # MAPE
    me = np.mean(list(np.array(forecast) - np.array(actual)))             # ME
    mae = np.mean(np.abs(list(np.array(forecast) - np.array(actual))))    # MAE
    mpe = np.mean((list(np.array(forecast) - np.array(actual)))/np.array(actual))   # MPE
    rmse = np.mean(list(np.array(list(np.array(forecast) - np.array(actual)))**2))**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse,'corr':corr})
            
def mape(forecast, actual):
    mape1 = np.mean(np.abs(list(np.array(forecast) - np.array(actual)))/np.abs(actual))  # MAPE
    return mape1
    
def sarima_mape(train_set,test_set):
#FINDING THE BEST PARAMETERS SARIMAX r2 mape mse  spss(ibm)

    p=d=q=range(0,5)
    sp=sd=sq=range(0,2)
    ss=[0,12]
    aux = None
    pdq = list(itertools.product(p,d,q))
    spdq = list(itertools.product(sp,sd,sq,ss))
    warnings.filterwarnings('ignore')
    test_v=test_set.values
    
    print('Finding the best parameters...')
    for param in pdq:
        for s_param in spdq:
            try:
                rest = param[1] + (s_param[1]*s_param[3]) + max(3*param[2]+ 1, 3*s_param[2]*s_param[3] + 1,param[0], s_param[0]*s_param[3])
                if rest > train_set.size:
                    continue
                model_arima = SARIMAX (train_set, order = param, seasonal_order = s_param)
                model_arima_fit = model_arima.fit(disp = 0)
                predictionSART = model_arima_fit.forecast(test_v.size)
                mape_v = mape(to_list_f(predictionSART.values), to_list_ac(test_v))
                if aux is None:
                    aux = [param,s_param, mape_v]
                if mape_v < aux[2]:
                    aux = [param,s_param, mape_v]
            except:
                continue
    print(aux) #mape less the better
    return(aux)
    
# Import
data = pd.read_csv(r'C:\Users\Juan Riofrio\Desktop\Documentos Interesantes\Tesis\Pedidos11.csv', names=['value'], header=0)
sales= data[:]

# Plot
sales.plot(title='Bottled Water Sales', figsize=(12,5))
plt.show()
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
plt.suptitle('Bottled Water Sales', fontsize=16)
plt.show()

#Decompose
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(sales, freq = 12)
decomp.plot()
plt.show()


#Autocorrelation - Partial Autocorrelation
plot_acf(sales)
plt.title('Sales Autocorrelation')
plt.show()

plot_pacf(sales, title = 'Sales Partial Autocorrelation', lags=36)
plt.show()


#Integrated - Converting to Stationary - Differencing
sales_diff = sales.diff(periods=12) #Integrated of order 1
sales_diff = sales_diff[12:]
sales_diff.plot()
plt.title('Sales(Integrated of order 12)')
plt.show()

#Autocorrelation - Partial Autocorrelation DIFF
plot_acf(sales_diff)
plt.title('Diff_Sales Autocorrelation')
plt.show()

plot_pacf(sales_diff, title='Diff_Sales Partial Autocorrelation', lags=36, method='ywm')
plt.show()


#Dividing data
#print(sales.size,len(sales)) #204
r_min = 0
train_r=28
test_r=8
r_max = r_min + train_r + test_r
period = data[r_min:r_max]
train = period[0:train_r] # data as train data
test = period[train_r : (train_r+test_r)]   # data as test data
predictionSAR = []


#SARIMAX TRAIN-TEST

# parameters = sarima_mape(train, test)
# print(parameters[0])
# print(parameters[1])

model_SARIMA= SARIMAX(train,order=(3,3,0), seasonal_order=(0,1,0,12))
model_SARIMA_fit=model_SARIMA.fit(disp=0)
predictionSAR = model_SARIMA_fit.forecast(test_r)

forecast_SARIMA_conf = model_SARIMA_fit.get_forecast(test_r)

lower_val = forecast_SARIMA_conf.conf_int()[['lower value']].values
upper_val = forecast_SARIMA_conf.conf_int()[['upper value']].values
lower_series = pd.Series(to_list_ac(lower_val), index=predictionSAR.index)
upper_series = pd.Series(to_list_ac(upper_val), index=predictionSAR.index)

print(model_SARIMA_fit.summary())

# Plot TRAIN-TEST
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label = 'Train Values')
plt.plot(test, label = 'Actual Values', color = 'blue')
plt.plot(predictionSAR, color='darkgreen', label = 'Predicted Values')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)
plt.legend()
plt.title("Forecast of Bottled Water Sales")
plt.show()

plt.figure(figsize=(12,5), dpi=100)
plt.plot(test, label = 'Actual Values', color = 'blue')
plt.plot(predictionSAR, color='darkgreen', label = 'Predicted Values')
plt.legend()
plt.title("Forecast of Bottled Water Sales")
plt.show()


#SARIMA TRAIN-TEST Errors
print('Predicted Values')
print(predictionSAR)
print('Actual Values')
print(test)
errors = forecast_accuracy(to_list_f(predictionSAR.values), to_list_ac(test.values))
print(errors)


#SARIMAX FINAL FORECASTING
n_periods =12
model_SARIMA= SARIMAX(sales,order=(3,3,0), seasonal_order=(0,1,0,12))
model_SARIMA_fit=model_SARIMA.fit(disp=0)
prediction_SARIMA = model_SARIMA_fit.forecast(n_periods)
forecast_SARIMA_conf = model_SARIMA_fit.get_forecast(n_periods)

lower_val = forecast_SARIMA_conf.conf_int()[['lower value']].values
upper_val = forecast_SARIMA_conf.conf_int()[['upper value']].values

model_SARIMA_fit.plot_diagnostics(figsize=(9,9))
plt.show()

# make series for plotting purpose
print('------------------------------------------------------------------------')
print(model_SARIMA_fit.summary())
print('')
print('Predicted values')
print(prediction_SARIMA)

lower_series = pd.Series(to_list_ac(lower_val), index=prediction_SARIMA.index)
upper_series = pd.Series(to_list_ac(upper_val), index=prediction_SARIMA.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(sales)
plt.plot(prediction_SARIMA, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of Bottled Water Sales")
plt.show()


# plot residual errors
from pandas import DataFrame
residuals = DataFrame(model_SARIMA_fit.resid)
residuals.plot()
plt.title('Residuals')
plt.show()
residuals.plot(kind='kde')
plt.title('Density Plot of Residuals')
plt.show()
print('Residuals description')
print(residuals.describe().transpose())