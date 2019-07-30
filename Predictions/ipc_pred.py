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
from fbprophet import Prophet

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
                    print(aux)
            except:
                continue
    print(aux) #mape less the better
    return(aux)
    
# Import
print('Ejecutando...')
data = pd.read_csv(r'C:\Users\Juan Riofrio\Desktop\Clean Data\IPC-General.csv',delimiter = ';', header = 0, engine = 'python')
data = data.transpose()
ipc= data[:]
ipc.columns = ['IPC']

# Plot
ipc.plot(title='IPC Ecuador', figsize=(12,5))
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
plt.suptitle('IPC Ecuador', fontsize=16)
plt.show()

#Decompose
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(ipc, freq = 12)
decomp.plot()
plt.show()


#Autocorrelation - Partial Autocorrelation
plot_acf(ipc)
plt.title('IPC Autocorrelation')
plt.show()

plot_pacf(ipc, title = 'IPC Partial Autocorrelation', lags=36)
plt.show()


#Integrated - Converting to Stationary - Differencing
ipc_diff = ipc.diff(periods=12) #Integrated of order 1
ipc_diff = ipc_diff[12:]
ipc_diff.plot()
plt.title('IPC(Integrated of order 12)')
plt.show()

#Autocorrelation - Partial Autocorrelation DIFF
plot_acf(ipc_diff)
plt.title('Diff_IPC Autocorrelation')
plt.show()

plot_pacf(ipc_diff, title='Diff_IPC Partial Autocorrelation', lags=36, method='ywm')
plt.show()


#Dividing data
#print(ipc.size,len(ipc)) #204
period = data[:]
split = int(len(period)*0.8)
train = period[:split] # data as train data
test = period[split:]   # data as test data
predictionSAR = []


#SARIMAX TRAIN-TEST

# parameters = sarima_mape(train, test)
# print(parameters[0])
# print(parameters[1])

model_SARIMA= SARIMAX(train,order=(1,2,1), seasonal_order=(0,1,1,12))
model_SARIMA_fit=model_SARIMA.fit(disp=0)
predictionSAR = model_SARIMA_fit.forecast(len(test))

forecast_SARIMA_conf = model_SARIMA_fit.get_forecast(len(test))

#lower_val = forecast_SARIMA_conf.conf_int()[['lower value']].values
#upper_val = forecast_SARIMA_conf.conf_int()[['upper value']].values
#lower_series = pd.Series(to_list_ac(lower_val), index=predictionSAR.index)
#upper_series = pd.Series(to_list_ac(upper_val), index=predictionSAR.index)

print(model_SARIMA_fit.summary())

# Plot TRAIN-TEST
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label = 'Train Values')
plt.plot(test, label = 'Actual Values', color = 'blue')
plt.plot(predictionSAR, color='darkgreen', label = 'Predicted Values')
# plt.fill_between(lower_series.index, 
                 # lower_series, 
                 # upper_series, 
                 # color='k', alpha=.15)
plt.legend()
plt.title("Prediccion IPC")
plt.show()

plt.figure(figsize=(12,5), dpi=100)
plt.plot(test, label = 'Actual Values', color = 'blue')
plt.plot(predictionSAR, color='darkgreen', label = 'Predicted Values')
plt.legend()
plt.title("Prdiccion IPC")
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
model_SARIMA= SARIMAX(ipc,order=(1,2,1), seasonal_order=(0,1,1,12))
model_SARIMA_fit=model_SARIMA.fit(disp=0)
prediction_SARIMA = model_SARIMA_fit.forecast(n_periods)
forecast_SARIMA_conf = model_SARIMA_fit.get_forecast(n_periods)

# lower_val = forecast_SARIMA_conf.conf_int()[['lower value']].values
# upper_val = forecast_SARIMA_conf.conf_int()[['upper value']].values

model_SARIMA_fit.plot_diagnostics(figsize=(9,9))
plt.show()

# make series for plotting purpose
print('------------------------------------------------------------------------')
print(model_SARIMA_fit.summary())
print('')
print('Predicted values')
print(prediction_SARIMA)

# lower_series = pd.Series(to_list_ac(lower_val), index=prediction_SARIMA.index)
# upper_series = pd.Series(to_list_ac(upper_val), index=prediction_SARIMA.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(ipc)
plt.plot(prediction_SARIMA, color='darkgreen')
# plt.fill_between(lower_series.index, 
                 # lower_series, 
                 # upper_series, 
                 # color='k', alpha=.15)

plt.title("Prediccion Final IPC")
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

from pmdarima.arima import auto_arima

# Seasonal - fit stepwise auto-ARIMA
smodel = auto_arima(train, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
                         
                         
                         
print(smodel.summary())



#df2.to_csv(r'C:\Users\Juan Riofrio\Desktop\Clean Data\IPC-General.csv',encoding = 'latin1')