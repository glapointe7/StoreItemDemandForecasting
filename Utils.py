import matplotlib.pyplot as plt
import numpy as np
import pandas

from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import adfuller
from beautifultable import BeautifulTable


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(actual, forecast):
    return 100 / len(actual) * np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))


def PlotAutoCorrelation(time_series, lags):
    plt.figure(figsize=(18, 10))

    acf = plt.subplot(1, 2, 1)
    tsaplots.plot_acf(time_series, lags=lags[0], ax=acf)
    plt.xlabel('Lags')
    plt.ylabel('ACF(k)')
    plt.title('Auto-correlation function \nin function of the lag k')

    pacf = plt.subplot(1, 2, 2)
    pacf = tsaplots.plot_pacf(time_series, lags=lags[1], ax=pacf)
    plt.xlabel('Lags')
    plt.ylabel('PACF(k)')
    plt.title('Partial auto-correlation function \nin function of the lag k')

    plt.show() 

    
def TestStationarity(time_series, critical_value):
    print('Results of the Augmented Dickey-Fuller Test:')
    print('-----------------------------------------------------\n')
    
    adf_stat, pvalue, critical_values, resstore = adfuller(time_series, regresults=True)

    print(resstore.resols.summary())
    
    alpha = list(critical_values.keys())
    critical_values = list(critical_values.values())
    
    table = BeautifulTable()
    table.column_headers = ["Test Statistic", "p-value", "alpha = " + alpha[0], "alpha = " + alpha[1], "alpha = " + alpha[2]]    
    table.append_row([adf_stat, pvalue, critical_values[0], critical_values[1], critical_values[2]])
    print(table)
    
    if pvalue >= critical_value:
        print("The time series is non-stationary\n")
    else:
        print("The time series is stationary\n")