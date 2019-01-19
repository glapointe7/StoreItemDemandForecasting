import matplotlib.pyplot as plt
import numpy as np

from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import adfuller
from beautifultable import BeautifulTable


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def PlotAutoCorrelation(store_item, lags):
    plt.figure(figsize=(18, 10))

    acf = plt.subplot(1, 2, 1)
    tsaplots.plot_acf(store_item, lags=lags[0], ax=acf)
    plt.xlabel('Lags')
    plt.ylabel('ACF')
    plt.title('Auto-correlation factor \nin function of the lag')

    pacf = plt.subplot(1, 2, 2)
    pacf = tsaplots.plot_pacf(store_item, lags=lags[1], ax=pacf)
    plt.xlabel('Lags')
    plt.ylabel('PACF')
    plt.title('Partial auto-correlation factor \nin function of the lag')

    plt.show() 

def TestStationarity(store_item, critical_value):
    print('Results of the Augmented Dickey-Fuller Test:')
    print('-----------------------------------------------------\n')
    
    adf_stat, pvalue, critical_values, resstore = adfuller(store_item, regression='ct', regresults=True)

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