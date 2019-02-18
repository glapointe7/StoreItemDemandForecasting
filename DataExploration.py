import matplotlib.pyplot as plt
import statsmodels.api as sm
from Item import *


# Determine the rolling means and the standard deviation per month of the item object.
def PlotRollingMeanAndStd(item):
    rollmean = item.dataset.sales.rolling(30).mean()
    rollstd = item.dataset.sales.rolling(30).std()

    # Plot the item sales, the rolling mean and the rolling standard deviation.
    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 20})
    plt.plot(item.dataset['sales'], color='blue',label='Original')
    plt.plot(rollmean, color='red', label='Rolling Mean')
    plt.plot(rollstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title("Sales in function of the date for the store " + str(item.store_id) + " - item " + str(item.item_id))
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()
    
    
def PlotSeasonalDecomposition(dataset, frequency):
    diagnostics = sm.tsa.seasonal_decompose(dataset, model='additive', freq=frequency)
    diagnostics.plot()
    plt.subplots_adjust(left=None, bottom=None, right=2, top=1.5, wspace=None, hspace=0.1)
    plt.show()