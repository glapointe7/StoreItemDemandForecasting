import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore')
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

from random import randint
from scipy import stats

import matplotlib.pyplot as plt
import pandas
import numpy as np
from Item import *


# Return the list of normals(1) or anomalies(-1) for each item of each store using the isolation forest algorithm.
def ListAllByApplyingIsolationForest(train, outliers_fraction):
    anomalies = []
    features = ['sales']
    for store_id in range(1, 11):
        for item_id in range(1, 51):
            for year in range(2013, 2018):
                item_by_year = train.loc[(train.store == store_id) & (train.item == item_id) & (train.year == year), features]
                item_by_year.index = item_by_year.index.values.astype(int)

                scaler = StandardScaler()
                np_scaled = scaler.fit_transform(item_by_year[features].astype(float))
                item_by_year = pandas.DataFrame(np_scaled)
                with warnings.catch_warnings():
                    warnings.filterwarnings(action='ignore')
                    model = IsolationForest(contamination=outliers_fraction)
                    model.fit(item_by_year) 
                anomalies.extend(model.predict(item_by_year))
    return anomalies


# Return the list of normals(0) or anomalies(-1) for each item of each store using the normalization method.
def ListAllByApplyingNormalization(train, number_of_std, alpha):
    normality = []
    anomalies = []
    
    for store_id in range(1, 11):
        for item_id in range(1, 51):
            item_data = train.loc[(train.store == store_id) & (train.item == item_id), ['sales', 'year', 'month_number']]
            
            first_month_sales = item_data.iloc[:29, item_data.columns.get_loc('sales')]
            item_data['rolling_mean'] = item_data.sales.rolling(30).mean()
            item_data.loc[:29, 'rolling_mean'] = first_month_sales

            item_data['rolling_std'] = item_data.sales.rolling(30).std()
            item_data.loc[:29, 'rolling_std'] = first_month_sales
            
            for year in range(2013, 2018):
                for month in range(1, 13):
                    item_by_month = item_data.loc[(item_data['year'] == year) & (item_data['month_number'] == month), ['sales', 'rolling_mean', 'rolling_std']]
                    w, p_value = stats.shapiro(np.array(item_by_month.sales))
                    normality.append(int(p_value > alpha))

                    scores = np.absolute((item_by_month.sales - item_by_month.rolling_mean) / item_by_month.rolling_std)
                    anomalies.extend([-int(score > number_of_std) for score in scores])
    return normality, anomalies
    
# Plot the anomalies detected in the sales of a random store item.
# anomaly_type is any of 'iforest', 'normal'.
def scatterPlot(item, anomaly_type):
    anomaly_feature = "anomaly_" + anomaly_type
    item_id = item.item.iloc[0]
    store_id = item.store.iloc[0]
    
    anomalies = item.loc[item[anomaly_feature] == -1, 'sales']
    print(anomalies)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(item.index, item.sales, color='blue', label='Normal')
    ax.scatter(x=anomalies.index, y=anomalies.values, color='red', label='Anomaly')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title("Anomalies detected in the sales for the store " + str(store_id) + " - item " + str(item_id))
    plt.legend(loc='best')
    plt.show();