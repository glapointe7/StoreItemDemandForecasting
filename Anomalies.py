from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from random import randint
from scipy import stats

import matplotlib.pyplot as plt
import pandas
import statistics
import numpy as np


# Return the list of normals(1) or anomalies(-1) for each item of each store using the isolation forest algorithm.
def ListAllByApplyingIsolationForest(train, outliers_fraction):
    anomalies = []
    for store_id in range(1, 11):
        for item_id in range(1, 51):
            store_item = train.loc[(train.store == store_id) & (train.item == item_id), ['date', 'sales', 'month_number']]
            store_item.date = store_item.date.str.replace('\D', '').astype(int)
            
            scaler = StandardScaler()
            np_scaled = scaler.fit_transform(store_item[['date', 'sales', 'month_number']])
            scaled_store_item = pandas.DataFrame(np_scaled)
            
            model = IsolationForest(contamination=outliers_fraction)
            model.fit(scaled_store_item) 
            anomalies.extend(model.predict(scaled_store_item))
    return anomalies


# Return the list of normals(0) or anomalies(-1) for each item of each store using the normalization method.
def ListAllByApplyingNormalization(train, number_of_std, alpha):
    normality = []
    anomalies = []
    for s in range(1, 11):
        for i in range(1, 51):
            item_by_year = train.loc[(train.store == s) & (train.item == i), ['date', 'sales', 'month_number']]
            item_by_year.date = pandas.DatetimeIndex(item_by_year.date).year

            for y in range(2013, 2018):
                for m in range(1, 13):
                    item_month = np.array(item_by_year.sales[(item_by_year.date == y) & (item_by_year.month_number == m)])

                    w, p_value = stats.shapiro(item_month)
                    normality.append(int(p_value > alpha))

                    sales_by_month_mean = statistics.mean(item_month)
                    sales_by_month_std = statistics.stdev(item_month)
                    scores = np.absolute((item_month - sales_by_month_mean) / sales_by_month_std)
                    anomalies.extend([-int(score > number_of_std) for score in scores])
    return normality, anomalies
    
# Plot the anomalies detected in the sales of a random store item.
# anomaly_type is any of 'iforest', 'normal'.
def PlotRandomStoreItem(train, anomaly_type):
    store_id = randint(1, 10)
    item_id = randint(1, 50)
    anomaly_feature = "anomaly_" + anomaly_type
    
    store_item = train.loc[(train.store == store_id) & (train.item == item_id), ['date', 'sales', anomaly_feature]]
    store_item.date = pandas.DatetimeIndex(store_item.date)
    anomalies = store_item.loc[store_item[anomaly_feature] == -1, ['date', 'sales']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(store_item.date.values, store_item.sales, color='blue', label='Normal')
    ax.scatter(anomalies.date.values, anomalies.sales, color='red', label='Anomaly')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title("Anomalies detected in the sales of the store " + str(store_id) + " - item " + str(item_id))
    plt.legend()
    plt.show();

# 
def PlotElbowCurve(nb_clusters_to_test, sum_squared_errors):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(nb_clusters_to_test, sum_squared_errors)
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Curve')
    plt.show();