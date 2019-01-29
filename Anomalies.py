from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from heapq import nlargest
from random import randint

import matplotlib.pyplot as plt
import pandas

# Return the list of normals(1) or anomalies(-1) for each item of each store using the SVM algorithm 
# where the data points are (x, y) = (date, sales).
def ListAllByApplyingSVM(train, outliers_fraction):
    anomalies = []
    for store in range(1, 11):
        for item in range(1, 51):
            store_item = train.loc[(train.store == store) & (train.item == item), ['date', 'sales']]
            store_item.date = store_item.date.str.replace('\D', '').astype(int)
            
            scaler = StandardScaler()
            np_scaled = scaler.fit_transform(store_item[['date', 'sales']])
            scaled_store_item = pandas.DataFrame(np_scaled)
            
            model = OneClassSVM(nu=outliers_fraction, 
                                kernel="rbf", 
                                gamma=0.3)
            model.fit(scaled_store_item)
            
            anomalies.extend(model.predict(scaled_store_item))
    return anomalies

# Return the list of normals(1) or anomalies(-1) for each item of each store using the isolation forest algorithm.
def ListAllByApplyingIsolationForest(train, outliers_fraction):
    anomalies = []
    for store in range(1, 11):
        for item in range(1, 51):
            store_item = train.loc[(train.store == store) & (train.item == item), ['date', 'sales']]
            store_item.date = store_item.date.str.replace('\D', '').astype(int)
            
            scaler = StandardScaler()
            np_scaled = scaler.fit_transform(store_item[['date', 'sales']])
            scaled_store_item = pandas.DataFrame(np_scaled)
            
            model =  IsolationForest(contamination=outliers_fraction)
            model.fit(scaled_store_item) 
            anomalies.extend(model.predict(scaled_store_item))
    return anomalies

# Since the elbow curve is similar between all store items, we chose the first store and item to get the number of clusters.
def GetKMeanNumberOfClusters(train, nb_clusters_to_test):
    store_item = train.loc[(train.store == 1) & (train.item == 1), ['date', 'sales']]
    store_item.date = store_item.date.str.replace('\D', '').astype(int)
    
    kmeans = [KMeans(n_clusters=k).fit(store_item[['date', 'sales']]) for k in nb_clusters_to_test]
    sum_squared_errors = [kmeans[k].inertia_ for k in range(len(kmeans))]
    
    return kmeans, sum_squared_errors

# Return the list of normals(0) or anomalies(-1) for each item of each store using the K-Mean algorithm.
def ListAllByApplyingKMean(train, outliers_fraction, cluster):
    anomalies = []
    for store in range(1, 11):
        for item in range(1, 51):
            store_item = train.loc[(train.store == store) & (train.item == item), ['date', 'sales']]
            store_item.date = store_item.date.str.replace('\D', '').astype(int)
            
            distances = cdist(store_item[['date', 'sales']], cluster.cluster_centers_)
            number_of_outliers = int(outliers_fraction * len(distances))

            distances = list(map(lambda x: max(x), distances))
            max_distances = nlargest(number_of_outliers, distances)

            threshold = max_distances[-1]
            anomalies.extend(-((distances >= threshold).astype(int)))
    return anomalies
            

# Plot the anomalies detected in the sales of a random store item.
# anomaly_type is any of 'svm', 'iforest', 'kmean'.
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