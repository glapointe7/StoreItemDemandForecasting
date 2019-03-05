import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore')
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

from random import randint
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas
import numpy as np
from Item import *


# Return the list of normals(1) or anomalies(-1) for each item of each store using the isolation forest algorithm.
def ListAllByApplyingIsolationForest(train, outliers_fraction):
    anomalies = []
    features = ['sales']
    for item_id in range(1, 51):
        for store_id in range(1, 11):
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
                anomalies.extend(-model.predict(item_by_year))
    return anomalies


# Return the list of normals(0) or anomalies(-1) for each item of each store using the normalization method.
def ListAllByApplyingNormalization(train, number_of_std, alpha):
    normality = []
    anomalies = []
    
    for item_id in range(1, 51):
        for store_id in range(1, 11):
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
                    anomalies.extend([int(score > number_of_std) for score in scores])
    return normality, anomalies


def ListAllByApplyingIQR(train, threshold):
    anomalies = []
        
    for item_id in range(1, 51):
        for store_id in range(1, 11):
            for year in range(2013, 2018):
                for month in range(1, 13):
                    monthly_sales = train.loc[(train.store == store_id) & (train.item == item_id) & (train.year == year) & (train.month_number == month), 'sales']
                    
                    Q1 = monthly_sales.quantile(0.25)
                    Q3 = monthly_sales.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = (monthly_sales < (Q1 - threshold * IQR)) | (monthly_sales > (Q3 + threshold * IQR))
                    anomalies.extend([int(x) for x in outliers])
    return anomalies
    
    
# Plot the anomalies detected in the sales of a random store item.
# anomaly_type is any of 'iforest', 'normal'.
def scatterPlot(item, anomaly_type):
    anomaly_feature = "anomaly_" + anomaly_type
    item_id = item.item.iloc[0]
    store_id = item.store.iloc[0]
    colors = item[anomaly_feature]
    item[anomaly_feature] = ['Outlier' if x == 1 else 'Normal' for x in item[anomaly_feature]]
    labels = list(set(item[anomaly_feature]))
    
    plt.figure(figsize=(12, 12))
    plt.scatter(x=item.index,
                y=item.sales,
                c=colors,
                cmap='bwr')
    
    red_patch = mpatches.Patch(color='red', label='Outlier')
    blue_patch = mpatches.Patch(color='blue', label='Normal')
    plt.legend(handles=[red_patch, blue_patch])
    
    plt.title("Outliers detected for the store " + str(store_id) + " - item " + str(item_id))
    plt.xticks(rotation=90)
    plt.show();