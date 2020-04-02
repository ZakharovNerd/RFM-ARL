import pandas as pd
online = pd.read_csv('C:/Users/nikit/Downloads/datar.csv')

import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Pre-processing data

online = online.rename(columns={"Дата и время": "InvoiceDay"})
online['InvoiceDay'] = pd.to_datetime(online['InvoiceDay'])

# extract year, month and day
date_df = online['InvoiceDay'].to_frame()
online['InvoiceDay'] = date_df['InvoiceDay'].dt.date

# print the time period
print('Min : {}, Max : {}'.format(min(online.InvoiceDay), max(online.InvoiceDay)))

# pin the last date
pin_date = max(online.InvoiceDay) + dt.timedelta(1)

# calculate RFM values
rfm = online.groupby('IDПользователя').agg({
    'InvoiceDay' : lambda x: (pin_date - x.max()).days,
    'IDчека' : 'count', 
    'Сумма' : 'sum'})
# rename the columns
rfm.rename(columns = {'InvoiceDay' : 'Recency', 
                      'IDчека' : 'Frequency', 
                      'Сумма' : 'Monetary'}, inplace = True)

# create labels and assign them to tree percentile groups 
r_labels = range(4, 0, -1)
r_groups = pd.qcut(rfm.Recency, q = 4, labels = r_labels)
f_labels = range(1, 5)
f_groups = pd.qcut(rfm.Frequency, q = 4, labels = f_labels)
m_labels = range(1, 5)
m_groups = pd.qcut(rfm.Monetary, q = 4, labels = m_labels)

# make a new column for group labels
rfm['R'] = r_groups.values
rfm['F'] = f_groups.values
rfm['M'] = m_groups.values

# sum up the three columns
rfm['RFM_Segment'] = rfm.apply(lambda x: str(x['R']) + str(x['F']) + str(x['M']), axis = 1)
rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis = 1)

# calculate average values for each RFM_score
rfm_agg = rfm.groupby('RFM_Score').agg({
    'Recency' : 'mean',
    'Frequency' : 'mean',
    'Monetary' : ['mean', 'count']
})

# assign labels from total score
score_labels = ['Green', 'Bronze', 'Silver', 'Gold']
score_groups = pd.qcut(rfm.RFM_Score, q = 4, labels = score_labels)
rfm['RFM_Level'] = score_groups.values

# define function for the values below 0
def neg_to_zero(x):
    if x <= 0:
        return 1
    else:
        return x
# apply the function to Recency and MonetaryValue column 
rfm['Recency'] = [neg_to_zero(x) for x in rfm.Recency]
rfm['Monetary'] = [neg_to_zero(x) for x in rfm.Monetary]
# unskew the data
rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)

#Visualization data 

# plot the distribution of RFM values
plt.subplot(3, 1, 1); sns.distplot(rfm.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm.Monetary, label = 'Monetary')

plt.tight_layout()
plt.show()

# scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# transform into a dataframe
rfm_scaled = pd.DataFrame(rfm_scaled, index = rfm.index, columns = rfm_log.columns)

# plot the distribution of RFM values
plt.subplot(3, 1, 1); sns.distplot(rfm_scaled.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm_scaled.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm_scaled.Monetary, label = 'Monetary')

plt.tight_layout()
plt.show()

# initiate an empty dictionary
#how many cluster to choose?
wcss = {}

# Elbow method with for loop
for i in range(1, 11):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', max_iter= 300)
    kmeans.fit(rfm_scaled)
    wcss[i] = kmeans.inertia_
    
sns.pointplot(x = list(wcss.keys()), y = list(wcss.values()))
plt.xlabel('K Numbers')
plt.ylabel('WCSS')
plt.show()

#Modeling

# choose n_clusters = 3
clus = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 300)
clus.fit(rfm_scaled)

# Assign the clusters to datamart
rfm['K_Cluster'] = clus.labels_
rfm_p = rfm['K_Cluster']
rfm_p.to_csv('solution.csv')

# assign cluster column 
rfm_scaled['K_Cluster'] = clus.labels_
rfm_scaled['RFM_Level'] = rfm.RFM_Level
rfm_scaled.reset_index(inplace = True)

# melt the dataframe
rfm_melted = pd.melt(frame= rfm_scaled, id_vars= ['IDПользователя', 'RFM_Level', 'K_Cluster'], 
                     var_name = 'Metrics', value_name = 'Value')

# a snake plot with RFM
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'RFM_Level', data = rfm_melted)
plt.title('Snake Plot of RFM')
plt.legend(loc = 'upper right')

# the mean value for each cluster
cluster_avg = rfm.groupby('RFM_Level').mean().iloc[:, 0:3]

# the mean value in total 
total_avg = rfm.iloc[:, 0:3].mean()

# the proportional mean value
prop_rfm = cluster_avg/total_avg - 1

# heatmap
ax = sns.heatmap(prop_rfm, cmap= 'Oranges', fmt= '.2f', annot = True)
plt.title('Heatmap of RFM quantile')
plt.plot()

ax.set_ylim(len(prop_rfm)-0.1, -0.01)

cluster_avg_K = rfm.groupby('K_Cluster').mean().iloc[:, 0:3]

# the proportional mean value
prop_rfm_K = cluster_avg_K/total_avg - 1
prop_rfm_K

# heatmap
px = sns.heatmap(prop_rfm_K, cmap= 'Blues', fmt= '.2f')
plt.title('Heatmap of K-Means')
plt.plot()

px.set_ylim(len(prop_rfm)-0.1, -0.01)







