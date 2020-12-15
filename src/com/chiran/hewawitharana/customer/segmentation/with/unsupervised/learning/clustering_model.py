from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

from data_pre_processing import creditcard_df

# The elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed to help find the appropriate number of clusters in a dataset.
# If the line chart looks like an arm, then the "elbow" on the arm is the value of k that is the best.
# Source:
# https://en.wikipedia.org/wiki/Elbow_method_(clustering)
# https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

# Let's scale the data first
scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(creditcard_df)

print(creditcard_df_scaled.shape)

creditcard_df_scaled

# decide no of clusters to be divided considering Elbow effect in WCSSS
scores_1 = []
range_values = range(1, 20)

for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(creditcard_df_scaled)
    scores_1.append(kmeans.inertia_)  # To get WCSSS

plt.plot(scores_1, 'bx-')

# From this we can observe that, 4th cluster seems to be forming the elbow of the curve.
# However, the values does not reduce linearly until 8th cluster.
# Let's choose the number of clusters to be 7 or 8.

# Apply K-Means
kmeans = KMeans(7)
kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_

print(kmeans.cluster_centers_.shape)
cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [creditcard_df.columns]) # To keep headers as it is
print(cluster_centers)

# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard_df.columns])
print(cluster_centers)

# First Customers cluster (Transactors): Those are customers who pay least amount of intrerest charges and careful with their money, Cluster with lowest balance ($104) and cash advance ($303), Percentage of full payment = 23%
# Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): highest balance ($5000) and cash advance (~$5000), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)
# Third customer cluster (VIP/Prime): high credit limit $16K and highest percentage of full payment, target for increase credit limit and increase spending habits
# Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance

# Save clusters df
cluster_centers.to_csv('src/resources/outputs/clusters.csv',index=False)

labels.shape # Labels associated to each data point

y_kmeans = kmeans.fit_predict(creditcard_df_scaled)
print(y_kmeans)

# concatenate the clusters labels to our original dataframe
creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster':labels})], axis = 1)
creditcard_df_cluster.head()

# Plot the histogram of various clusters
for i in creditcard_df.columns:
    plt.figure(figsize=(35, 5))
    for j in range(7):
        plt.subplot(1, 7, j + 1)
        cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
        cluster[i].hist(bins=20)
        plt.title('{}    \nCluster {} '.format(i, j))

    plt.show()