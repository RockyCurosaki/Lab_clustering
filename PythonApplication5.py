import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
dataset = pd.read_csv('D:\\Dataset.csv', encoding = 'ISO-8859-1')
f1 = dataset['x'].values
f2 = dataset['y'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1,f2, label='True Position')
plt.show()
kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(f1, f2, c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.show()