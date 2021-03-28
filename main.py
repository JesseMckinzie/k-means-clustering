import numpy as np
import matplotlib.pyplot as plt
import kmeans as km
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

"""
Initialization of k
"""

data = np.loadtxt("ds4.txt", skiprows=2)  # Load the data file
x_data = data[:, 0]  # Split into x and y coordinates
y_data = data[:, 1]

k_power = 6
avg_diam = []
k_range = 2**np.arange(1, k_power)
num_cores = multiprocessing.cpu_count()
inputs = tqdm(k_range)
if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(delayed(km.get_avg_diameter)(data, i) for i in inputs)
    np.savetxt('processed.out', processed_list, delimiter=',')
    #for i in k_range:
        #avg_diam.append(km.get_avg_diameter(data, i))
    plt.plot(k_range, processed_list)
    plt.title('Average Diameter vs Number of Clusters')
    plt.ylabel('Average diameter')
    plt.xlabel('Number of clusters')
    plt.show()

k = 8

#x_centroids, y_centroids = km.random_centroids(data, k)  # Select k random points from the data set as the intial centroids
x_centroids, y_centroids = km.k_means_pp(data, k)  # k-means++ method for selecting initial k centroids
nx, ny = data.shape

"""
# Plot the data along with the initial centroids
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(x_centroids[:], y_centroids[:], c='0')
plt.show()
"""

# Initializes a dictionary to store each point in its respective cluster
clusters_x = {}
clusters_y = {}
for i in range(k):
    print(i)
    clusters_x.setdefault(i, [])
    clusters_y.setdefault(i, [])

dist = np.zeros((k,))  # Initialization of np array to store distances

#k-means algorithm
for i in (range(nx)):
    px = x_data[i]  # x-coordinate of the current point
    py = y_data[i]  # y-coordinate of the current point

    for j in range(k): # Loop over the centroid
        cx = x_centroids[j]  # x-coordinate of the current centroid
        cy = y_centroids[j]  # y-coordinate of the current centroid
        dist[j] = km.distance(px, py, cx, cy)

    ind = int(np.where(dist == np.amin(dist))[0])  # find the closest centroid

    clusters_x[ind].append(px)  # add the point to the closest cluster
    clusters_y[ind].append(py)

    x_centroids[ind] = np.mean(np.array(clusters_x[ind]))  # update the centroids by taking the centroid to be the mean of each cluster
    y_centroids[ind] = np.mean(np.array(clusters_y[ind]))

"""
Plot of the clusters
"""
for i in range(k):
    plt.scatter(clusters_x[i], clusters_y[i])

plt.scatter(x_centroids[:], y_centroids[:])
plt.show()


