import numpy as np
import math
import random


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def get_random_centroids(data, k):
    numx = np.zeros(k)
    numy = np.zeros(k)
    for i in range(k):
        rand = random.randrange(data.shape[0]-1)
        numx[i] = data[rand, 0]
        numy[i] = data[rand, 1]
    return numx, numy

def k_means_pp(data, k):
    centroids = []
    np.array(centroids)
    centroids.append(data[random.randrange(data.shape[0])])

    for i in range(k-1):
        min_dist = np.min(np.sum((np.array(centroids) - data[:, None, :]) ** 2, axis=2), axis=1)
        centroids.append(data[np.random.choice(range(data.shape[0]), p=min_dist/np.sum(min_dist))])

    return np.array(centroids)[:, 0], np.array(centroids)[:, 1]


def get_avg_diameter(data, k):
    x_centroids, y_centroids = k_means_pp(data, k)
    x_data = data[:,0]
    y_data = data[:,1]
    nx, ny = data.shape
    clusters_x = {}
    clusters_y = {}
    dist1 = []
    diameter = []
    for i in range(k):
        clusters_x.setdefault(i, [])
        clusters_y.setdefault(i, [])

    dist = np.zeros((k,))  # Initialization of np array to store distances

    # k-means algorithm
    for i in (range(nx)):
        px = x_data[i]  # x-coordinate of the current point
        py = y_data[i]  # y-coordinate of the current point

        for j in range(k):  # Loop over the centroid
            cx = x_centroids[j]  # x-coordinate of the current centroid
            cy = y_centroids[j]  # y-coordinate of the current centroid
            dist[j] = distance(px, py, cx, cy)

        ind = int(np.where(dist == np.amin(dist))[0])  # find the closest centroid

        clusters_x[ind].append(px)  # add the point to the closest cluster
        clusters_y[ind].append(py)

        x_centroids[ind] = np.mean(
            np.array(clusters_x[ind]))  # update the centroids by taking the centroid to be the mean of each cluster
        y_centroids[ind] = np.mean(np.array(clusters_y[ind]))
    for i in range(k):
        x_cluster = np.array(clusters_x[i])
        y_cluster = np.array(clusters_y[i])
        x_centroid = np.array(x_centroids[i])
        y_centroid = np.array(y_centroids[i])
        that = x_cluster.shape
        for j in range(x_cluster.shape[0]):
            pointx = x_cluster[j]
            pointy = y_cluster[j]
            dist1.append(distance(pointx, pointy, x_centroid, y_centroid))
        diameter.append(np.max(np.array(dist1)))
    avg_diam = np.mean(diameter)

    return avg_diam
