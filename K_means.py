#1
import numpy as np
import matplotlib.pyplot as plt
import random

# Load dataset into 2D list "Data"
data_path = 'dataset2.txt'
my_data = np.genfromtxt(data_path, delimiter=' ')




# K-Means clustering algorithm
def kmeans(data, K, max_iterations=100, tolerance=50):
    # Randomly select K different data points from “Data” and store them into 2D list "Centers"
    centers_idx = np.random.choice(data.shape[0], K, replace=False)
    centers = data[centers_idx]

    # Initialize clusters
    clusters = [[] for _ in range(K)]

    # Initialize "Shift" for convergence check
    shift = np.inf

    # Iteration counter
    itr = 0

    while itr < max_iterations and shift >= tolerance:
        # Clear clusters
        clusters = [[] for _ in range(K)]

        # Assign each sample/data point to the closest center
        for i, sample in enumerate(data):
            # Calculate distances to all centers
            distances = np.linalg.norm(centers - sample, axis=1)
            # Find the index of the closest center
            closest_center_index = np.argmin(distances)
            # Append the sample index to the corresponding cluster
            clusters[closest_center_index].append(i)

        # Calculate new centers by taking the mean of the points in each cluster
        new_centers = np.array([np.mean(data[cluster], axis=0) for cluster in clusters])

        # Calculate the shift from previous centers
        shift = np.sum(np.linalg.norm(new_centers - centers, axis=1))

        # Update centers
        centers = new_centers

        # Increment iteration counter
        itr += 1

    return clusters, centers


# Function to plot clusters
def plot_clusters(data, clusters, centers, K):
    colors = ['m', 'r', 'g', 'b', 'c', 'k', 'y']
    for k in range(K):
        x = data[clusters[k]][:, 0]
        y = data[clusters[k]][:, 1]
        plt.scatter(x, y, c=colors[k], label="Cluster " + str(k))

    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, c='black', label='Centers')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='lower left')
    plt.title(f'K-Means Clustering with K={K}')
    plt.show()


# Function to calculate inertia
def calculate_inertia(data, clusters, centers):
#Inertia is the sum of squared distances of samples to their closest cluster center.
    inertia = 0
    for k in range(len(clusters)):
        cluster_center = centers[k]
        cluster_points = data[clusters[k]]
        inertia += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
    return inertia


# Perform K-Means clustering and generate plots for different values of K
for K in [2, 4, 6, 7]:
    clusters, centers = kmeans(my_data, K)
    plot_clusters(my_data, clusters, centers, K)
    inertia = calculate_inertia(my_data, clusters, centers)
    print(f'Inertia for K={K}: {inertia}')











"""
#2
import numpy as np
import matplotlib.pyplot as plt

# Load dataset into 2D list "Data"
data_path = 'dataset2.txt'
my_data = np.loadtxt(data_path, delimiter=' ')


# K-Means clustering algorithm
def kmeans(data, K, max_iterations=100, tolerance=0.001):
    # Randomly select K different data points from “Data” and store them into 2D list "Centers"
    centers_idx = np.random.choice(data.shape[0], K, replace=False)
    centers = data[centers_idx]

    # Initialize clusters
    clusters = [[] for _ in range(K)]

    # Initialize "Shift" for convergence check
    shift = np.inf

    # Iteration counter
    itr = 0

    while itr < max_iterations and shift >= tolerance:
        # Clear clusters
        clusters = [[] for _ in range(K)]

        # Assign each sample/data point to the closest center
        for i, sample in enumerate(data):
            # Calculate distances to all centers
            distances = np.linalg.norm(centers - sample.reshape(1, -1), axis=1)  # Reshape to ensure proper broadcasting
            # Find the index of the closest center
            closest_center_index = np.argmin(distances)
            # Append the sample index to the corresponding cluster
            clusters[closest_center_index].append(i)

        # Calculate new centers by taking the mean of the points in each cluster
        new_centers = []
        for k in range(K):
            if len(clusters[k]) == 0:
                # If a cluster is empty, assign the center to one of the data points
                new_center = data[np.random.randint(0, len(data))]
            else:
                new_center = np.mean(data[clusters[k]], axis=0)
            new_centers.append(new_center)

        # Calculate the shift from previous centers
        shift = np.sum(np.linalg.norm(np.array(new_centers) - centers.reshape(K, -1), axis=1))  # Reshape centers

        # Update centers
        centers = np.array(new_centers)

        # Increment iteration counter
        itr += 1

    return clusters, centers



# Function to plot clusters
def plot_clusters(data, clusters, centers, K):
    colors = ['m', 'r', 'g', 'b', 'c', 'k', 'y']
    for k in range(K):
        x = data[clusters[k]][:, 0]
        y = data[clusters[k]][:, 1]
        plt.scatter(x, y, c=colors[k], label="Cluster " + str(k))

    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, c='black', label='Centers')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='lower left')
    plt.title(f'K-Means Clustering with K={K}')
    plt.show()


# Function to calculate inertia
def calculate_inertia(data, clusters, centers):
    inertia = 0
    for k in range(len(clusters)):
        cluster_center = centers[k]
        cluster_points = data[clusters[k]]
        inertia += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
    return inertia


# Perform K-Means clustering and generate plots for different values of K
for K in [2, 4, 6, 7]:
    clusters, centers = kmeans(my_data, K)
    plot_clusters(my_data, clusters, centers, K)
    inertia = calculate_inertia(my_data, clusters, centers)
    print(f'Inertia for K={K}: {inertia}')
"""
