# External libraries
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Oen libraries
from python.metadata.responses import Responses

if __name__ == '__main__':
    # Generate example data
    data, _ = make_blobs(
        n_samples=300, centers=4, random_state=0, cluster_std=0.60
    )

    # Calculate WCSS for different values of k
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(
            n_clusters=i,
            init='k-means++',
            max_iter=300,
            n_init=10,
            random_state=0,
        )
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Plot the elbow method
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.show()

    print(Responses.elbow)
