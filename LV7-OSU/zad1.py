import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# 7.5.1
for flagc in range(1, 6):
    X = generate_data(500, flagc)
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'podatkovni primjeri (flagc = {flagc})')
    plt.show()

# 7.5.2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# vrijednosti K koje zelis testirati
K_values = [2, 3, 4, 5]

for flagc in range(1, 6):
    X = generate_data(500, flagc)
    
    plt.figure(figsize=(12, 8))
    
    for i, K in enumerate(K_values, 1):
        kmeans = KMeans(n_clusters=K, n_init=10)
        labels = kmeans.fit_predict(X)
        
        plt.subplot(2, 2, i)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:, 0],
                    kmeans.cluster_centers_[:, 1],
                    c='red', marker='x', s=100)
        
        plt.title(f'flagc={flagc}, K={K}')
    
    plt.tight_layout()
    plt.show()

# 7.5.3
optimal_K = {
    1: 3,
    2: 3,
    3: 4,
    4: 2,
    5: 2
}

for flagc in range(1, 6):
    X = generate_data(500, flagc)
    K = optimal_K[flagc]

    kmeans = KMeans(n_clusters=K, n_init=10)
    labels = kmeans.fit_predict(X)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(f'flagc={flagc}, optimalni K={K}')
    plt.show()
