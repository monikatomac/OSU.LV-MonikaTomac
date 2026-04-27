import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("LV7/imgs/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()


# 7.5.2, 1)

unique_colors = np.unique(img_array, axis=0)
print("Broj razlicitih boja:", len(unique_colors))

# 7.5.2, 2)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# broj boja (klastera) koje želimo u slici
K = 5

# K-means na RGB vrijednostima
kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
labels = kmeans.fit_predict(img_array)

# centri (nove boje)
centers = kmeans.cluster_centers_

img_array_aprox = centers[labels]

img_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Originalna slika")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title(f"Kvantizirana slika (K={K})")
plt.imshow(img_aprox)

plt.tight_layout()
plt.show()


# 7.5.2, 4)

K_values = [2, 5, 10, 20]

plt.figure(figsize=(12, 8))

# original
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

for i, K in enumerate(K_values, 2):
    # K-means
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = kmeans.fit_predict(img_array)
    centers = kmeans.cluster_centers_

    # zamjena piksela centrom
    img_array_aprox = centers[labels]

    # reshape u sliku
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    # prikaz
    plt.subplot(2, 3, i)
    plt.imshow(img_aprox)
    plt.title(f"K = {K}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# 7.5.2, 5)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# vrijednosti K koje testiramo
K_values = [2, 5, 10, 20]

# petlja kroz slike
for img_id in range(2, 7):

    # ucitaj sliku
    path = fr"LV7/imgs/test_{img_id}.jpg"
    img = Image.imread(path)

    # normalizacija
    if img_id != 4:
        img = img.astype(np.float64) / 255

    # reshape u RGB vektore
    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))

    plt.figure(figsize=(12, 8))
    plt.suptitle(f"test_{img_id}", fontsize=14)

    # original
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    # K-means za različite K
    for i, K in enumerate(K_values, 2):

        kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
        labels = kmeans.fit_predict(img_array)
        centers = kmeans.cluster_centers_

        img_array_aprox = centers[labels]

        img_aprox = np.reshape(img_array_aprox, (w, h, d))

        plt.subplot(2, 3, i)
        plt.imshow(img_aprox)
        plt.title(f"K = {K}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# 7.5.2, 6)

K_values = range(1, 21)
J_values = []

for K in K_values:
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    kmeans.fit(img_array)
    
    J_values.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_values, J_values, marker='o')

plt.title("Elbow metoda - ovisnost J o K")
plt.xlabel("Broj klastera K")
plt.ylabel("Inertia (J)")
plt.grid(True)

plt.show()



# 7.5.2, 7)

K = 5  

kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
labels = kmeans.fit_predict(img_array)

plt.figure(figsize=(10, 6))

for k in range(K):
    mask = (labels == k).reshape(w, h)

    plt.subplot(2, 3, k + 1)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Klaster {k}")
    plt.axis("off")

plt.tight_layout()
plt.show()
