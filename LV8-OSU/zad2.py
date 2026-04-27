import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# učitaj model
model = keras.models.load_model("mnist_model.keras")

# učitaj MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# skaliranje
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

# predikcije
y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

# pronađi pogrešne klasifikacije
wrong_idx = np.where(y_pred_classes != y_test)[0]

print("Broj pogrešno klasificiranih:", len(wrong_idx))

# prikaži nekoliko (npr. 10)
plt.figure(figsize=(10, 4))

for i in range(10):
    idx = wrong_idx[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
