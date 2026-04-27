import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image
import os

# 1. učitaj model
model = keras.models.load_model("mnist_model.keras")

# 2. folder sa slikama
folder_path = r"D:\p\osu_lv\LV8"

# 3. iteracija kroz sve test slike
for i in range(10):
    img_path = os.path.join(folder_path, f"test{i}.png")
    
    # učitaj sliku
    img = Image.open(img_path).convert("L")
    
    # resize na 28x28
    img = img.resize((28, 28))
    
    # u numpy
    img_array = np.array(img)
    
    # normalizacija
    img_array = img_array.astype("float32") / 255.0

    img_array = 1 - img_array
    
    # dodaj dimenzije
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    # predikcija
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction)
    
    # očekivana klasa iz imena fajla
    true_class = i
    
    print(f"Slika: test{i}.png | True: {true_class} | Pred: {predicted_class}")
    
    # prikaz
    plt.figure()
    plt.imshow(np.squeeze(img_array), cmap='gray')
    plt.title(f"True: {true_class}, Pred: {predicted_class}")
    plt.axis('off')
    plt.show()
