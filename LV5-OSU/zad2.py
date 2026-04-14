import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            edgecolor='w',
            label=labels[cl]
        )

df = pd.read_csv("LV5/penguins.csv")

print(df.isnull().sum())

df = df.drop(columns=['sex'])
df.dropna(axis=0, inplace=True)

df['species'] = df['species'].map({
    'Adelie': 0,
    'Chinstrap': 1,
    'Gentoo': 2
})

print(df.info())

output_variable = ['species']
input_variables = ['bill_length_mm', 'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# a) 
classes, train_counts = np.unique(y_train, return_counts=True)
_, test_counts = np.unique(y_test, return_counts=True)

x_pos = np.arange(len(classes))
width = 0.35

plt.figure()
plt.bar(x_pos - width / 2, train_counts, width, label='Train')
plt.bar(x_pos + width / 2, test_counts, width, label='Test')
plt.xticks(x_pos, [labels[c] for c in classes])
plt.xlabel('Vrsta pingvina')
plt.ylabel('Broj primjera')
plt.title('Broj primjera po klasi (train/test)')
plt.legend()
plt.show()
plt.close()

# b) 
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# c) 
print("Koeficijenti (coef_):\n", model.coef_)
print("Intercept (intercept_):\n", model.intercept_)

# d)
plot_decision_regions(X_train, y_train, model)
plt.xlabel('bill_length_mm')
plt.ylabel('flipper_length_mm')
plt.title('Granica odluke - skup za ucenje')
plt.legend()
plt.show()
plt.close()

# e) 
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Tocnost: {acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[labels[i] for i in range(3)]
)
disp.plot()
plt.title('Matrica zabune - testni skup')
plt.show()
plt.close()

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=[labels[i] for i in range(3)]
))

# f) 
input_variables_vise = [
    'bill_length_mm',
    'flipper_length_mm',
    'bill_depth_mm',
    'body_mass_g'
]

X2 = df[input_variables_vise].to_numpy()

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=123
)

model2 = LogisticRegression(max_iter=5000)
model2.fit(X2_train, y2_train)

y2_pred = model2.predict(X2_test)

acc2 = accuracy_score(y2_test, y2_pred)
print(f"\nTocnost s vise ulaznih velicina ({input_variables_vise}): {acc2:.4f}")

print(classification_report(
    y2_test,
    y2_pred,
    target_names=[labels[i] for i in range(3)]
))
