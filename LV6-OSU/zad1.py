import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)

data = pd.read_csv("LV6/Social_Network_Ads.csv")
print(data.info())
data.hist()
plt.show()

X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)
print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)

y_train_knn = KNN_model.predict(X_train_n)
y_test_knn = KNN_model.predict(X_test_n)
print("KNN (K=5): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_knn))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_knn))))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=5) - Točnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_knn))))
plt.tight_layout()
plt.show()

print("\nUsporedba rezultata:")
print("Logisticka regresija tocnost na treningu: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p)))
print("KNN tocnost na treningu: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn)))
print("Logisticka regresija tocnost na testu: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p)))
print("KNN tocnost na testu: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn)))

KNN_model_1 = KNeighborsClassifier(n_neighbors=1)
KNN_model_1.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=KNN_model_1)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=1)")
plt.tight_layout()
plt.show()

KNN_model_100 = KNeighborsClassifier(n_neighbors=100)
KNN_model_100.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=KNN_model_100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=100)")
plt.tight_layout()
plt.show()

param_grid = {'n_neighbors': range(1, 21)} 
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_n, y_train)

optimal_k = grid_search.best_params_['n_neighbors']
print(f"Optimalan K za KNN: {optimal_k}")

KNN_model_7 = KNeighborsClassifier(n_neighbors=7)
KNN_model_7.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=KNN_model_7)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=7)")
plt.tight_layout()
plt.show()

svm_rbf = svm.SVC(kernel='rbf', C=1, gamma=0.5)
svm_rbf.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=svm_rbf)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM s RBF kernelom (C=1, γ=0.5)")
plt.tight_layout()
plt.show()

svm_rbf_C_100 = svm.SVC(kernel='rbf', C=10, gamma=0.1)
svm_rbf_C_100.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=svm_rbf_C_100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM s RBF kernelom (C=10, γ=0.1)")
plt.tight_layout()
plt.show()
svm_rbf_gamma_10 = svm.SVC(kernel='rbf', C=1, gamma=0.01)
svm_rbf_gamma_10.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=svm_rbf_gamma_10)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM s RBF kernelom (C=1, γ=0.01)")
plt.tight_layout()
plt.show()

svm_rbf_C_100 = svm.SVC(kernel='linear', C=10, gamma=0.1)
svm_rbf_C_100.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=svm_rbf_C_100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM s LINEAR kernelom (C=10, γ=0.1)")
plt.tight_layout()
plt.show()
svm_rbf_gamma_10 = svm.SVC(kernel='sigmoid', C=1, gamma=0.01)
svm_rbf_gamma_10.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=svm_rbf_gamma_10)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM s SIGMOID kernelom (C=1, γ=0.01)")
plt.tight_layout()
plt.show()

param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.5, 1, 10]}
grid_search_svm = GridSearchCV(svm.SVC(kernel='rbf'), param_grid_svm, cv=5, scoring='accuracy')

grid_search_svm.fit(X_train_n, y_train)

print("Najbolji parametri za SVM: ", grid_search_svm.best_params_)
print("Najbolja točnost:", grid_search_svm.best_score_)
