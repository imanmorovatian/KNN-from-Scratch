from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from knn import KNearestNeighbors


data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

knn = KNearestNeighbors(5, weights='distance')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f'The accuracy of model is: {accuracy_score(y_test, y_pred)}')