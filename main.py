from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import accuracy_score, r2_score

from knn import KNearestNeighbors


print()
print('*** Classification Section ***')

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

classifier = KNearestNeighbors(5, distance_metric='euclidean', weights='distance', classification=True)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f'The accuracy of the model is : {accuracy_score(y_test, y_pred)}')


print()
print('*** Regression Section ***')

data = load_diabetes()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = KNearestNeighbors(5, distance_metric='euclidean', weights='uniform', classification=False)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f'The R2 score of the model is: {r2_score(y_test, y_pred)}')
print()
