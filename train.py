import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decisionTree import decisionTree

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

data = datasets.load_breast_cancer()
x, y = data.data, data.target

x_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)

clf = decisionTree(max_depth=10)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)