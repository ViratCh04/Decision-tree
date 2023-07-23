import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decisionTree import decisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def accuracy(y_test, y_pred):
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    return round(accuracy * 100, 5)

def dtree_sklearn(x_train, x_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy(y_test, y_pred)
    return acc

def cf_matrix(x_train, x_test, y_train, y_test):
    dtc = DecisionTreeClassifier()
    mdtc = decisionTree(max_depth=10)
    dtc.fit(x_train, y_train)
    mdtc.fit(x_train, y_train)
    y_head_dtc = dtc.predict(x_test)
    y_head_mdtc = mdtc.predict(x_test)

    # using confusion matrices to visualise accuracy of predictions for both algorithms
    cm_dtc = confusion_matrix(y_test, y_head_dtc)
    cm_mdtc = confusion_matrix(y_test, y_head_mdtc)

    plt.figure(figsize=(12, 6))
    plt.suptitle("Confusion Matrices")
    plt.subplots_adjust(wspace = 0.4, hspace = 0.4)

    plt.subplot(2,3,1)
    plt.title("Scikit-learn's Decision Tree Classifier Confusion Matrix")
    sns.heatmap(cm_dtc, annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 12})

    plt.subplot(2,3,3)
    plt.title("My Decision Tree Classifier's Confusion Matrix")
    sns.heatmap(cm_mdtc, annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 12})

    plt.show()

def main():
    data = datasets.load_digits()
    x, y = data.data, data.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )

    clf = decisionTree(max_depth=10)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    my_acc = accuracy(y_test, y_pred)
    sk_acc = dtree_sklearn(x_train, x_test, y_train, y_test)

    print("Accuracy for my model:", my_acc)
    print("Accuracy for Scikit-learn's model: ", sk_acc)

    cf_matrix(x_train, x_test, y_train, y_test) 
    
if __name__ == "__main__":
    main()