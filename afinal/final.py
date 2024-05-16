import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_results(c,m):
    print("\nClasificador: ",c)
    print("Accuracy: ", m[0],"%")
    print("Precision:", m[1],"%")
    print("Recall:", m[2],"%")
    print("F1 Score:", m[3],"%")

def get_performance(test, pred):
    metrics = []
    metrics.append("{:.2f}".format(accuracy_score(test, pred) * 100))
    metrics.append("{:.2f}".format(precision_score(test, pred,
                                                   average='weighted', zero_division=0.0) * 100))
    metrics.append("{:.2f}".format(recall_score(test, pred,
                                                average='weighted') * 100))
    metrics.append("{:.2f}".format(f1_score(test, pred,
                                            average='weighted') * 100))
    return metrics

df1 = pd.read_csv('zoo2.csv')
df2 = pd.read_csv('zoo3.csv')

df = pd.concat([df1, df2])

features = df.drop(columns=['class_type', 'animal_name'])
names = df['animal_name']
target = df['class_type']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=34)

log_regression = LogisticRegression(max_iter=500)
log_regression.fit(x_train,y_train)
predict = log_regression.predict(x_test)

metrics = get_performance(y_test, predict)
print_results('Regresión lógistica',metrics)

k_neighbors = KNeighborsClassifier()
k_neighbors.fit(x_train,y_train)
predict = k_neighbors.predict(x_test)

metrics = get_performance(y_test, predict)
print_results('K Vecinos',metrics)

svc = SVC()
svc.fit(x_train,y_train)
predict = svc.predict(x_test)

metrics = get_performance(y_test, predict)
print_results('SVC',metrics)

naive = GaussianNB()
naive.fit(x_train,y_train)
predict = naive.predict(x_test)

metrics = get_performance(y_test, predict)
print_results('Naive Bayes', metrics)

