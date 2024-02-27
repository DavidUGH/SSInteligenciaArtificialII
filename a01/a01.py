import matplotlib.pyplot as plt
import numpy as np 
import csv

def main():
    ep = int(input("Ingrese el numero de epocas de entrenamiento "))
    nu = float(input("Ingrese la tasa de aprendizaje "))
    file = open("input/OR_trn.csv")
    file2 = open("input/OR_tst.csv")
    csvreader = csv.reader(file)
    x1 = []
    x2 = []
    eo = []
    for row in csvreader:
        x1.append(float(row[0]))
        x2.append(float(row[1]))
        eo.append(float(row[2]))
    csvreader = csv.reader(file2)
    t_x1 = []
    t_x2 = []
    for row in csvreader:
        t_x1.append(float(row[0]))
        t_x2.append(float(row[1]))
    w = [0.5, 0.5, 0.5]
    x = [x1, x2]
    w = perceptron(x, w, eo, nu, ep)
    for i in range (0, len(t_x1)):
        o = sig(w[0] + w[1]*x[0][i] + w[2]*x[1][i])
        if o > 0:
            plt.plot(t_x1[i], t_x2[i], 'o', color='red')
        if o < 0:
            plt.plot(t_x1[i], t_x2[1][i], 'o', color='blue')
    plt.show()


def perceptron(x, w, d, nu, ep):
    new_w = w
    #weighing the inputs
    epoch = ep
    for i in range(0, epoch):
        new_w = update_all(x, new_w, d, nu)
    return new_w
        
def update_all(x, w, d, nu):
    new_w = w
    for i in range(0, len(x)):
        new_w = update(x[0][i], x[1][i], new_w, d[i], nu)
    return new_w

def update(x1, x2, w, d, nu):
    new_w = [0, 0, 0]
    new_w[0] = w[0] + nu*d*1
    new_w[1] = w[1] + nu*d*x1
    new_w[2] = w[2] + nu*d*x2
    return new_w


def sig(x):
    return 1/(1 + np.exp(-x))

main()
