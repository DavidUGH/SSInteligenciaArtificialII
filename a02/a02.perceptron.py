import matplotlib.pyplot as plt
import numpy as np 
import random
import csv

def main():
    pn = int(input("Ingrese el numero de particiones "))
    tper = int(input("Ingrese el porcentaje de entrenamiento "))
    genper = int(input("Ingrese el porcentaje de generalización "))
    ep = int(input("Ingrese el numero de epocas de entrenamiento "))
    nu = float(input("Ingrese la tasa de aprendizaje "))
    file = open("input/spheres2d70.csv")
    w = [0.5, 0.5, 0.5, 0.5]
    ctl = csvtolist(file)
    [x, gen] = partition(ctl, pn, tper, genper)
    #print("len of x is: ")
    #print(len(x[0][0]))
    #print("len of gen is: ")
    #print(len(gen[0][0]))
    for p in x:
        w = perceptron(p, w, nu, ep)
    ax = plt.axes(projection='3d')
    for p in gen:
        for i in range (0, len(p[0])):
            o = sig(w[0] + w[1]*p[0][i] + w[2]*p[1][i] + w[3]*p[2][i])
            if o < 0:
                colorstr = 'black'
            if o > 0:
                colorstr = 'red'
                ax.plot3D(p[0][i], p[1][i], p[2][i], 'o', color=colorstr)
    plt.show()

def csvtolist(f):
    x = [[], [], [], []]
    csvreader = csv.reader(f)
    for row in csvreader:
        x[0].append(float(row[0]))
        x[1].append(float(row[1]))
        x[2].append(float(row[2]))
        x[3].append(float(row[3]))
    return x

def partition(x, n, tper, genper):
    partitions_t = []
    partitions_gen = []
    t_sz = len(x[0]) * (tper/100)
    for i in range(0, n):
        partitions_t.append([[], [], [], []])
        partitions_gen.append([[], [], [], []])

    #print("RRRRRRR")
    #print(partitions_t)
    counter = 0
    while(len(x[0]) > 0):
        if counter > n - 1:
            counter = 0
        mem = [[], [], [], []]
        r_idx = random.randint(0, len(x[0]) - 1)
        mem[0] = x[0].pop(r_idx)
        mem[1] = x[1].pop(r_idx)
        mem[2] = x[2].pop(r_idx)
        mem[3] = x[3].pop(r_idx)
        if len(x[0]) > t_sz:
            partitions_gen[counter][0].append(mem[0])
            partitions_gen[counter][1].append(mem[1])
            partitions_gen[counter][2].append(mem[2])
            partitions_gen[counter][3].append(mem[3])
        else:
            partitions_t[counter][0].append(mem[0])
            partitions_t[counter][1].append(mem[1])
            partitions_t[counter][2].append(mem[2])
            partitions_t[counter][3].append(mem[3])
        counter+=1
    return [partitions_t, partitions_gen]

def perceptron(x, w, nu, ep):
    xx = [ [], [], [] ]
    xx[0] = x[0]
    xx[1] = x[1]
    xx[2] = x[2]
    d = x[3]
    new_w = w
    #weighing the inputs
    epoch = ep
    for i in range(0, epoch):
        new_w = update_all(xx, new_w, d, nu)
    return new_w
        
def update_all(x, w, d, nu):
    new_w = w
    for i in range(0, len(x)):
        new_w = update(x[0][i], x[1][i], x[2][i], new_w, d[i], nu)
    return new_w

def update(x1, x2, x3, w, d, nu):
    new_w = [0, 0, 0, 0]
    new_w[0] = w[0] + nu*d*1
    new_w[1] = w[1] + nu*d*x1
    new_w[2] = w[2] + nu*d*x2
    new_w[3] = w[3] + nu*d*x3
    return new_w


def sig(x):
    return 1/(1 + np.exp(-x))

main()
