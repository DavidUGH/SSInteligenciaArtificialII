import matplotlib.pyplot as plt
import numpy as np 
import csv

def main():
    file = open("input/spheres2d50.csv")
    csvreader = csv.reader(file)
    x1 = []
    x2 = []
    x3 = []
    eo = []
    for row in csvreader:
        x1.append(float(row[0]))
        x2.append(float(row[1]))
        x3.append(float(row[2]))
        eo.append(float(row[3]))

    ax = plt.axes(projection='3d')
    for i in range(0, len(x1)):
        ax.plot3D(x1[i], x2[i], x3[i], 'o', color='red')
    plt.show()

main()
