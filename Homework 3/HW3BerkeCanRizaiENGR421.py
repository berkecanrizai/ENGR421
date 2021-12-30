# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:21:30 2021

@author: canri
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd
import math
import scipy.stats as stats

np.random.seed(521)

class_means = np.array([[0, -2.5, 2.5],[2.5,-2,-2]])
# standard deviation parameters
class_deviations0 = np.array([[3.2, 0],[0,1.2]])
class_deviations1 = np.array([[1.2, 0.8],[0.8,1.2]])
class_deviations2 = np.array([[1.2, -0.8],[-0.8,1.2]])
# sample sizes
class_sizes = np.array([120, 80, 100])

points1 = np.random.multivariate_normal(class_means[:,0], class_deviations0, class_sizes[0])

points2 = np.random.multivariate_normal(class_means[:,1], class_deviations1, class_sizes[1])

points3 = np.random.multivariate_normal(class_means[:,2], class_deviations2, class_sizes[2])

y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))

plt.figure(figsize = (10, 10))
#plt.plot(xy=[points1[:,0],points1[:,1]], marker="k.", markersize = 10,color="b")
plt.plot(points1[:,0], points1[:,1], "k.", markersize = 10,color="r")
plt.plot(points2[:,0], points2[:,1], "k.", markersize = 10,color="g")
plt.plot(points3[:,0], points3[:,1], "k.", markersize = 10,color="b")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

eta = 0.01
epsilon = 0.001

K=3

def dnum(w, x, w0):
    return (1+np.exp(-1 * (np.matmul(x, w) + w0)))

def sigmoid(x, w, w0):
    scor=np.exp(-1 * (np.matmul(x, w) + w0))
    return 1/(1+scor)

allP=np.concatenate((points1, points2, points3), axis=0)
x=allP
y_truth=y

def gradient_w0(y_truth, y_predicted):
    return (-np.sum((y_truth - y_predicted) * (y_predicted - (y_predicted**2)), axis = 0))

def addDim(arr):
    return np.array([arr]*2).T

w = np.random.uniform(low = -0.01, high = 0.01, size = (x.shape[1], 3))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, 3))

def gradient_w(X, y_truth, y_predicted):
    result = [np.sum(addDim(y_truth[:, c] - y_predicted[:, c]) * addDim(y_predicted[:, c]) * addDim(y_predicted[:, c] - 1) * X, axis=0) for c in range(K)]
    return np.array(result).T

N = 300


Y_truth = np.zeros((N, 3)).astype(int)
Y_truth[range(N), y_truth - 1] = 1

y_truth=Y_truth

w = np.random.uniform(low = -0.01, high = 0.01, size = (x.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))
iteration = 1
objective_values = []
while 1:
    #if(iteration%1000==0): print(w, np.sqrt(np.sum(w0 - w0_old) ** 2 + np.sum((w - w_old) ** 2)))
    y_predicted=sigmoid(x, w, w0)
    objective_values = np.append(objective_values, 0.5 * np.sum((y_truth-y_predicted) **2))
    #old=np.sqrt((w0-w0_old)**2 + np.sum(w-w_old)**2)
    w_old=w
    w0_old=w0

    
    w = w - eta*gradient_w(x, y_truth, y_predicted)
    w0 = w0 - eta*gradient_w0(y_truth, y_predicted)
    
    if np.sqrt(np.sum(w0 - w0_old) ** 2 + np.sum((w - w_old) ** 2)) < epsilon:
        break
    #print(np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((w - w_old)**2)))
    #print(np.sum(w0 - w0_old), w0, w0_old)
    iteration=iteration+1
    
print(w)
print(w0) 

# plot objective function during iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# calculate confusion matrix
a = np.argmax(y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(a, y, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)

# evaluate discriminant function on a grid
plt.figure(figsize = (10, 10))

color='r.'
W=w


X = np.vstack((points1, points2, points3))
x1_interval = np.linspace(-8, +8, 1201)
x2_interval = np.linspace(-8, +8, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
for c in range(K):
    discriminant_values[:,:,c] = W[0, c] * x1_grid + W[1, c] * x2_grid + w0[0, c]
    
plt.plot(points1[:,0], points1[:,1], "k.", markersize = 10,color="r")
plt.plot(points2[:,0], points2[:,1], "k.", markersize = 10,color="g")
plt.plot(points3[:,0], points3[:,1], "k.", markersize = 10,color="b")

for i in range(len(allP)):
    if y[i]==1 and a[i]!=1:
            plt.plot(allP[i][0], allP[i][1], "ko", markersize = 12, fillstyle = "none")
    if y[i]==2:
        if a[i]!=2:
            plt.plot(allP[i][0], allP[i][1], "ko", markersize = 12, fillstyle = "none")
        color = 'g.'
    if y[i]==3:
        if a[i]!=3:
            plt.plot(allP[i][0], allP[i][1], "ko", markersize = 12, fillstyle = "none")
        color = 'b.'
    
    
A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C



plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
