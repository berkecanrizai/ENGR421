# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:45:51 2021

@author: canri
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd
import math
import scipy.stats as stats

np.random.seed(421)

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

# calculate sample means
allP=[points1,points2,points3]
K=3
sample_means = [np.mean(i,axis=0) for i in allP]

print(sample_means)

sample_cov = [np.cov(i,rowvar=False) for i in allP]

print(sample_cov)

class_priors=[(class_sizes[j]/np.sum(class_sizes)) for j in range(K)]

print(class_priors)

def score_func(x,means,cova,count):
    result = np.sum((-1*np.log(2*math.pi)) - (0.5*np.log(abs(cova))) - 0.5*(np.dot(np.array(x-means).T,2*np.linalg.inv(cova))*(x-means))) + np.log(count)
    return result

def getResult(x):
    maxPt=0
    mex=score_func(x, sample_means[0], sample_cov[0], class_priors[0])
    for i in range(K):
        if(score_func(x, sample_means[i], sample_cov[i], class_priors[i]) > mex):
            maxPt=i
            mex=score_func(x, sample_means[i], sample_cov[i], class_priors[i])
    return maxPt

allTrain=np.vstack((points1, points2,points3))
allTrain=pd.DataFrame(np.vstack((points1, points2,points3)))

result=list()

for row in allTrain.iterrows():
    result.append([getResult([row[1][0],row[1][1]]),[row[1][0],row[1][1]], row[0]])
    
actual = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))

result=np.array(result)


confusion_matrix = pd.crosstab(result[:,0], actual, rownames = ['result'], colnames = ['actual'])
print(confusion_matrix)

def getScoreVal(x,i):
    return score_func(x, sample_means[i], sample_cov[i], class_priors[i])

x1_interval= np.linspace(-6, +6, 401)
x2_interval = np.linspace(-6, +6, 401)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values1, discriminant_values2, discriminant_values3 = np.zeros((len(x1_interval),len(x2_interval))), np.zeros((len(x1_interval),len(x2_interval))), np.zeros((len(x1_interval),len(x2_interval)))

for i in range(len(x1_interval)):
    for j in range(len(x2_interval)):
        x1, x2 = x1_grid[i,j],x2_grid[i,j]
        vector = np.array([x1,x2])
        discriminant_values1[i,j]= getScoreVal(vector,0)
        discriminant_values2[i,j]= getScoreVal(vector,1)
        discriminant_values3[i,j]= getScoreVal(vector,2)
        

##takes a bit of time as it makes many calculations
plt.figure(figsize = (12, 12))
for k in result[:]:
    son0=score_func(k[1], sample_means[0], sample_cov[0], class_priors[0])
    son1=score_func(k[1], sample_means[1], sample_cov[1], class_priors[1])
    son2=score_func(k[1], sample_means[2], sample_cov[2], class_priors[2])
    if(son0>son1 and son0>son2): 
        plt.plot(k[1][0],k[1][1], "r.", markersize = 10)
        if(actual[k[2]]!=1): plt.plot(k[1][0],k[1][1], "ko", markersize = 12, fillstyle = "none")
    elif(son1>son0 and son1>son2):
        if(actual[k[2]]!=2): plt.plot(k[1][0],k[1][1], "ko", markersize = 12, fillstyle = "none")
        plt.plot(k[1][0],k[1][1], "g.", markersize = 10)
    else:
        if(actual[k[2]]!=3): plt.plot(k[1][0],k[1][1], "ko", markersize = 12, fillstyle = "none")
        plt.plot(k[1][0],k[1][1], "b.", markersize = 10)

discriminant_values1[(discriminant_values2> discriminant_values1)& (discriminant_values3> discriminant_values1)]= np.nan
discriminant_values2[(discriminant_values1> discriminant_values2)& (discriminant_values3> discriminant_values2)]= np.nan
discriminant_values3[(discriminant_values1> discriminant_values3)& (discriminant_values2> discriminant_values3)]= np.nan

plt.contour(x1_grid, x2_grid, discriminant_values1-discriminant_values2, levels=0,colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values1-discriminant_values3, levels=0,colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values2-discriminant_values3, levels=0,colors="k")
plt.show()


### shows the doubt space
plt.figure(figsize = (12, 12))
x1_interval = np.linspace(-6, +6, 401)
x2_interval = np.linspace(-6, +6, 401)
for i in x1_interval:
    for j in x2_interval:
        son0=score_func([i,j], sample_means[0], sample_cov[0], class_priors[0])
        son1=score_func([i,j], sample_means[1], sample_cov[1], class_priors[1])
        son2=score_func([i,j], sample_means[2], sample_cov[2], class_priors[2])
        if(abs(son2-son1)<4 or abs(son2-son0)<4): plt.plot(i,j, "b.", markersize = 0.4)
        if(abs(son1-son0)<4  or abs(son2-son0)<4): plt.plot(i,j, "r.", markersize = 0.4)
        #if(abs(son2-son0)<4): plt.plot(i,j, "r.", markersize = 1)
for k in result[:]:        
    son0=score_func(k[1], sample_means[0], sample_cov[0], class_priors[0])
    son1=score_func(k[1], sample_means[1], sample_cov[1], class_priors[1])
    son2=score_func(k[1], sample_means[2], sample_cov[2], class_priors[2])
    if(son0>son1 and son0>son2): 
        plt.plot(k[1][0],k[1][1], "r.", markersize = 10)
        if(actual[k[2]]!=1): plt.plot(k[1][0],k[1][1], "ko", markersize = 12, fillstyle = "none")
    elif(son1>son0 and son1>son2):
        if(actual[k[2]]!=2): plt.plot(k[1][0],k[1][1], "ko", markersize = 12, fillstyle = "none")
        plt.plot(k[1][0],k[1][1], "g.", markersize = 10)
    else:
        if(actual[k[2]]!=3): plt.plot(k[1][0],k[1][1], "ko", markersize = 12, fillstyle = "none")
        plt.plot(k[1][0],k[1][1], "b.", markersize = 10)
plt.show()