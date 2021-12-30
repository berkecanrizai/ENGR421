# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 20:45:05 2021

@author: canri
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd
import math
import scipy.stats as stats

dfimg=np.genfromtxt('hw02_images.csv', delimiter=',').astype('uint8')

dflab=np.genfromtxt('hw02_labels.csv').astype('uint8')

trainimg=dfimg[:30000]
trainlab=dflab[:30000]

testimg=dfimg[30000:]
testlab=dflab[30000:]

points=[trainimg[np.asarray(trainlab==i+1)] for i in range(5)]

#print(points)

means=[points[i].mean(axis=0) for i in range(5)]
print("means")
print(means)

sample_dev = [np.std(i, axis=0) for i in points]
print("sample_dev")
print(sample_dev)

classpriors=[np.asarray(trainlab==i+1).sum()/len(trainlab) for i in range(5)]
print("classpriors")

print(classpriors)
def safelog(x):
    return(np.log(x+1e-100))

def freq(i):
    return np.log(classpriors[i])

def piSigmaSq(i):
    return np.sqrt(2*math.pi*(sample_dev[i]**2))

def getExp(i, x):
    return (-1*((x - means[i])**2) / (2*(sample_dev[i]**2)))

def firstTerm(i, x):
    return safelog((1/piSigmaSq(i))) + getExp(i, x)

def scoreFunc(i, x):
    return sum(firstTerm(i, x)) + freq(i)

def getResult(x):
    maxPt=0
    mex=scoreFunc(0, x)
    for i in range(5):
        if(scoreFunc(i, x) > mex):
            maxPt=i
            mex=scoreFunc(i, x)
    return maxPt+1



guess=list()
print('Be patient, it takes a bit long...')
print('It runs faster in Jupyter Lab for some reason')
for i in range(len(trainimg)):
    guess.append(getResult(trainimg[i]))
    
confusion_matrix = pd.crosstab(np.array(guess), np.array(trainlab), rownames = ['result'], colnames = ['actual'])
print(confusion_matrix)

guess=list()



for i in range(len(testimg)):
    guess.append(getResult(testimg[i]))
    
confusion_matrix = pd.crosstab(np.array(guess), np.array(testlab), rownames = ['result'], colnames = ['actual'])
print(confusion_matrix)