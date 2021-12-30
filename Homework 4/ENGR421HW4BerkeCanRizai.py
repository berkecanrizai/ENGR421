#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np


data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",")

x_ = data_set[1:,0]
y_ = data_set[1:,1].astype(int)

K = np.max(y_)
N = data_set.shape[0]


x_train = x_[:150]
y_train = y_[:150]


x_test = x_[150:]
y_test = y_[150:]


bin_width = 0.37
minimum_value = 1.5
maximum_value = np.max(x_train)
left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)
#p_hat = np.asarray([np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b])) for b in range(len(left_borders))]) / (N * bin_width)
p_hat = np.asarray([np.sum(y_train[(left_borders[b] < x_train) & (x_train <= right_borders[b])] / np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b]))) for b in range(len(left_borders))])

plt.figure(figsize = (10, 6))
plt.plot(x_train, y_train, "b.", markersize = 10)
plt.plot(x_test, y_test, "r.", markersize = 10)

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")    
plt.show()


sums = 0
for i in range(len(x_test)):
    indx = int((x_test[i] - minimum_value) // 0.37)
    prediction = p_hat[indx]
    errorsq = (y_test[i] - prediction) ** 2
    sums = sums + errorsq


rmse = math.sqrt(sums / len(y_test) )


print("Regressogram => RMSE is " +str(rmse) +" when h is 0.37")


data_interval = np.linspace(minimum_value, np.max(x_train), 2501)


bin_width = 0.37
p_hat = np.asarray([np.sum(y_train[((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))] / len(y_train[((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))])) for x in data_interval]) 

plt.figure(figsize = (10, 6))
plt.plot(x_train, y_train, "b.", markersize = 10)
plt.plot(x_test, y_test, "r.", markersize = 10)
plt.plot(data_interval, p_hat, "k-")
plt.show()


import bisect


sums = 0
for i in range(len(x_test)):
    indx = bisect.bisect_left(data_interval, x_test[i])
    prediction = p_hat[indx]
    errorsq = (y_test[i] - prediction) ** 2
    sums = sums + errorsq


rmse = math.sqrt(sums / len(y_test) )


print("Running Mean Smoother => RMSE is " +str(rmse) +" when h is 0.37")





data_interval = np.linspace(minimum_value, np.max(x_train), 1901)


bin_width = 0.37
p_hat = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2) * y_train) / (np.sum(1/np.sqrt(2 * math.pi) * np.exp((-0.5 * (x - x_train)**2) / bin_width**2)))  for x in data_interval])
#p_hat = p_hat * np.sum(x_train)
plt.figure(figsize = (10, 6))
plt.plot(x_train, y_train, "b.", markersize = 10)
plt.plot(x_test, y_test, "r.", markersize = 10)
plt.plot(data_interval, p_hat, "k-")
plt.show()


sums = 0
for i in range(len(x_test)):
    indx = bisect.bisect_left(data_interval, x_test[i])
    prediction = p_hat[indx]
    errorsq = (y_test[i] - prediction) ** 2
    sums = sums + errorsq


rmse = math.sqrt(sums / len(y_test) )


print("Kernel Smoother => RMSE is " +str(rmse) +" when h is 0.37")

