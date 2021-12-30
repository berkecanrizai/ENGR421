#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt

data_set = np.genfromtxt("hw06_images.csv", delimiter = ",")
labels = np.genfromtxt("hw06_labels.csv", delimiter = ",")


# In[2]:


X_train = data_set[:1000]
y_train = labels[:1000]


# In[3]:


X_test = data_set[1000:]
y_test = labels[1000:]


# In[4]:


# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)


# In[5]:


N_train = len(y_train)
D_train = X_train.shape[1]


# In[ ]:





# In[ ]:





# In[6]:


y1 = np.array([1 if i == 1 else -1 for i in y_train]).astype(float)
y2 = np.array([1 if i == 2 else -1 for i in y_train]).astype(float)
y3 = np.array([1 if i == 3 else -1 for i in y_train]).astype(float)
y4 = np.array([1 if i == 4 else -1 for i in y_train]).astype(float)
y5 = np.array([1 if i == 5 else -1 for i in y_train]).astype(float)


# In[ ]:





# In[7]:


def getSVM(y_train, s = 10, C = 10):
    # calculate Gaussian kernel
    K_train = gaussian_kernel(X_train, X_train, s)
    yyK = np.matmul(y_train[:,None], y_train[None,:]) * K_train

    # set learning parameters

    epsilon = 1e-3

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_train[None,:])
    b = cvx.matrix(0.0)

    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    print(C)
    return alpha, w0, K_train


# In[8]:


'''# calculate predictions on training samples
f_predicted = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0

# calculate confusion matrix
y_predicted = 2 * (f_predicted > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)'''


# In[9]:


a1, w01, ktrain1 = getSVM(y1)


# In[10]:


a2, w02, ktrain2 = getSVM(y2)


# In[11]:


a3, w03, ktrain3 = getSVM(y3)


# In[12]:


a4, w04, ktrain4 = getSVM(y4)


# In[13]:


a5, w05, ktrain5 = getSVM(y5)


# In[14]:


# calculate predictions on training samples
f_predicted1 = np.matmul(ktrain1, y1[:,None] * a1[:,None]) + w01

# calculate confusion matrix
y_predicted1 = 2 * (f_predicted1 > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted1, N_train), y1, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


# In[15]:


# calculate predictions on training samples
f_predicted2 = np.matmul(ktrain2, y2[:,None] * a2[:,None]) + w02

# calculate confusion matrix
y_predicted2 = 2 * (f_predicted2 > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted2, N_train), y2, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


# In[16]:


# calculate predictions on training samples
f_predicted3 = np.matmul(ktrain3, y3[:,None] * a3[:,None]) + w03

# calculate confusion matrix
y_predicted3 = 2 * (f_predicted3 > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted3, N_train), y3, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


# In[17]:


# calculate predictions on training samples
f_predicted4 = np.matmul(ktrain4, y4[:,None] * a4[:,None]) + w04

# calculate confusion matrix
y_predicted4 = 2 * (f_predicted4 > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted4, N_train), y4, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


# In[18]:


# calculate predictions on training samples
f_predicted5 = np.matmul(ktrain5, y5[:,None] * a5[:,None]) + w05

# calculate confusion matrix
y_predicted5 = 2 * (f_predicted5 > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted5, N_train), y5, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


# In[19]:


preds = np.argmax([f_predicted1,f_predicted2,f_predicted3,f_predicted4,f_predicted5], axis=0)
preds = preds + 1


# In[20]:


confusion_matrix = pd.crosstab(np.reshape(preds, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


# In[21]:


s = 10
K_test = gaussian_kernel(X_test, X_train, s)
#np.matmul(ktrain4, y4[:,None] * a4[:,None])
f_predicted1 = np.matmul(K_test, y1[:,None] * a1[:,None]) + w01


# In[22]:


f_predicted2 = np.matmul(K_test, y2[:,None] * a2[:,None]) + w02


# In[23]:


f_predicted3 = np.matmul(K_test, y3[:,None] * a3[:,None]) + w03


# In[24]:


f_predicted4 = np.matmul(K_test, y4[:,None] * a4[:,None]) + w04


# In[25]:


f_predicted5 = np.matmul(K_test, y5[:,None] * a5[:,None]) + w05


# In[26]:


preds = np.argmax([f_predicted1,f_predicted2,f_predicted3,f_predicted4,f_predicted5], axis=0)
preds = preds + 1


# In[27]:


confusion_matrix = pd.crosstab(np.reshape(preds, 4000), y_test, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


# In[28]:


trainAcc = list()
testAcc = list()
acc = 0
cList = [0.1, 1, 10, 100, 1000]
for c in cList:
    print(c)
    a1, w01, ktrain1 = getSVM(y1, C = c)
    a2, w02, ktrain2 = getSVM(y2, C = c)
    a3, w03, ktrain3 = getSVM(y3, C = c)
    a4, w04, ktrain4 = getSVM(y4, C = c)
    a5, w05, ktrain5 = getSVM(y5, C = c)
    
    f_predicted1 = np.matmul(ktrain1, y1[:,None] * a1[:,None]) + w01
    f_predicted2 = np.matmul(ktrain2, y2[:,None] * a2[:,None]) + w02
    f_predicted3 = np.matmul(ktrain3, y3[:,None] * a3[:,None]) + w03
    f_predicted4 = np.matmul(ktrain4, y4[:,None] * a4[:,None]) + w04
    f_predicted5 = np.matmul(ktrain5, y5[:,None] * a5[:,None]) + w05
    preds = np.argmax([f_predicted1,f_predicted2,f_predicted3,f_predicted4,f_predicted5], axis=0)
    preds = preds + 1
    
    confusion_matrix = pd.crosstab(np.reshape(preds, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
    
    acc = np.sum(np.diag(np.array(confusion_matrix))) / np.sum(np.array(confusion_matrix))
    trainAcc.append(acc)
    
    s = 10
    K_test = gaussian_kernel(X_test, X_train, s)
    #np.matmul(ktrain4, y4[:,None] * a4[:,None])
    f_predicted1 = np.matmul(K_test, y1[:,None] * a1[:,None]) + w01
    f_predicted2 = np.matmul(K_test, y2[:,None] * a2[:,None]) + w02
    f_predicted3 = np.matmul(K_test, y3[:,None] * a3[:,None]) + w03
    f_predicted4 = np.matmul(K_test, y4[:,None] * a4[:,None]) + w04
    f_predicted5 = np.matmul(K_test, y5[:,None] * a5[:,None]) + w05
    
    preds = np.argmax([f_predicted1,f_predicted2,f_predicted3,f_predicted4,f_predicted5], axis=0)
    preds = preds + 1
    
    confusion_matrix = pd.crosstab(np.reshape(preds, 4000), y_test, rownames = ['y_predicted'], colnames = ['y_train'])
    acc = np.sum(np.diag(np.array(confusion_matrix))) / np.sum(np.array(confusion_matrix))
    
    testAcc.append(acc)


# In[29]:


trainAcc


# In[30]:


testAcc


# In[ ]:





# In[31]:


(10**np.log10(cList))


# In[32]:


import matplotlib
fig1, ax1 = plt.subplots(figsize=(10, 6))
plt.figure(figsize = (10, 6))
ax1.plot(cList, trainAcc,"bo-", label = 'Training')
ax1.plot(cList, testAcc,"ro-", label = 'Test')
ax1.legend(loc='upper left')
ax1.set_xscale('log')
#ax1.set_xticks(cList)
ax1.set_xticks(ticks = cList)
#ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


# In[ ]:





# In[ ]:




