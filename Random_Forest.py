#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import all required libraries
from __future__ import division 

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

np.random.seed(0)
get_ipython().magic(u'matplotlib inline')


# In[5]:


#import the X and Y training data
X = np.genfromtxt('C:\data\X_train.txt', delimiter=None)
Y = np.genfromtxt('C:\data\Y_train.txt', delimiter=None)
X,Y = ml.shuffleData(X,Y)


# In[6]:


[Xtr,Xva,Ytr,Yva] = ml.splitData(X,Y)
Xte = np.genfromtxt('C:\data\X_test.txt', delimiter=None)


# In[13]:


class BaggedTree(ml.base.classifier):
    def __init__(self, learners):
        """Constructs a BaggedTree class with a set of learners. """
        self.learners = learners
    
    def predictSoft(self, X):
        """Predicts the probabilities with each bagged learner and average over the results. """
        n_bags = len(self.learners)
        preds = [self.learners[l].predictSoft(X) for l in range(n_bags)]
        return np.mean(preds, axis=0)
    


# In[40]:


n_bags = 7
bags = []   # self.learners
for l in range(n_bags):
    # Each boosted data is the size of the original data. 
    Xi, Yi = ml.bootstrapData(Xtr, Ytr, Xtr.shape[0])

    # Train the model on that draw
    tree = ml.dtree.treeClassify(Xi, Yi, minParent=2**6,maxDepth=100, nFeatures=6)

    bags.append(tree)


# In[41]:


bt = BaggedTree(bags)
bt.classes = np.unique(Y)

print("{0:>15}: {1:.4f}".format('Train AUC', bt.auc(Xtr, Ytr)))
print("{0:>15}: {1:.4f}".format('Validation AUC', bt.auc(Xva, Yva)))


# In[27]:


train_auc = []
validation_auc = []
train_err = []
validation_err = []
features = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

for i in features:
    print i
    n_bags = 25
    bags = []   # self.learners
    for l in range(n_bags):
        # Each boosted data is the size of the original data. 
        Xi, Yi = ml.bootstrapData(Xtr, Ytr, Xtr.shape[0])

        # Train the model on that draw
        tree = ml.dtree.treeClassify(Xi, Yi, minParent=2**6,maxDepth=100, nFeatures=i)
        bags.append(tree)
        
    bt = BaggedTree(bags)
    bt.classes = np.unique(Y)
    train_auc.append(bt.auc(Xtr, Ytr))
    validation_auc.append(bt.auc(Xva, Yva))
    train_err.append(bt.err(Xtr,Ytr))
    validation_err.append(bt.err(Xva,Yva))

f, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.plot(features, train_auc, 'r-' ,lw=4, alpha=0.75, label='Train')
ax.plot(features, validation_auc, 'g-' ,lw=4, alpha=0.75, label='Validation')
ax.set_xlabel("number of features")
ax.set_ylabel("AUC")
ax.legend(fontsize=30, loc=4)

f, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.plot(features, train_err, 'r-' ,lw=4, alpha=0.75, label='Train')
ax.plot(features, validation_err, 'g-' ,lw=4, alpha=0.75, label='Validation')
ax.set_xlabel("number of features")
ax.set_ylabel("Error")
ax.legend(fontsize=30, loc=4)

plt.show()



# In[33]:


train_auc = []
validation_auc = []
train_err = []
validation_err = []

for i in range(50):
    print(i)
    n_bags = i
    bags = []   # self.learners
    for l in range(n_bags):
        # Each boosted data is the size of the original data. 
        Xi, Yi = ml.bootstrapData(Xtr, Ytr, Xtr.shape[0])

        # Train the model on that draw
        tree = ml.dtree.treeClassify(Xi, Yi, minParent=2**6,maxDepth=100, nFeatures=4)
        bags.append(tree)
        
    bt = BaggedTree(bags)
    bt.classes = np.unique(Y)
    train_auc.append(bt.auc(Xtr, Ytr))
    validation_auc.append(bt.auc(Xva, Yva))
    train_err.append(bt.err(Xtr,Ytr))
    validation_err.append(bt.err(Xva,Yva))

f, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.plot(temp, train_auc, 'r-' ,lw=4, alpha=0.75, label='Train')
ax.plot(temp, validation_auc, 'g-' ,lw=4, alpha=0.75, label='Validation')
ax.set_xlabel("number of features")
ax.set_ylabel("AUC")
ax.legend(fontsize=30, loc=4)

f, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.plot(temp, train_err, 'r-' ,lw=4, alpha=0.75, label='Train')
ax.plot(temp, validation_err, 'g-' ,lw=4, alpha=0.75, label='Validation')
ax.set_xlabel("number of features")
ax.set_ylabel("Error")
ax.legend(fontsize=30, loc=4)

plt.show()


# In[36]:



Xte = np.genfromtxt('C:\data\X_test.txt', delimiter=None)
Yte = np.vstack((np.arange(Xte.shape[0]), bt.predictSoft(Xte)[:,1])).T
# Output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('C:\data\Y_submit.txt',Yte,'%d, %.2f',header='ID,Prob1',comments='',delimiter=',')


# In[ ]:




