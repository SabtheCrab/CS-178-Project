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


bt = BaggedTree(bags)
bt.classes = np.unique(Y)

print("{0:>15}: {1:.4f}".format('Train AUC', bt.auc(Xtr, Ytr)))
print("{0:>15}: {1:.4f}".format('Validation AUC', bt.auc(Xva, Yva)))

Xte = np.genfromtxt('C:\data\X_test.txt', delimiter=None)
Yte = np.vstack((np.arange(Xte.shape[0]), bt.predictSoft(Xte)[:,1])).T
# Output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('C:\data\Y_submit.txt',Yte,'%d, %.2f',header='ID,Prob1',comments='',delimiter=',')
