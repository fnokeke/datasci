
# coding: utf-8

# In[23]:

####
# @author: Nwamaka Nzeocha and Fabian Okeke
# @course: CS 5304/Data Science in the Wild
####


# # Environment setup

#  - useful tensor flow notebook: http://bit.ly/1NjhcfQ
#  - pip install scikit-learn
#  - pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl

# In[2]:

import random
import os.path
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().magic(u'matplotlib inline')


# In[2]:

def exists(filepath):
    return os.path.exists(filepath)


# # Get dataset

# In[ ]:

X_train, y_train = '', ''
X_va, y_va = '', ''


# # Fit logistic regression model

# In[ ]:

model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)
predicted_LR = model_LR.predict(X_va)

model_LR


# # Fit multi-layer perceptron model

# In[ ]:

model_MLP = MultiLayerPerceptron()
model_MLP.fit(X_train, y_train)
predicted_MLP = model_MLP.predict(X_va)

model_MLP


# # Pre calibration Brier loss and Brier curves

# In[ ]:

print "pre-calibration brier loss:"
print "pre-calibration brier curve:"


# # Post calibration Brier loss and Brier curves

# In[ ]:

print "post-calibration brier loss:"
print "post-calibration brier curve:"


# # ROC curves, Accuracy scores, AUC scores

# In[ ]:

y_predictions = {
    'Logistic Regression': predicted_LR,
    'Tensor Flow': predicted_MLP,
}

fig, axes = plt.subplots(1, len(y_predictions), sharey=True)
fig.set_size_inches(15,5)
i = 0

for title, prediction in y_predictions.items():
    accuracy = metrics.accuracy_score(y_va, prediction)
    fulltitle = title + ' (accuracy = %0.2f)' % accuracy
    fpr, tpr, thresholds = metrics.roc_curve(y_va, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    label = 'AUC = %0.2f' % auc

    axes[i].plot(fpr, tpr, label=label)
    axes[i].set_title(fulltitle)
    axes[i].legend(loc='lower right')
    i += 1

