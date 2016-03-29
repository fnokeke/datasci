
# coding: utf-8

# In[4]:

###
# @author: Nwamaka Nzeocha and Fabian Okeke
# @course: CS 5304/Data Science in the Wild
####


# In[ ]:

import random
import os.path
import pandas as pd
import numpy as np
import csv

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
py.offline.init_notebook_mode()
from plotly.offline import plot
from plotly.graph_objs import Histogram

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().magic(u'matplotlib inline')


# In[137]:

def exists(filepath):
    return os.path.exists(filepath)


# # Split 48 million rows to 10 million(training) & 38 million(testing)

# In[3]:

if (exists('train10M.txt') and exists('test38M.txt')):
    print "train10M.txt and test38M.txt already created before."
else:
    get_ipython().system(u'split -l 2000000 train.txt ff')
    get_ipython().system(u'cat ffaa ffat ffaf ffaq ffaj > train10M.txt')
    get_ipython().system(u'rm ffaa ffat ffaf ffaq ffaj')
    get_ipython().system(u'cat ff* > test38M.txt')
    get_ipython().system(u'rm ff*')
    print "Newly created files: train10M.txt and test38M.txt."


# # Split train10M data into 3 partitions

# In[4]:

if (exists('train5M.txt') and 
    exists('validation2M.txt') and 
    exists('test3M.txt')):
    print "train5M.txt, validation2M.txt, test3M.txt already created before."
else:
    get_ipython().system(u'split -l 1000000 train10M.txt ff')
    get_ipython().system(u'cat ffaa ffaj ffad ffaf ffah > train5M.txt    ')
    get_ipython().system(u'cat ffai ffae > validation2M.txt')
    get_ipython().system(u'cat ffac ffag ffab > test3M.txt')
    get_ipython().system(u'rm ff*')
    print "Newly created files: train5M.txt, validation2M.txt, test3M.txt."


# # Load train5M

# In[78]:

df_train = pd.read_table('train5M.txt', header=None)


# # Plot histograms

# In[ ]:

# plotly allows you to only make 50 plots per day
for i in range(1, 40):
    label = "feat" + str(i)
    trace1 = go.Histogram(x=df_train[i])
    
    data = [trace1]
    layout = go.Layout(
        title= label + ' histogram',
        xaxis=dict(title= label + 'values', autorange=True),
        yaxis=dict(title='log occurance count', type='log', autorange=True)
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


# # Select categorical features that should be kept out of all 26 possible cases

# In[134]:

selected_categ_cols = [15,20,23,24,26,27,28,30,31,36,37,39]
len(selected_categ_cols)


# # Normalize integer cols and compute rate-value on categorical cols

# In[125]:

def get_train_xy(mdf, selected_categorical_cols):
    """
        normalize integer columns of dataframe and perform weighted freq count on selected categorical columns
        @param mdf: massive/large dataframe to be operated on (cols 1-13 are assumed to be integer columns)
        @param selected_categ_cols: hashed string columns to be converted to float values
        @return X: dataframe same as mdf but with 1 less column than mdf
        @return y: 1D array
    """
    int_cols = range(1,14)
    cols_to_keep = int_cols + selected_categorical_cols  
    y = mdf[0].values
    X = mdf[cols_to_keep].fillna(0)
    
    # normalize integers features
    X[int_cols] = X[int_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # compute weighted freq: each column entry is freq/cumulative_freq
    cum_freq = len(mdf) * 1.0
    for col in selected_categorical_cols:
        freq = m_count_distinct(X[col].values)
        X[col] = X[col].apply(lambda x: freq.get(x)/cum_freq) 
    return X, y

def m_count_distinct(col):
    """
        count distinct values in all columns
        @param col: numpy 1D array
        @return: dictionary of value, freq
    """
    d = {}
    for x in list(col):
        if type(x) == np.float64 and np.isnan(x):
            k = 'nan-' + col
            d[k] = d.get(k, 0) + 1
        else:
            d[x] = d.get(x, 0) + 1
    return d

def score_higher_freq(y):
    """
        accuracy score for baseline method that always predicts the value with higher frequency
        @param y: 1D array
        @return: float value between 0.0 and 1.0
    """
    d = {}
    for i in y:
        d[i] = d.get(i,0) + 1
    return max(d.values())/float(len(y))


# # Prepare training and validation data

# In[112]:

X_train, y_train = get_train_xy(df_train, selected_categ_cols)


# In[113]:

df_va = pd.read_table('validation2M.txt', header=None)
X_va, y_va = get_train_xy(df_va, selected_categ_cols)


# # Accuracy score if we naively predict 1 every time

# In[ ]:

print "Baseline score:", score_higher_freq(y_va)


# # Fit models and predict

# In[122]:

model_NB = GaussianNB()
model_NB.fit(X_train, y_train)
predicted_NB = model_NB.predict(X_va)

model_NB


# In[29]:

model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)
predicted_LR = model_LR.predict(X_va)

model_LR


# In[34]:

model_RF = RandomForestClassifier(n_estimators=10)
model_RF.fit(X_train, y_train)
predicted_RF = model_RF.predict(X_va)

model_RF


# # ROC curves, Accuracy scores, AUC scores

# In[36]:

get_ipython().magic(u'matplotlib inline')

y_predictions = {
    'Naive Bayes': predicted_NB,
    'Logistic Regression': predicted_LR,
    'Random Forest': predicted_RF,
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


# # Test 3M

# In[37]:

df_test = pd.read_table('test3M.txt', header=None)
X_test, y_test = get_train_xy(df_test, selected_categ_cols)


# In[38]:

predicted_test = model_LR.predict(X_test)
print "Test3M accuracy:", metrics.accuracy_score(y_test, predicted_test)
print "Test3M auc:", metrics.roc_auc_score(y_test, predicted_test)


# # Test38M: load files in chunks and aggregate metrics on chunks

# In[ ]:

all_chunks_accuracy = []
all_chunks_auc = []
chunksize = 10 ** 6
i=1

for chunk in pd.read_table("test38M.txt", chunksize=chunksize):
    chunk.columns = range(40) 
    X_chunk, y_chunk = get_train_xy(chunk, selected_categ_cols)
    
    # predict on y-chunk
    predicted_chunk = model_LR.predict(X_chunk)
    chunk_accuracy = metrics.accuracy_score(y_chunk, predicted_chunk)
    chunk_auc = metrics.roc_auc_score(y_chunk, predicted_chunk)
    
    all_chunks_accuracy.append(chunk_accuracy)
    all_chunks_auc.append(chunk_auc)
    print "Done with chunk %d." % i
    i += 1

print "================="
print "avg accuracy", sum(all_chunks_accuracy)/len(all_chunks_accuracy)
print "avg auc", sum(all_chunks_auc)/len(all_chunks_auc)

