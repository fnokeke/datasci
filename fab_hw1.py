
# coding: utf-8

# In[9]:

import random
import os.path

def exists(filepath):
    return os.path.exists(filepath)


# # Split 48 million rows: 10 million(training) & 38 million(testing)

# In[10]:

get_ipython().system(u'wc -l train.txt')


# In[12]:

get_ipython().system(u'split -l 4775064 train.txt newfiles')


# In[2]:

if (exists('train10M.txt') and exists('test38M.txt')):
    print "train10M.txt and test38M.txt already created before."
else:
    ftr = open('train10M.txt', 'a')
    fte = open('test38M.txt', 'a')

    count = 0
    with open('train.txt') as f:
        for line in f:
            if count <= 10000000:
                ftr.write(line)
                count += 1
            else:
                fte.write(line)
    ftr.close()
    fte.close()    
    print "Newly created files: train10M.txt and test38M.txt."


# # Split training data into 3 partitions

# In[3]:

if (exists('train5M.txt') and 
    exists('validation2M.txt') and 
    exists('test3M.txt')):
    print "train5M.txt, validation2M.txt, test3M.txt already created before."
    
else:
    ftr = open('train5M.txt', 'a')
    fva = open('validation2M.txt', 'a')
    fte = open('test3M.txt', 'a')

    with open('train10M.txt') as f:
        for line in f:
            value = random.randint(1,100)
            if value <= 50:
                ftr.write(line)
            elif value <= 70:
                fva.write(line)
            else:
                fte.write(line)

    ftr.close()
    fva.close()
    fte.close()
    print "Newly created files: train5M.txt, validation2M.txt, test3M.txt."


# # Create subsets of partitions for debug purposes

# In[4]:

def create_subset(finput, foutput, noOfRows):
    count = 0
    fout = open(foutput, 'a')
    
    with open(finput) as fin:
        for line in fin:
            while count < noOfRows:
                fout.write(line)
                count += 1
    fout.close()


# In[5]:

if (exists('debug_train5M.txt') and 
    exists('debug_validation2M.txt') and 
    exists('debug_test3M.txt')):
    print "debug_train5M.txt, debug_validation2M.txt, debug_test3M.txt already created before."
    
else:
    create_subset('train5M.txt', 'debug_train5M.txt', 10000)
    create_subset('validation2M.txt', 'debug_validation2M.txt', 10000)
    create_subset('test3M.txt', 'debug_test3M.txt', 10000)
    print "Newly created files: debug_train5M.txt, debug_validation2M.txt, debug_test3M.txt."


# # Load test data

# In[6]:

import pandas as pd
test_data = pd.read_table('test3M.txt')
print test_data.shape
test_data.head()


# In[8]:

get_ipython().magic(u'store test_data')


# # Process training data

# In[9]:

df = pd.read_table('train5M.txt')
print "df shape:", df.shape


# In[10]:

get_ipython().magic(u'store df')


# # Show column names

# In[14]:

print "There are %d columns." % len(df.columns.values)
df.columns.values


# In[15]:

df.head()


# # Rename columns

# In[41]:

new_col_names = ['f' + str(num) for num in range(len(df.columns.values))]
df.columns = new_col_names
df = df.drop('f0', 1) # first col is irrelevant
df.head()


# ### Find out which columns are all null (THIS SHOULD BE REMOVED)

# In[42]:

[(col, df[col].isnull().sum()) for col in problematic_cols]


# ### Reasons to drop a cell
#  - if the entire column is all null and part of categorical features, then just drop the column as we may imply that that removing that category doesn't do much harm. Further, replacing NAN with 0's is misleading as the entire category should be hashed strings not numbers.
#   - CHECK THAT YOUR '' IS REPLACED WITH NAN

# In[28]:

df_no_NAN = df.dropna(axis=1, how='all')
df_no_NAN = df_no_NAN.fillna(0)
df_no_NAN.head()


# # Visualization

# In[29]:

get_ipython().magic(u'matplotlib inline')

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[30]:

df.head()


# ### Plot histogram for all columns

# In[ ]:

histogramCount = 0
typeErrorCount = 0
valueErrorCount = 0
problematic_cols = []

all_columns = list(df.columns.values)
selected_columns =  all_columns[1:2] + all_columns[24:26]
for col in selected_columns:
    try:
        ax = df_no_NAN[[col]].apply(pd.value_counts).plot(kind='hist', logy='True')
        ax.set_xlabel("values")
        ax.set_title("Histogram (col='" + col + "')")
        histogramCount += 1
    except TypeError as err:
        print "Problematic col:" + col
        print "Error details:", err
        problematic_cols.append(col)
        typeErrorCount += 1
    except ValueError as err:
        print "Value error for col:" + col
        print "Error details:", err
        valueErrorCount += 1
        
print "Histogram printed:", histogramCount
print "TypeError count:", typeErrorCount
print "ValueError count:", valueErrorCount


# # Summary stats for only integer columns

# In[37]:

integer_columns = all_columns[:13]
df[integer_columns].describe()


# # Normalize integer columns after transpose

# In[ ]:

df_norm = df[integer_columns].T.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.head()


# # Select features to be used

# In[ ]:

cols_to_keep =  
    all_columns[:13] +  
    all_columns[18:21] + 
    ['f22', 'f24', 'f26'] +
    all_columns[27:31] + 
    all_columns[33:36]
    
cols_to_keep


# # One hot encoding

# In[ ]:

#cols_to_keep = all_columns[:13] + all_columns[17:24]
df_onehot = pd.get_dummies(df[cols_to_keep])
print df_onehot.shape
df_onehot.head()


# # Model and Evaluation

# In[495]:

X = df_onehot
y = test_data.index.values


# ### Logistic Regression

# In[502]:

from sklearn.linear_mode import LogisticRegression
model = LogisticRegression()
model.fit(X,y)


# In[505]:

model.score(X,y)


# ### Split up the data set into test and training sets

# In[510]:

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
testLogRModel = LogisticRegression()
testLogRModel.fit(x_train, y_train)


# In[511]:

testLogRModel.score(X,y)


# ### Predict class labels for the test set

# In[514]:

predicted = testLogRModel.predict(x_test)
print predicted


# ### AUC score from ROC

# In[517]:

from sklearn import metrics
print "accuracy:", metrics.accuracy_score(y_test, predicted)
# print "auc:", metrics.roc_auc_score(y_test, predicted)


# ### Perform cross validation using n folds

# In[526]:

from sklearn.cross_validation import cross_val_score
#n = 5
#scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=n)
#print scores
#print scores.mean()


# ### Random Forest Classifier

# In[523]:

from sklearn.ensemble import RandomForestClassifier
clf_random_forest = RandomForestClassifier(n_estimators=10)
clf_random_forest.fit(X,y)


# In[525]:

clf_random_forest.score(X,y)


# In[ ]:



