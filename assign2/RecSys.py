
# coding: utf-8

# # Recommendation System Datasets

#  - MovieLens 10M data set
#  - MovieLens 22M data set
#  - Million song data set

# # Split dataset into 60-20-20 train-validate-test partitions

# In[25]:

import os

def exists(filepath):
    return os.path.exists(filepath)


# In[43]:

# show current files
get_ipython().system(u'ls -l ml-10M100K/')


# In[44]:

if (exists('ml-10M100K/train60.dat') and exists('ml-10M100K/validation20.dat') and exists('ml-10M100K/test20.dat')):
    print "Already created files: train60.dat, validation20.dat, test20.dat"    

else:
    # sort by timestamp (4th column)
    print 'sorting file...'
    get_ipython().system(u"sort -t ':' -k4 ml-10M100K/ratings.dat > ml-10M100K/new_ratings.dat ")
    print "sorting complete."
    
    # split into 5 parts of 2 million each: train(3 parts), validation (1 part), test (1 part)
    print "splitting file..."
    get_ipython().system(u'split -l 2000000 ml-10M100K/new_ratings.dat ff')
    get_ipython().system(u'cat ffaa ffab ffac > ml-10M100K/train60.dat')
    get_ipython().system(u'mv ffad ml-10M100K/validation20.dat')
    get_ipython().system(u'mv ffae ml-10M100K/test20.dat')
    
    # remove tmp files used to create partitions
    get_ipython().system(u'rm new_ratings.dat ff*')
    print "splitting complete."    
    print "Newly created files: train60.dat, validation20.dat, test20.dat"


# # Using train data, learn ALS model

# In[ ]:




# # Using validation data, choose different regularization parameters with different latent factors

# In[ ]:




# # Using test data, test chosen model and report metric error

# In[ ]:



