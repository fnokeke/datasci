
# coding: utf-8

# ## Install networkx module

# In[4]:

get_ipython().system(u' pip install networkx')


# In[5]:


import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
py.offline.init_notebook_mode()
from plotly.offline import plot
from plotly.graph_objs import Histogram


# In[6]:

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic(u'matplotlib inline')


# ## Import Data as Nodes Into Pandas df

# In[146]:

fields = ['Actor1Code', 'Actor2Code']
df = pd.read_csv('query.csv', usecols=fields)


# ## Replace all occurrance of PSEREB with PSE because both are: Palestine

# In[147]:

df = df.replace('PSEREB', 'PSE') 


# ## Create a third column of tuples combining the Actor1Code & Actor2Code

# In[160]:

df['Pairs'] = zip(df.Actor1Code, df.Actor2Code)


# ## Add fourth column which counts pairs

# In[161]:

df['CountPairs'] = df.groupby(['Pairs','Actor1Code']).transform('count')
df.head()


# ## Enforce threshold of minimum interactions between countries

# In[162]:

light_df = df[(df.CountPairs > 20)]
light_df.head()


# ## Build a directed graph from this data using Python networkx

# In[181]:

G = nx.DiGraph() # directed graph
G.clear() # clear all lingering nodes

for r in light_df.values:
    G.add_edge(r[0],r[1], weight=r[3])


# In[165]:

plt.figure(figsize=(9,8));
plt.title('Network of countries involved in Israel-Palestine conflicts, from 1979-1989.\n (Who initiated the most wars?)', fontsize=20, fontweight='bold')

import matplotlib.patches as mpatches
isr_patch = mpatches.Patch(color='w', label='ISR=Isreal')
pse_patch = mpatches.Patch(color='w', label='PSE=Palestine')
syr_patch = mpatches.Patch(color='w', label='SYR=Syria')
lbn_patch = mpatches.Patch(color='w', label='LBN=Lebanon')

plt.legend(handles=[isr_patch, pse_patch, syr_patch, lbn_patch])

d=nx.degree(G)
nx.draw_networkx(G, with_labels=True, font_size=18, arrows=True, node_color='c', edge_color='k', alpha=0.6, node_size=[v*1000 for v in d.values()])


# ## Pagerank (count > 20)

# In[182]:

from pprint import pprint

pagerank = nx.pagerank(G)
print "Pagerank (count > 20):"
pprint(pagerank)


# In[183]:

G = nx.DiGraph() # directed graph
G.clear() # clear all lingering nodes

for r in df.values:
    G.add_edge(r[0],r[1], weight=r[3])
pagerank = nx.pagerank(G)

print "All Pagerank:"
pprint(pagerank)


# ## Plot world map

# In[184]:

# initialize every page rank value to 0 
# then set the values of pagerank
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
df.columns = ['country', 'pagerank', 'code']
df.pagerank = 0

for key in pagerank:
    df.ix[df.code==key, 'pagerank'] = pagerank[key]
    
df.head()


# In[190]:

for key in pagerank:
    print 'key=', key, 
    print df.ix[df.code==key, 'pagerank']
    print "---------"


# In[186]:

data = [ dict(
        type = 'choropleth',
        locations = df['code'],
        z = df['pagerank'],
        text = df['country'],
        colorscale = [[0.1,"rgb(5, 10, 172)"],[0.2,"rgb(40, 60, 190)"],[0.3,"rgb(70, 100, 245)"],\
            [0.4,"rgb(90, 120, 245)"],[0.5,"rgb(106, 137, 247)"],[0.6,"rgb(220, 220, 220)"]],
        autocolorscale = True,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            )
        ),
        colorbar = dict(
            title = 'pagerank'
        ),
    ) ]

layout = dict(
    title = 'Pagerank for middle east conflict',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:



