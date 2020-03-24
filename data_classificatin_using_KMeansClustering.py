#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv('College_Data',index_col=0)


# In[6]:


df.head()


# In[10]:


df.describe()


# In[9]:


df.info()


# In[16]:


sns.scatterplot(x='Room.Board',y='Grad.Rate',data=df,hue='Private')


# In[17]:


sns.scatterplot(x='Outstate',y='F.Undergrad',data=df,hue='Private')


# In[27]:


fgrid=sns.FacetGrid(df,hue='Private',size=6)
fgrid.map(plt.hist,'Outstate',bins=50)


# In[28]:


fgrid=sns.FacetGrid(df,hue='Private',size=6)
fgrid.map(plt.hist,'Grad.Rate',bins=50)


# In[29]:


df[df['Grad.Rate']>100]


# In[30]:


df['Grad.Rate']['Cazenovia College']=100


# In[31]:


df[df['Grad.Rate']>100]


# In[32]:


fgrid=sns.FacetGrid(df,hue='Private',size=6)
fgrid.map(plt.hist,'Grad.Rate',bins=50)


# In[33]:


#KMC 


# In[34]:


from sklearn.cluster import KMeans


# In[35]:


km=KMeans(n_clusters=2)


# In[37]:


km.fit(df.drop('Private',axis=1))


# In[38]:


km.cluster_centers_


# In[39]:


#evaluation


# In[40]:


def change(str):
    if str=='Yes':
        return 1
    else:
        return 0
    
df['Cluster']=df['Private'].apply(change)


# In[41]:


df.head()


# In[42]:


from sklearn.metrics import classification_report,confusion_matrix


# In[46]:


print(confusion_matrix(df['Cluster'],km.labels_))
print()
print(classification_report(df['Cluster'],km.labels_))


# In[ ]:




