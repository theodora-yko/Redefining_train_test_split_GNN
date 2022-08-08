#!/usr/bin/env python
# coding: utf-8

# In[1]:


#datasets
from torch_geometric.data import Data
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.datasets import Planetoid 
from torch_geometric.transforms import NormalizeFeatures


# In[2]:


# download datasets
datasets = ['cora_A', 'citeseer_A', 'pubmed_A', 'cora_P', 'citeseer_P', 'pubmed_P']

# AGD
cora_A = AttributedGraphDataset(root='AGD', name='Cora')
citeseer_A = AttributedGraphDataset(root='AGD', name='CiteSeer')
pubmed_A =  AttributedGraphDataset(root='AGD', name='PubMed')

# planetoid
cora_P = Planetoid(root='Planetoid', name='Cora', transform=NormalizeFeatures())
citeseer_P = Planetoid(root='Planetoid', name='CiteSeer',transform=NormalizeFeatures())
pubmed_P =  Planetoid(root='Planetoid', name='PubMed',transform=NormalizeFeatures())


# In[ ]:




