#!/usr/bin/env python
# coding: utf-8

# # READ_ME
# 
# - adapated example from Pytorch.Geometric website using GraphSAGE instead of GCN
# - datasets from Cora, CiteSeer, PubMed 
# - loss function: cross_entropy() / noteL the example uses nll instead but this returns a very low accuracy for some reason 
# - lr: tried 0.01, 0.001, epoch: tried 200, 500

# ## libraries

# In[25]:


# libraries
import torch
from torch_geometric.data import Data
import pandas as pd 
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit  
import torch.nn.functional as F
import torch.nn
from torch_geometric.nn import SAGEConv
import matplotlib.pyplot as plt

#datasets
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.datasets import Planetoid 
from torch_geometric.transforms import NormalizeFeatures


# ## algorithm

# In[2]:


cora_A = AttributedGraphDataset(root='AGD', name='Cora')
citeseer_A = AttributedGraphDataset(root='AGD', name='CiteSeer')
pubmed_A =  AttributedGraphDataset(root='AGD', name='PubMed')

# planetoid
cora_P = Planetoid(root='Planetoid', name='Cora', transform=NormalizeFeatures())
citeseer_P = Planetoid(root='Planetoid', name='CiteSeer',transform=NormalizeFeatures())
pubmed_P =  Planetoid(root='Planetoid', name='PubMed',transform=NormalizeFeatures())


# In[3]:


#implement a two-layer GraphSage from GCN example:

class GraphSage(torch.nn.Module):
    def __init__(self, attributes, classes):
        super(GraphSage, self).__init__() 
        self.conv1 = SAGEConv(attributes, 32) 
        self.conv2 = SAGEConv(32, classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index 

        x = self.conv1(x, edge_index) #layer 1 
        x = F.relu(x)
        x = F.dropout(x, training=self.training) #layer 2 
        x = self.conv2(x, edge_index)

        return x


# In[14]:


class accuracy():
    def __init__(self, device, dataset, data):
        self.model = GraphSage(dataset.num_node_features, dataset.num_classes).to(device)
        self.data = data
    
    def train(self, test_size):
        loss_vals= []

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)
        for epoch in range(test_size): 
            optimizer.zero_grad()
            out = self.model(self.data)
            
            loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask]) ## note: not crossentropyloss(), cross_entropy()
            loss_vals.append((loss, epoch)) 
            loss.backward()
            optimizer.step()
        
    def evaluate(self):
        self.model.eval()
        pred = self.model(self.data).argmax(dim=1)
        correct = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).sum()
        acc = int(correct) / int(self.data.test_mask.sum())
        print(f'Accuracy: {acc:.4f}')


# In[11]:


## define transform_nodes
transform_nodes = RandomNodeSplit(split = 'test_rest') # RandomNodeSplit is a class 


# In[12]:


# split into training and testing data points
# caution use data not a dataset 
cora_A_data = transform_nodes(cora_A[0])
citeseer_A_data = transform_nodes(citeseer_A[0])
pubmed_A_data = transform_nodes(pubmed_A[0]) # caution, don't reiterate


# ## training
# * learning rate = 0.01, 0.001

# In[15]:


#cora_A 
device1A = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset1A = cora_A
data1A = cora_A_data.to(device1A)
cora_A0 = accuracy(device1A, dataset1A, data1A)
cora_A0.train(500)
cora_A0.evaluate()


# In[16]:


#cora_P
device1P = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset1P = cora_P
data1P = cora_P[0].to(device1P)
cora_P0 = accuracy(device1P, dataset1P, data1P)
cora_P0.train(500)
cora_P0.evaluate() 


# In[17]:


#citeseer_A
device2A = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset2A = citeseer_A
data2A = citeseer_A_data.to(device2A)
citeseer_A0 = accuracy(device2A, dataset2A, data2A)
citeseer_A0.train(500)
citeseer_A0.evaluate()


# In[18]:


#citeseer_p
device2P = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset2P = citeseer_P
data2P = citeseer_P[0].to(device2P)
citeseer_P0 = accuracy(device2P, dataset2P, data2P)
citeseer_P0.train(500)
citeseer_P0.evaluate()


# In[19]:


#pubmed_A
device3A = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset3A = pubmed_A
data2A = pubmed_A_data.to(device3A)
pubmed_A0 = accuracy(device3A, dataset3A, data2A)
pubmed_A0.train(500)
pubmed_A0.evaluate()


# In[23]:


#pubmed_P
device3P = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset3P = pubmed_P
data2P = pubmed_P[0].to(device3P)
pubmed_P0 = accuracy(device3P, dataset3P, data2P)
pubmed_P0.train(500)
pubmed_P0.evaluate()


# ## result

# In[36]:


# accuracy_1 = [0.3013, 0.1440, 0.1579, 0.0780, 0.3870, 0.4070]
accuracy_1 = {'cora_A': [0.3013,  0.1262, 0.6983], 'cora_P': [0.1440, 0.0910, 0.7940], 'citeseer_A':[0.1579, 0.1371, 0.5490], 'citeseer_P' : [0.0, 0.0770, 0.7130], 'pubmed_A':[0.3870, 0.3997, 0.7364], 'pubmed_P': [0.0, 0.4130, 0.7740]}
result = pd.DataFrame.from_dict(data=accuracy_1, orient='index', columns=['lr = 0.1, epoch 200, nll', 'lr = 0.01, epoch 500, nll', 'lr = 0.01, epoch 500, cross_entropy'])
display(result)


# ## try to take out different classes

# In[ ]:




