#!/usr/bin/env python
# coding: utf-8

# In[3]:


#neural network model 
import torch 
from torch_geometric.nn import SAGEConv


# ## define your NN here

# In[4]:


#implemented a two-layer GraphSAGE from GCN example:
class GraphSAGE(torch.nn.Module):
    def __init__(self, attributes, classes, dimension=32):
        super(GraphSAGE, self).__init__() 
        self.conv1 = SAGEConv(attributes, dimension) 
        self.conv2 = SAGEConv(dimension, classes)
        self.name = "GraphSAGE"
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index 

        x = self.conv1(x, edge_index) #layer 1 
        x = F.relu(x)
        x = F.dropout(x, training=self.training) 
        x = self.conv2(x, edge_index) #layer 2 
        return x


# ## training function

# In[6]:


## training & evaluating 
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch.nn
import matplotlib.pyplot as plt

class accuracy():
    def __init__(self, model, device, dataset, data, indicate=True):
        self.model = model(attributes=dataset.num_node_features, classes=dataset.num_classes, dimension=32).to(device)
        self.data = data
        self.indicate = indicate
    
    def train(self, num_epochs, lr=0.001):
        loss_vals= []
        valid_vals= []
        optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=5e-4)
        if self.indicate: print(f'Training the model')

        ## training 
        for epoch in range(num_epochs): 
            optimizer.zero_grad() 

            out = self.model(self.data)
            loss = F.cross_entropy(input=out[self.data.train_mask], target=self.data.y[self.data.train_mask]) 
            loss.backward() 
            optimizer.step() 
            
    def evaluate(self, dimension=1, write_confusion=False, return_prediction=False):
        # returns a confusion matrix if set write_confusion=True
        if self.indicate: print('Evaluating the model')
        self.model.eval()
        pred = self.model(self.data).argmax(dimension)        
        correct = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).sum()
        acc = int(correct) / int(self.data.test_mask.sum())
        if self.indicate: print(f'Accuracy: {acc:.4f}\n')
    
        if write_confusion: 
            new = confusion_matrix(self.data.y[self.data.test_mask].numpy(), pred[self.data.test_mask].numpy(), normalize='true')
            return(pd.DataFrame(data=new)), acc
        
        if return_prediction: 
            return pred[self.data.test_mask], acc
        return acc


# In[ ]:




