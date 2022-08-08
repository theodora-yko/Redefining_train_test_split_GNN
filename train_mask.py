#!/usr/bin/env python
# coding: utf-8

# In[1]:


#defining a manual train mask function 
#train_mask = a list of booleans to mask, same length as num_nodes
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms import RemoveTrainingClasses, RandomNodeSplit  

def random_train_mask(dataset):
#     cora_A_data = transform_nodes(cora_A[0])'
    transform_nodes = RandomNodeSplit(split = 'test_rest') # RandomNodeSplit is a class 
    data = transform_nodes(dataset[0])
    return data

def split(which_nodes, num_nodes, num_test_nodes) -> Tuple[Tensor, Tensor, Tensor]:
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.ones(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    count = 0
    for i in range(num_nodes):  #separate the testing set #use randomperm to randomise this even more
        if count == num_test_nodes: break 
        if i not in which_nodes: 
            test_mask[i] = True
            count+=1

    for i in which_nodes: #separate the training and validation set
        train_mask[i] = True  
        val_mask[i] = False

    return train_mask, val_mask, test_mask

def create_train_mask(dataset, which_nodes, test_ratio, default_train=False):
    """
    which_nodes: a list of integers(node indices) that will be masked
    test_ratio: float, what percentage of the entire nodes that will be used for the testing set
    
    returns: data with train_mask, val_mask, test_mask attributes manually set
    """
    
    data = dataset[0]
    num_nodes = data.num_nodes
    num_test_nodes = test_ratio * data.num_nodes
        
    if default_train: 
        # random split using the pyg function 
        return random_train_mask(dataset) 
    
    else: 
        # manual split taking out the designated nodes
        for store in data.node_stores:
            train_masks, val_masks, test_masks = zip(*[split(which_nodes, num_nodes, num_test_nodes)])
            store.train_mask = torch.stack(train_masks, dim=-1).squeeze(-1)
            store.val_mask = torch.stack(val_masks, dim=-1).squeeze(-1)
            store.test_mask = torch.stack(test_masks, dim=-1).squeeze(-1)
        return data

def from_numpy(train, val, test):
    return torch.from_numpy(train), torch.from_numpy(val), torch.from_numpy(test)

def mask_a_node(dataset, which_node):
    """
    Given a dataset, mask a node of given index into False (i.e. mask one datapoint in the training set)
    The node masked witll be put into test_mask 
    note: the dataset must already have a preexisting train_mask tensor
    
    returns: data with train_mask, val_mask, test_mask attributes manually set
    
    which_node: an integer index of a node to be masked 
    """
    data = dataset[0]
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()
    
    train_mask[which_node] = 'False'
    val_mask[which_node] = 'False'
    test_mask[which_node] = 'True'
    
    for store in data.node_stores:
        train_masks, val_masks, test_masks = zip(*[from_numpy(train_mask, val_mask, test_mask)])
        store.train_mask = torch.stack(train_masks, dim=-1).squeeze(-1)
        store.val_mask = torch.stack(val_masks, dim=-1).squeeze(-1)
        store.test_mask= torch.stack(test_masks, dim=-1).squeeze(-1)
    return data


# In[ ]:




