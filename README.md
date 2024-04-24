## Introduction 

Ranging from visualising social network service connections amongst people to modelling polypharmacy
side effects based on protein interactions, graphs are a useful form of data representation in various
different academic fields. Recently, there has been a growing interest in adapting a neural network model
on graph format data, or Graph Neural Networks (GNNs), and, moreover, interpretation of predictions
by GNNs. This paper focuses on machines that perform node classification. It describes an attempt to
debias the output data through node masking input data and demonstrates that correcting a training data
to reflect the group fairness may lead to higher accuracy. This paper proposes the challenge GNN poses
on Explainable Artificial Intelligence (XAI) and definitions used throughout the paper. Then, the paper
explains the framework of our approach to the problem and provides pseudocode of the actual codes used
in the experiment. The codes were implemented using mainly Pytorch Geometric (PyG) and NetworkX
libraries. Also, note that in our approach, although we used Planetoid dataset provided by PyG library
and GraphSAGE model, any dataset and any GNN model can be analysed using our proposed framework.

# Redefining Train - Test dataset split 

## Background & Documentation
GraphSage - https://theodorayko.blogspot.com/2022/07/implementation-of-graphsage-using.html

## Problem Statement & Definitions
The inspiration for this paper rose from understanding the definition of a blackbox model. As useful
as artificial intelligence (AI) and machine learning (ML) become in terms of their efficiency in assessing
and producing predictions based on dense, big data, their accuracy is often met with scepticism due to
their rather unclear data-processing method, the reason ML models are considered a “black-box”. GNN
in particular, from producing the first set of hyperparameters through the first layer through producing
those of the final layer, has a particularly complex data transformation process due to the larger size of an
adjacency matrix. They are considered black-box as any GNN model’s behaviour cannot be comprehended
easily, regardless of having access to its structure and weights. This proposes challenges to expanding
explainability of AI over graph data, albeit their applicability.

TLDR Goals: From the current PyTorch Geometric library's train_test_split function, we noticed the split is randomly done by simple list slicing. We want to therefore answer the following questions regarding definition of train-test data split and node classification accuracy of GraphSAGE model: 

- What are effective features to construct training sets for node classification using GraphSAGE Neural Network?

## The Proposed Framework
However, although we do not have access to the transformation process within the black box, we do have
access to the input and the output dataset, leading to our proposed framework. By manually masking a
node in the input training data at a time, we are able to take a look at the influence of a single node in the
overall output and, furthermore, access the fairness of our training data set.

1. Node Masking
More formally, we will define our approach as such:
Consider a node classification problem where we map some input space $X$ with $n$ many training points
from $z_1, · · · , z_n$ to an output space $Y$. Given $n$ training points, our blackbox GNN model will produce n
outputs that labels each training points with one of $k$ many labels.

Our initial goal is to access the influence of a node to a model’s prediction. We define a test as masking
all nodes in the training set, one node at a time, and retraining the model and comparing the output
accuracy. We record the index of a node that led to an increase in output accuracy. We define such node
as an influential node. Then, we will perform the test multiple times to get a final list of influential nodes
and perform various tests based on it.

Once we have collected a list of influential nodes, we ran several different tests to identify common features
of the influential nodes such as node similarities and node centralities. The most distinctive feature they
had in common was their node centrality indices in terms of their degrees. The following images are the
distribution of degree centrality and betweeness centrality of the training set of the Cora dataset.

3. Redefining Training Sets
Once we have identified influential nodes whose masking leads to an accuracy improvement, we redefine
training sets based on influential nodes and other metrics that we have found by analysing the influential
nodes. We have mainly trained the model based on four following sets. Note that Cora dataset has 7
labels and 140 nodes in total for the training set.

We redefined training sets using following features:

- node centrality
- influential nodes
- node attributes

## Experiment Results & Evaluation
Our experiment gave a range of accuracy improvements at varying degrees. In terms of influential nodes,
exclusion of individual or all influential nodes did not lead to significant improvement of the accuracy.
However, in terms of node centralities, nodes selected based on degree centralities led to better accuracy
than betweenness centrality and, moreover, led to highest accuracy improvement when the training set
also reflected the label distribution ratio of the entire input data. We correlate such to how there is a more
extreme skewness of the data in terms of degree centralities, leading to our focus on redefining a new
training set in terms of node degree centralities.

## Repo Summary
**note**: For more detailed explanations & conditions of each function/class, check the specific file.
1. **GNN**
- contains the definition of neural network models in use   
- GNN algorithms in use: GraphSAGE
- accuracy: class, used for training & evaluating a ML model
2. **data**:
downloads datasets in need
3. **train_mask**
manually defines the train mask of a given dataset 
- random_train_mask: creates a random training set using PyG's inbuilt function
- create_train_mask: if default_train set False, returns data with train_mask, val_mask, test_mask attributes manually set
- mask_a_node: Given a dataset, mask a node of given index into False (i.e. mask one datapoint in the training set)
4. **preliminary analysis**
- return_accuracy: returns predictions, gradients if comput_gradient set True, accuracy of the model trained using the given data 
- find_the_influential_class: class, self.influential_accuracy returns a list of influential points and gradients
- multiple_testing: repeats finding the influential points multiple times  

<hr>

# Notes from Before - while working

## Roadmap
- [x] preliminary GNN model
- [x] implement leave-one-out manual_masks
- [x] analysis (1) - identifying the influential points using leave-one-out manual_masks
- [x] analysis (2) - compute & compare the gradient of each training set
- [x] analysis (3) - use different node features to redefine train - test data set and make classifications
- [ ] preliminary analysis (3) - plot the distribution
- [ ] preliminary analysis (4) - spatial relationships between the points 
- [ ] implement influence function from https://arxiv.org/pdf/1703.04730.pdf
- [ ] link manual_masks with influence function and improve accuracy

## Challenges ToT
- reaching the accuracy level of the example I was imitating smh took SO long
- switching between different types of data (ESPECIALLY torch tensors) 
- understanding the structure of a PyG dataset, was very confused initially
- each training takes time to compute, and with a bigger epoch size, it was often the case that I had to wait for several minutes to figure out there is an error in my code. 
