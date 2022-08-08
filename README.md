# ML

## Background
GraphSage - https://theodorayko.blogspot.com/2022/07/implementation-of-graphsage-using.html

## Documentation
**note**: For more detailed explanations & conditions of each function/class, check the specific file.
1. **GNN**
- contains the definition of neural network models in use   
- GNN algorithms in use: GraphSAGE
- accuracy: class, used for training & evaluating a ML model
2. **data**:
: downloads datasets in need
4. **train_mask**:
: manually defines the train mask of a given dataset 
- random_train_mask: creates a random training set using PyG's inbuilt function
- create_train_mask: if default_train set False, returns data with train_mask, val_mask, test_mask attributes manually set
- mask_a_node: Given a dataset, mask a node of given index into False (i.e. mask one datapoint in the training set)
5. preliminary analysis
- find_the_influential_class: class, self.influential_accuracy returns a list of influential points
- multiple_testing: repeats finding the influential points multiple times  

## Roadmap
- [x] preliminary GNN model
- [x] manual_masks
- [x] preliminary analysis (1) - identifying the influential points 
- [ ] preliminary analysis (2) - compute & compare the gradient of each training set 
- [ ] preliminary analysis (3) - plot the distribution
- [ ] preliminary analysis (4) - spatial relationships between the points 
- [ ] implement influence function from https://arxiv.org/pdf/1703.04730.pdf
- [ ] link manual_masks with influence function and improve accuracy
