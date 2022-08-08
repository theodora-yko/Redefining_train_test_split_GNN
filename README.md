# ML

## Background and Documentation:
GraphSage - https://theodorayko.blogspot.com/2022/07/implementation-of-graphsage-using.html

## Usage
1. GNN: 
- contains the definition of neural network models in use   
- class "accuracy" defined: used for training & evaluating a ML model
2. data: downloads datasets in need
3. preliminary analysis: 
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
