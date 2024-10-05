## Weight Generation with Graph Metanetworks

Based on ["Graph Metanetworks for Processing Diverse Neural Architectures"](https://arxiv.org/pdf/2312.04501) by Lim et al.

[Link to the original repository](https://github.com/cptq/graph_metanetworks)

This is a fun project that aims to explore an unconventional approach to model training. Given a neural network architecture with randomly initialized weights, can we transform 
these weights into something adequate without relying on gradient descent algorithms? Neural architectures can be converted into parameter graphs, where each parameter is assigned
to an edge, as demonstrated by Lim et al. Graph Metanetworks (GMNs) operate on these parameter graphs. An example of this problem is predicting the test accuracy for an unseen model
(this is a graph-level problem, as the output is a scalar). Now, let’s assume the input space is restricted to image classifiers. An example of an edge-level problem is transforming
the randomly initialized edge weights of an unseen classifier architecture.

### TODO
- [ ] Write Resnet image classification training pipeline on CIFAR-10
- [ ] Prepare a dataset of pretrained Resnets and store them as PyG graphs
- [ ] Split the dataset into the train and test sets such that there is no overlap between distinct architectures in the sets
- [ ] Modify GMN such that it takes time as input
- [ ] Impelement flow matching training pipeline
