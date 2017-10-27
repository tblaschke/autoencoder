Implementation of various autoencoder used in "Application of Generative Autoencoder in De Novo Molecular Design"
=================================================================================================================

This repository holds the code used to create, train, sample and search autoencoder as described in 
[Application of Generative Autoencoder in De Novo Molecular Design](https://doi.org/10.1002/minf.201700123)


Requirements
------------
This software has been tested Tesla K80 GPUs. We think it should work with other GPUs quite easily. 
The code requires an old version of PyTorch. We tested it with 0.1.12 but version 0.3.0 seem to work as well.

Before you start to use the code, please unzip the pretrained SVM classifier in `data/clf.pkl.gz` and save it as 
`data/clf.pkl`.

Bugs, Errors, Improvements, etc...
----------------------------------

We have tested the software, but if you find any bug (which there probably are some) don't hesitate to contact us.
If you have any other question, you can contact us at `blaschke@bit.uni-bonn.de` and we will be happy to answer you.