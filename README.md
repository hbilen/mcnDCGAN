# DCGAN demo

This folder contains an example implementation of DCGAN [1] in
MatConvNet. The example trains on the CELEB-A data.

There are two entry-point scripts:

* `dcgan_train.m`: trains a new model from scratch.
* `dcgan_generate.m`: generates images by using the trained model.

## Data

First download the dataset from
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Train 
To use the training code using a gpu on your system, use
something like:

    opts.train.gpus = 1 ;
    dcgan_train(opts) ;
    
## Generate 

    % load trained network
    d = dir(fullfile(opts.expDir,'net-epoch-*.mat'));
    load(fullfile(opts.expDir,d(end).name));
    netG = dagnn.DagNN.loadobj(netG);
    opts.network = netG ; 
    opts.gpu = 1 ;
    dcgan_generate(opts) ;

## References

1. *Unsupervised Representation Learning with Deep Convolutional Generative 
    Adversarial Networks*, Alec Radford, Luke Metz, Soumith Chintala, 2016.
