# DCGAN demo

This folder contains an example implementation of DCGAN [1] in
MatConvNet. The example trains on the CELEB-A data [2].

## Data

First download and extract the aligned face images (`img_aligin_celeba.zip`) to `data/celeba` by using the link
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


There are two entry-point scripts:

* `dcgan_train.m`: trains a new model from scratch.
* `dcgan_generate.m`: generates images by using the trained model.

## Train 
To use the training code using a gpu on your system, use
something like:
    
    opts.train.gpus = 1 ;
    dcgan_train(opts) ;
    
## Generate 

    % load trained generative network from the last epoch
    d = dir(fullfile(opts.expDir,'net-epoch-*.mat'));
    load(fullfile(opts.expDir,d(end).name));
    netG = dagnn.DagNN.loadobj(netG);
    opts.network = netG ; 
    opts.gpu = 1 ;
    dcgan_generate(opts) ;

## References

1. *Unsupervised Representation Learning with Deep Convolutional Generative 
    Adversarial Networks*, Alec Radford, Luke Metz, Soumith Chintala, 2016.

2. *Deep learning face attributes in the wild.*, Liu, Z., Luo, P., Wang, X., & Tang, X. Proceedings of the IEEE International Conference on Computer Vision. 2015.
