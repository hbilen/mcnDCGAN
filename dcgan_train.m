function [netG, netD, info] = dcgan_train(varargin)
% DCGAN_TRAIN demonstrates training a generative adversarial net on CelebA 

% Copyright (C) 2017 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','..', 'matlab', 'vl_setupnn.m')) ;

addpath(fullfile(vl_rootnn,'examples','imagenet'));
opts.dataDir = fullfile(vl_rootnn, 'data','celeba') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(vl_rootnn, 'exp', 'celeba-dcgan') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
% opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
% crop images into 96x96 patches
if ~exist(sprintf('%s/cropped',opts.dataDir))
  preprocess_celeba(opts.dataDir);
end
d = dir(sprintf('%s/cropped/*.jpg',opts.dataDir));
imdb.images.name = cell(numel(d),1);
imdb.imageDir = fullfile(opts.dataDir,'cropped');
for i=1:numel(d)
  imdb.images.name{i} = d(i).name;
end
imdb.images.set = ones(numel(d),1);

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
% generative model
netG = dcgan_gen_init;
% discriminator model
netD = dcgan_disc_init;

netD.meta.normalization.averageImage = [];
netD.meta.normalization.imageSize = [64 64 3];
netD.meta.normalization.cropSize = 64/96;
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

[netG, netD, info] = dcgan_train_dag(netG, netD, imdb, ... 
                      getBatchFn(opts, netD.meta), ...
                      'expDir', opts.expDir, ...
                      netD.meta.trainOpts, ...
                      opts.train) ;


% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', meta.normalization.cropSize, ...
  'subtractAverage', []) ;

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
for f = fieldnames(meta.augmentation)'
  f = char(f) ;
  bopts.train.(f) = meta.augmentation.(f) ;
end

fn = @(x,y) getBatch(bopts,useGpu,x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end

data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
    % scale down to [-1,1]
    data = (data-128)/128;
    varargout{1} = {'input',  data};
end
