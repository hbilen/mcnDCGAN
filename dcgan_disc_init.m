function net = dcgan_disc_init()
% DCGAN_DISC_INIT defines discriminative network for DCGAN 

% Copyright (C) 2017 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


net = dagnn.DagNN();

ndf = 64;

count = 1;
% -- input is (nc) x 64 x 64
% netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
% netD:add(nn.LeakyReLU(0.2, true))
[net,count] = generate_conv_block(net,[4 4 3 ndf],'1D','1D',count,1,0.2);
net.layers(net.getLayerIndex('conv1D')).inputs{1} = 'input';

% -- state size: (ndf) x 32 x 32
% netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
% netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
[net,count] = generate_conv_block(net,[4 4 ndf 2*ndf],'2D','2D',count,1,0.2);

% -- state size: (ndf*2) x 16 x 16
% netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
% netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
[net,count] = generate_conv_block(net,[4 4 2*ndf 4*ndf],'3D','3D',count,1,0.2);

% -- state size: (ndf*4) x 8 x 8
% netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
% netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
[net,count] = generate_conv_block(net,[4 4 4*ndf 8*ndf],'4D','4D',count,1,0.2);

% -- state size: (ndf*8) x 4 x 4
% netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
% netD:add(nn.Sigmoid())
% -- state size: 1 x 1 x 1
% netD:add(nn.View(1):setNumInputDims(3))
% -- state size: 1

conv5d = dagnn.Conv('size',[4,4,8*ndf,1],...
  'stride', 1, ...
  'hasBias', false, ...
  'pad', 0);

net.addLayer('conv5D', ...
  conv5d, sprintf('x%d',count),sprintf('x%d',count+1),...
  {'conv5D'});

count = count + 1;

s5D = dagnn.Sigmoid();
net.addLayer('sigmoid5DF',s5D,...
  sprintf('x%d',count),sprintf('x%d',count+1));
net.vars(net.getVarIndex(sprintf('x%d',count+1))).precious = 1;
net.addLayer('loss',dagnn.Loss('loss','binarylog'),{sprintf('x%d',count+1),'label'},'loss',{});
net.vars(net.getVarIndex('loss')).precious = 1;

%   net.layers(net.getLayerIndex('Tanh5G')).outputs{1}},...
  
%% init param
for i=1:numel(net.layers)
  layer = net.getLayer(i);
  pidx = layer.paramIndexes;
  
  if isa(layer.block,'dagnn.ConvTranspose') || isa(layer.block,'dagnn.Conv')
    net.params(pidx(1)).value = 0.02 * randn(layer.block.size,'single');
  elseif isa(layer.block,'dagnn.BatchNorm')
    nChn = layer.block.numChannels;
    net.params(pidx(1)).value = 1 + 0.02 * randn([nChn 1],'single');
    net.params(pidx(2)).value = zeros([nChn 1],'single');
    net.params(pidx(3)).value = zeros([nChn 2],'single');
  end
end

% for i=1:numel(net.vars)
%   net.vars(i).precious = 1;
% end
% Meta parameters
net.meta.inputSize = [96 96 3] ;
net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = 0 ;
net.meta.augmentation.jitterAspect = 0 ;


lr = logspace(-5, -6, 20) ;
bs = 64;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = bs ;
net.meta.trainOpts.weightDecay = 0.0005 ;

% -------------------------------------------------------------------------
function [net,count] = generate_conv_block(net,sz,lid,pid,count,bn,leaky)
% -------------------------------------------------------------------------
conv1 = dagnn.Conv('size',sz,...
  'hasBias', false, ...
  'stride', [2 2],...
  'pad', 1);

if isempty(net.layers)
%   inp = sprintf('input%s',lid);
    inp = 'input';
else
  inp = sprintf('x%d',count);
  count = count + 1;
end
net.addLayer(sprintf('conv%s',lid), ...
  conv1,inp,sprintf('x%d',count),...
  {sprintf('conv%sf',pid)});

if bn
  bn1 = dagnn.BatchNorm('numChannels',sz(4));
  net.addLayer(sprintf('bn%s',lid), ...
    bn1,sprintf('x%d',count),sprintf('x%d',count+1),...
    {sprintf('bn%sf',pid),sprintf('bn%sx',pid),sprintf('bn%sm',pid)});
  count = count + 1;
end

re1 = dagnn.ReLU('leak',leaky);
net.addLayer(sprintf('relu%s',lid), ...
    re1,sprintf('x%d',count),sprintf('x%d',count+1), ...
  {});
count = count + 1;
