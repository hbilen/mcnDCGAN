function net = dcgan_gen_init
% DCGAN_GEN_INIT defines generative network for DCGAN 

% Copyright (C) 2017 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

net = dagnn.DagNN();

nz = 100;
ngf = 64;
nc = 3;
% netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
% netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
count = 1;
[net,count] = generate_deconv_block(net,[4,4,nz,ngf*8],'1G',count,1);
net.layers(1).block.upsample = [1 1];
net.layers(1).block.crop = [0 0 0 0];
% -- state size: (ngf*8) x 4 x 4
% netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
% netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
[net,count] = generate_deconv_block(net,[4,4,ngf*8,ngf*4],'2G',count,1);

% -- state size: (ngf*4) x 8 x 8
% netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
% netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
[net,count] = generate_deconv_block(net,[4,4,ngf*4,ngf*2],'3G',count,1);


% -- state size: (ngf*2) x 16 x 16
% netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
% netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

[net,count] = generate_deconv_block(net,[4,4,ngf*2,ngf],'4G',count,1);

% -- state size: (ngf) x 32 x 32
% netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
% netG:add(nn.Tanh())
% -- state size: (nc) x 64 x 64
conv5 = dagnn.ConvTranspose('size',[4,4,nc,ngf],...
  'upsample', [2 2], ...
  'crop', [1 1 1 1], ...
  'hasBias', false);

net.addLayer('conv5G', ...
  conv5, sprintf('x%d',count),sprintf('x%d',count+1),...
  {'conv5G'});

count = count + 1;

s5G = dagnn.Tanh();
net.addLayer('Tanh5G',s5G,...
  sprintf('x%d',count),'inputFake');

net.vars(net.getVarIndex('inputFake')).precious = 1;

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

% Meta parameters
net.meta.inputSize = [1 1 100] ;


% -------------------------------------------------------------------------
function [net,count] = generate_deconv_block(net,sz,id,count,bn)
% -------------------------------------------------------------------------
conv1 = dagnn.ConvTranspose('size',sz([1 2 4 3]),...
  'hasBias', false, ...
  'upsample', [2 2],...
  'crop', [1 1 1 1]);

if isempty(net.layers)
%   inp = sprintf('input%s',id);
    inp = 'input' ;
else
  inp = sprintf('x%d',count);
  count = count + 1;
end
net.addLayer(sprintf('conv%s',id), ...
  conv1,inp,sprintf('x%d',count),...
  {sprintf('conv%sf',id)});

if bn
  bn1 = dagnn.BatchNorm('numChannels',sz(4));
  net.addLayer(sprintf('bn%s',id), ...
    bn1,sprintf('x%d',count),sprintf('x%d',count+1),...
    {sprintf('bn%sf',id),sprintf('bn%sx',id),sprintf('bn%sm',id)});
  count = count + 1;
end

re1 = dagnn.ReLU();
net.addLayer(sprintf('relu%s',id), ...
    re1,sprintf('x%d',count),sprintf('x%d',count+1), ...
  {});
count = count + 1;
