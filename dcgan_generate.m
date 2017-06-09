function dcgan_generate(varargin)
% DCGAN_GENERATE: given a trained generative model (opts.network) generates
% new images by using random noise as input

% Copyright (C) 2017 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','..', 'matlab', 'vl_setupnn.m')) ;

addpath(fullfile(vl_rootnn,'examples'));
opts.gpu = [] ;
opts.network = [] ;
opts.expDir = fullfile('exp','dcgan') ;
[opts, varargin] = vl_argparse(opts, varargin) ;


opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

netG = opts.network ;
if ~isempty(opts.gpu)
  gpuDevice(opts.gpu)
  netG.move('gpu') ;
end
netG.mode = 'test' ;
netG.vars(netG.getVarIndex('inputFake')).precious = 1;
% -------------------------------------------------------------------------
%                                                                  Generate 
% -------------------------------------------------------------------------

N = 64;
for t=1:10
  if ~isempty(opts.gpu)
    noise = gpuArray.randn([1,1,100,N],'single');
  else
    noise = randn([1,1,100,N],'single');
  end
  
  netG.eval({'input',noise});
  inputFake = netG.getVar('inputFake');
  inputFake = gather(inputFake.value);
  
  sz = size(inputFake);  
  imo = zeros(8*sz(1),8*sz(2),3,'uint8');
  for i=1:8
    for j=1:8
      imo((i-1)*sz(1)+1:i*sz(1),(j-1)*sz(2)+1:j*sz(2),:) = ...
        imsingle2uint8(inputFake(:,:,:,(i-1)*8+j));
    end
  end
%   imwrite(imo,sprintf('pose-freeze/%02d.png',t));
  imshow(imo);
  pause();
end
% -------------------------------------------------------------------------
function imo = imsingle2uint8(im)
% -------------------------------------------------------------------------
mini = min(im(:));
im = im - mini;
maxi = max(im(:));
if maxi<=0
    maxi = 1;
end
imo = uint8(255 * im ./ maxi);
