function preprocess_celeba(dataDir)
% PREPROCESS_CELEBA prepares celeba data for training

% Copyright (C) 2017 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


d = dir(sprintf('%s/img_align_celeba/*.jpg',dataDir));
if isempty(d), error('No images found in %s',dataDir); end
inDir = sprintf('%s/img_align_celeba/',dataDir);
outDir = sprintf('%s/cropped/',dataDir);

mkdir(outDir);

for i=1:numel(d)
  if mod(i,100)==1
    fprintf('%d // %d\n',i,numel(d));
  end
  im = imread(fullfile(inDir,d(i).name));
%   figure(1),imshow(im);
  
  [h,w,~] = size(im);
  
  imc = im(36:h-35,16:163,:);
  imr = imresize(imc,[96 96]);
  imwrite(imr,fullfile(outDir,d(i).name));
%   figure(2),imshow(imc);
%   pause();
  
end
