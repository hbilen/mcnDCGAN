function [netG,netD,stats] = dcgan_train_dagnn(netG, netD, imdb, getBatch, varargin)
% DCGAN_TRAIN_DAGNN demonstrates training a convolutional GAN using the DagNN wrapper

% Copyright (C) 2017 Hakan Bilen and 2014-17 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 10 ;
opts.learningRate = 0.002 ;
opts.weightDecay = 0.0005 ;
opts.labelSmoothing = false ;
opts.noiseType = 'normal' ;

opts.solver = @solver.adam ;  % Empty array means use the default SGD solver
opts.solverOpts.beta1 = 0.5 ;

[opts, varargin] = vl_argparse(opts, varargin) ;
if ~isempty(opts.solver)
    assert(isa(opts.solver, 'function_handle') && nargout(opts.solver) == 2,...
        'Invalid solver; expected a function handle with two outputs.') ;
    % Call without input arguments, to get default options
    opts.solverOpts = opts.solver() ;
end

opts.momentum = 0.9 ;
opts.saveSolverState = true ;
opts.nesterovUpdate = false ;
opts.randomSeed = 0 ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.plotStatistics = true;
opts.postEpochFn = [] ;  % postEpochFn(net,params,state) called after each epoch; can return a new learning rate, 0 to stop, [] for no change
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
    if isempty(opts.derOutputs)
        error('DEROUTPUTS must be specified when training.\n') ;
    end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
    [netG, netD, state, stats] = loadState(modelPath(start)) ;
else
    state = [] ;
end

for epoch=start+1:opts.numEpochs
    
    % Set the random seed based on the epoch and opts.randomSeed.
    % This is important for reproducibility, including when training
    % is restarted from a checkpoint.
    
    rng(epoch + opts.randomSeed) ;
    prepareGPUs(opts, epoch == start+1) ;
    
    % Train for one epoch.
    params = opts ;
    params.epoch = epoch ;
    params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
    params.val = opts.val(randperm(numel(opts.val))) ;
    params.imdb = imdb ;
    params.getBatch = getBatch ;
    
    if numel(opts.gpus) <= 1
        [netG, netD, state] = processEpoch(netG, netD, state, params, 'train') ;
        %     [netG, netD, state] = processEpoch(netG, netD, state, params, 'val') ;
        if ~evaluateMode
            saveState(modelPath(epoch), netG, netD, state) ;
        end
        lastStats = state.stats ;
    else
        error('Multi-gpu support is not implemented!\n');
    end
    
    stats.train(epoch) = lastStats.train ;
    stats.val(epoch) = struct() ;
    clear lastStats ;
    saveStats(modelPath(epoch), stats) ;
    
    if opts.plotStatistics
        switchFigure(1) ; clf ;
        plots = setdiff(...
            cat(2,...
            fieldnames(stats.train)', ...
            fieldnames(stats.val)'), {'num', 'time'}) ;
        for p = plots
            p = char(p) ;
            values = zeros(0, epoch) ;
            leg = {} ;
            for f = {'train', 'val'}
                f = char(f) ;
                if isfield(stats.(f), p)
                    tmp = [stats.(f).(p)] ;
                    values(end+1,:) = tmp(1,:)' ;
                    leg{end+1} = f ;
                end
            end
            subplot(1,numel(plots),find(strcmp(p,plots))) ;
            plot(1:epoch, values','o-') ;
            xlabel('epoch') ;
            title(p) ;
            legend(leg{:}) ;
            grid on ;
        end
        drawnow ;
        print(1, modelFigPath, '-dpdf') ;
    end
    
    if ~isempty(opts.postEpochFn)
        if nargout(opts.postEpochFn) == 0
            opts.postEpochFn(net, params, state) ;
        else
            lr = opts.postEpochFn(net, params, state) ;
            if ~isempty(lr), opts.learningRate = lr; end
            if opts.learningRate == 0, break; end
        end
    end
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end

% -------------------------------------------------------------------------
function [netG, netD, state] = processEpoch(netG, netD, state, params, mode)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% initialize with momentum 0
if isempty(state) || isempty(state.solverStateG) || isempty(state.solverStateD)
    state.solverStateG = cell(1, numel(netG.params)) ;
    state.solverStateG(:) = {0} ;
    state.solverStateD = cell(1, numel(netD.params)) ;
    state.solverStateD(:) = {0} ;
end


% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
    netG.move('gpu') ;
    netD.move('gpu') ;
    
    for i = 1:numel(state.solverStateG)
        s = state.solverStateG{i} ;
        if isnumeric(s)
            state.solverStateG{i} = gpuArray(s) ;
        elseif isstruct(s)
            state.solverStateG{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
        end
    end
    for i = 1:numel(state.solverStateD)
        s = state.solverStateD{i} ;
        if isnumeric(s)
            state.solverStateD{i} = gpuArray(s) ;
        elseif isstruct(s)
            state.solverStateD{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
        end
    end
end

if numGpus > 1
    error('Multi-gpu is not supported!');
    %   parserv = ParameterServer(params.parameterServer) ;
    %   net.setParameterServer(parserv) ;
else
    parserv = [] ;
end

% profile
if params.profile
    if numGpus <= 1
        profile clear ;
        profile on ;
    else
        mpiprofile reset ;
        mpiprofile on ;
    end
end

num = 0 ;
epoch = params.epoch ;
subset = params.(mode) ;
adjustTime = 0 ;

stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;

start = tic ;
figure(2);
n = 0;
stats.errorG = 0;
stats.errorD = 0;

for t=1:params.batchSize:numel(subset)
    fprintf('%s: epoch %02d: %3d/%3d:', mode, epoch, ...
        fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
    batchSize = min(params.batchSize, numel(subset) - t + 1) ;
    
    
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) ;
    batchEnd = min(t+params.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end
    
    inputs = params.getBatch(params.imdb, batch) ;
    
    if params.prefetch
        if s == params.numSubBatches
            batchStart = t + (labindex-1) + params.batchSize ;
            batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
        else
            batchStart = batchStart + numlabs ;
        end
        nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
        params.getBatch(params.imdb, nextBatch) ;
    end
    
    
    if strcmp(params.noiseType,'normal')
      noise = randn([1,1,100,numel(batch)],'single');
    elseif strcmp(params.noiseType,'uniform')
      noise = 2 * rand([1,1,100,numel(batch)],'single') - 1;
    else
      error('Unknown noise type!') ;
    end
    
    labelFake = -ones(1,numel(batch),'single');

    if params.labelSmoothing
      % between 0.7 and 1
      labelReal = 0.3 * rand(1,numel(batch),'single') + 0.7 ;
    else
      labelReal = -labelFake;
    end

    if numGpus>0
      noise = gpuArray(noise) ;
      labelFake = gpuArray(labelFake) ;
    end
    
    % generate images from random noise
    netG.mode = 'normal' ;
    netG.eval({'input',noise});
    inputFake = netG.getVar('inputFake');
    inputFake = inputFake.value;
   
    % show generated images
    if mod(ceil((t-1)/params.batchSize),20)==0
      sz = size(inputFake) ;
      im = zeros(5*sz(1),5*sz(2),3,'uint8');
      for ii=1:5
        for jj=1:5
          idx = 5*(ii-1)+jj ;
          if idx<=sz(4)
            im((ii-1)*sz(1)+1:ii*sz(1),(jj-1)*sz(2)+1:jj*sz(2),:) = ...
              gather(imsingle2uint8(inputFake(:,:,:,idx))) ;
          end
        end
      end
      imshow(im);
      drawnow;
    end
     
    % train discriminator with fake and real data
    netD.mode = 'normal' ;
    netD.accumulateParamDers = 0 ;
    netD.eval({'input',inputFake,'label',labelFake}, {'loss',1},'holdOn', 1) ;
    
    errorFake = netD.getVar('loss');
    errorFake = gather(errorFake.value);
    netD.accumulateParamDers = 1 ;
    
    netD.eval({'input',inputs{2},'label',labelReal}, {'loss',1}, 'holdOn', 0) ;
    
    errorReal = netD.getVar('loss');
    errorReal = gather(errorReal.value);

    errorD = errorFake + errorReal;
    
    % update netD
    state.solverState = state.solverStateD;
    state = accumulateGradients(netD, state, params, 2 * batchSize, parserv);
    state.solverStateD = state.solverState;
    
    netD.eval({'input',inputFake,'label',labelReal}, {'loss',1}, 'holdOn', 0) ;
    errorG = netD.getVar('loss');
    errorG = gather(errorG.value);
    
    df_dg = netD.vars(1).der;
    
    % cleanup der
    for p=1:numel(netD.params)
      netD.params(p).der = [];
    end
    
    for v=1:numel(netD.vars)
      netD.vars(v).der = [];
    end
    
    netG.mode = 'normal' ;
    netG.accumulateParamDers = 0 ;
    netG.eval({'input',noise},{'inputFake',df_dg}, 'holdOn', 0);
    
    
    % update netG
    state.solverState = state.solverStateG;
    state = accumulateGradients(netG, state, params, batchSize, parserv);
    state.solverStateG = state.solverState;
    
    
    % Get statistics.
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == 3*params.batchSize + 1
        % compensate for the first three iterations, which are outliers
        adjustTime = 4*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    
    
    % loss may get inf values
    stats.errorG = stats.errorG + min(batchSize*10,errorG);
    stats.errorD = stats.errorD + min(batchSize*10,errorD);
    
    fprintf(' errorG: %.3f errorD: %.3f', stats.errorG/(t + batchSize - 1),...
        stats.errorD/(t + batchSize - 1)/2) ;
    
%   fprintf(' errorG: %.3f errorD: %.3f', errorD/batchSize, errorG/batchSize) ;
    fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;

    fprintf('\n') ;
end
stats.errorG = stats.errorG / numel(subset) ;
stats.errorD = stats.errorD / numel(subset) ;

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
    if numGpus <= 1
        state.prof.(mode) = profile('info') ;
        profile off ;
    else
        state.prof.(mode) = mpiprofile('info');
        mpiprofile off ;
    end
end
if ~params.saveSolverState
    state.solverStateG = [] ;
    state.solverStateD = [] ;
else
    for i = 1:numel(state.solverStateG)
        s = state.solverStateG{i} ;
        if isnumeric(s)
            state.solverStateG{i} = gather(s) ;
        elseif isstruct(s)
            state.solverStateG{i} = structfun(@gather, s, 'UniformOutput', false) ;
        end
    end
    for i = 1:numel(state.solverStateD)
        s = state.solverStateD{i} ;
        if isnumeric(s)
            state.solverStateD{i} = gather(s) ;
        elseif isstruct(s)
            state.solverStateD{i} = structfun(@gather, s, 'UniformOutput', false) ;
        end
    end
end

netG.reset() ;
netG.move('cpu') ;
netD.reset() ;
netD.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for p=1:numel(net.params)
    
    if ~isempty(parserv)
        parDer = parserv.pullWithIndex(p) ;
    else
        parDer = net.params(p).der ;
    end
    
    switch net.params(p).trainMethod
        
        case 'average' % mainly for batch normalization
            thisLR = net.params(p).learningRate ;
            net.params(p).value = vl_taccum(...
                1 - thisLR, net.params(p).value, ...
                (thisLR/batchSize/net.params(p).fanout),  parDer) ;
            
        case 'gradient'
            thisDecay = params.weightDecay * net.params(p).weightDecay ;
            thisLR = params.learningRate * net.params(p).learningRate ;
            
            if thisLR>0 || thisDecay>0
                % Normalize gradient and incorporate weight decay.
                parDer = vl_taccum(1/batchSize, parDer, ...
                    thisDecay, net.params(p).value) ;
                
                if isempty(params.solver)
                    % Default solver is the optimised SGD.
                    % Update momentum.
                    state.solverState{p} = vl_taccum(...
                        params.momentum, state.solverState{p}, ...
                        -1, parDer) ;
                    
                    % Nesterov update (aka one step ahead).
                    if params.nesterovUpdate
                        delta = params.momentum * state.solverState{p} - parDer ;
                    else
                        delta = state.solverState{p} ;
                    end
                    
                    % Update parameters.
                    net.params(p).value = vl_taccum(...
                        1,  net.params(p).value, thisLR, delta) ;
                    
                else
                    % call solver function to update weights
                    [net.params(p).value, state.solverState{p}] = ...
                        params.solver(net.params(p).value, state.solverState{p}, ...
                        parDer, params.solverOpts, thisLR) ;
                end
            end
            
        otherwise
            error('Unknown training method ''%s'' for parameter ''%s''.', ...
                net.params(p).trainMethod, ...
                net.params(p).name) ;
    end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
    s = char(s) ;
    total = 0 ;
    
    % initialize stats stucture with same fields and same order as
    % stats_{1}
    stats__ = stats_{1} ;
    names = fieldnames(stats__.(s))' ;
    values = zeros(1, numel(names)) ;
    fields = cat(1, names, num2cell(values)) ;
    stats.(s) = struct(fields{:}) ;
    
    for g = 1:numel(stats_)
        stats__ = stats_{g} ;
        num__ = stats__.(s).num ;
        total = total + num__ ;
        
        for f = setdiff(fieldnames(stats__.(s))', 'num')
            f = char(f) ;
            stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;
            
            if g == numel(stats_)
                stats.(s).(f) = stats.(s).(f) / total ;
            end
        end
    end
    stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
for i = 1:numel(sel)
    stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, netG_, netD_, state)
% -------------------------------------------------------------------------
netG = netG_.saveobj() ;
netD = netD_.saveobj() ;
save(fileName, 'netG', 'netD', 'state') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
if exist(fileName)
    save(fileName, 'stats', '-append') ;
else
    save(fileName, 'stats') ;
end

% -------------------------------------------------------------------------
function [netG, netD, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'netG', 'netD', 'state', 'stats') ;
netG = dagnn.DagNN.loadobj(netG) ;
netD = dagnn.DagNN.loadobj(netD) ;
if isempty(whos('stats'))
    error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
    try
        set(0,'CurrentFigure',n) ;
    catch
        figure(n) ;
    end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
    % check parallel pool integrity as it could have timed out
    pool = gcp('nocreate') ;
    if ~isempty(pool) && pool.NumWorkers ~= numGpus
        delete(pool) ;
    end
    pool = gcp('nocreate') ;
    if isempty(pool)
        parpool('local', numGpus) ;
        cold = true ;
    end
    
end
if numGpus >= 1 && cold
    fprintf('%s: resetting GPU\n', mfilename)
    clearMex() ;
    if numGpus == 1
        gpuDevice(opts.gpus)
    else
        spmd
            clearMex() ;
            gpuDevice(opts.gpus(labindex))
        end
    end
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
