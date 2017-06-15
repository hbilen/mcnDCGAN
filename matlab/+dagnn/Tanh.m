classdef Tanh < dagnn.ElementWise
% TANH layer

% Copyright (C) 2017 Hakan Bilen.
% All rights reserved.
%

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nntanh(inputs{1});
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nntanh(inputs{1}, derOutputs{1}) ;
      derParams = {} ;
    end
  end
end
