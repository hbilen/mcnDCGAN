function out = vl_nntanh(x,dzdy)
% VL_NNTANH CNN tanh unit.
%   Y = VL_NNTANH(X) computes the tanh of the data X. X can
%   have an arbitrary size. The sigmoid is defined as follows:
%   SIGMOID(X) = (EXP(2X)-1) / (EXP(2X)+1).
%
%   DZDX = VL_NNTANH(X, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same
%   dimensions as X and Y respectively.

if nargin <= 1 || isempty(dzdy)
  out = 2 * vl_nnsigmoid(2 * x) - 1 ;
else
  out = 4 * vl_nnsigmoid(2 * x, dzdy) ;
end

