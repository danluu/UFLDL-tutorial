% where myfun is a MATLAB function such as:
function [f,g] = myfun(x)
f = sum(sin(x) + 3);
if ( nargout > 1 ), g = cos(x); end
