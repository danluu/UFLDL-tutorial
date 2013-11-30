x = load('ex4x.dat');
y = load('ex4y.dat');

% find returns the indices of the
% rows meeting the specified condition
pos = find(y == 1); neg = find(y == 0);

m = length(y);
x = [ones(m, 1), x]; %add a column of ones to x

% Assume the features are in the 2nd and 3rd
% columns of x
plot(x(pos, 2), x(pos,3), '+'); hold on
plot(x(neg, 2), x(neg, 3), 'o')

g = inline('1.0 ./ (1.0 + exp(-z))'); 

theta = zeros(size(x,2),1);

J = zeros(20, 1);

size(h .* 1-h)
size(h)
size(x)

for i = 1:20
  h = g(x*theta);
  err = h - y;
  grad = x' * err / m;

  % Well, this gives us the correct dimensions, so it's probably ok
  % But, there must be a better way to do this.
  hess = (repmat(1-h, 1, 3) .* x)' * (repmat(h,1,3) .* x) / m;
  
  theta = theta - hess\grad;
end

theta
