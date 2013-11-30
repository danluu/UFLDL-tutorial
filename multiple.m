x = load('ex3x.dat');
y = load('ex3y.dat');

m = length(y);
x = [ones(m, 1), x]; %add a column of ones to x

sigma = std(x);
mu = mean(x);
x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);

theta = zeros(size(x(1,:)))';
alpha = 0.07;
delta = ones(size(x(1,:)))';
J = zeros(50, 1); 

for i = 1:50    
  h = sum(x * theta,2);
  err = h - y;
    
  J(i) = sum(err.^2) / (2*m);
  
  delta = x' * err / m;
  theta = theta - alpha * delta;
end

figure;
plot(0:49, J(1:50), '-')
xlabel('Number of iterations')
ylabel('Cost J')

% Part of the exercise here is to try different values of alpha.
% That seems both boring and trivial. I may try it later when I'm really
% bored.

% Part 2: normal equations
x = load('ex3x.dat');
x = [ones(m, 1), x];
theta = pinv(x) * y

x \ y
