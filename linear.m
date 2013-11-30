x = load('ex2x.dat');
y = load('ex2y.dat');

figure % open a new figure window
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')

m = length(y);
x = [ones(m, 1), x]; %add a column of ones to x

theta = zeros(size(x,2),1);
alpha = 0.07;

delta = ones(size(theta),1);

while abs(max(delta(:))) > 0.00001
  h = sum(x * theta,2);
  err = h - y;
  delta = x' * err / m;
  theta = theta - alpha * delta;
end

hold on % Plot new data without clearing old plot
plot(x(:,2), x*theta, '-') % remember that x is now a matrix with 2
                           % columns
                           % and the second column contains the
                           % time info
legend('Training data', 'Linear regression')

J_vals = zeros(100, 100);   % initialize Jvals to 100x100 matrix of
                            % 0's
                            

theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);
for i = 1:length(theta0_vals)
  for j = 1:length(theta1_vals)
    t = [theta0_vals(i); theta1_vals(j)];
    h = sum(x * t);
    J_vals(i,j) = sum((h - y).^2) / (2*m);
  end
end

% Plot the surface plot
% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be
% flipped
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1')
