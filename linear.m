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
    delta
    theta = theta - alpha * delta;
end

hold on % Plot new data without clearing old plot
plot(x(:,2), x*theta, '-') % remember that x is now a matrix with 2
                           % columns
                           % and the second column contains the
                           % time info
legend('Training data', 'Linear regression')


