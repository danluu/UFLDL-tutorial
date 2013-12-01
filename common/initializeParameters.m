function theta = initializeParameters(hiddenSize, visibleSize)

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
W2 = rand(visibleSize, hiddenSize) * 2 * r - r;

b1 = zeros(hiddenSize, 1);
b2 = zeros(visibleSize, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

