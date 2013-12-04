addpath ../common/
addpath ../common/fminlbfgs

%% CS294A/CS294W Convolutional Neural Networks Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  convolutional neural networks exercise. In this exercise, you will only
%  need to modify cnnConvolve.m and cnnPool.m. You will not need to modify
%  this file.

%%======================================================================
%% STEP 0: Initialization
%  Here we initialize some parameters used for the exercise.

imageDim = 64;         % image dimension
imageChannels = 3;     % number of channels (rgb, so 3)

patchDim = 8;          % patch dimension
numPatches = 50000;    % number of patches

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize = visibleSize;   % number of output units
hiddenSize = 400;           % number of hidden units 

epsilon = 0.1;	       % epsilon for ZCA whitening

poolDim = 19;          % dimension of pooling region

%%======================================================================
%% STEP 1: Train a sparse autoencoder (with a linear decoder) to learn 
%  features from color patches. If you have completed the linear decoder
%  execise, use the features that you have obtained from that exercise, 
%  loading them into optTheta. Recall that we have to keep around the 
%  parameters used in whitening (i.e., the ZCA whitening matrix and the
%  meanPatch)

% --------------------------- YOUR CODE HERE --------------------------
% Train the sparse autoencoder and fill the following variables with 
% the optimal parameters:

% optTheta =  zeros(2*hiddenSize*visibleSize+hiddenSize+visibleSize, 1);
% ZCAWhite =  zeros(visibleSize, visibleSize);
% meanPatch = zeros(visibleSize, 1);

load '../linear_decoder_exercise/STL10Features.mat';

% --------------------------------------------------------------------

% Display and check to see that the features look good
W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

displayColorNetwork( (W*ZCAWhite)');

%%======================================================================
%% STEP 2: Implement and test convolution and pooling
%  In this step, you will implement convolution and pooling, and test them
%  on a small part of the data set to ensure that you have implemented
%  these two functions correctly. In the next step, you will actually
%  convolve and pool the features with the STL10 images.

%% STEP 2a: Implement convolution
%  Implement convolution in the function cnnConvolve in cnnConvolve.m

% Note that we have to preprocess the images in the exact same way 
% we preprocessed the patches before we can obtain the feature activations.

load stlTrainSubset.mat; % loads numTrainImages, trainImages, trainLabels

%% Use only the first 8 images for testing
convImages = trainImages(:, :, :, 1:8); 

% % NOTE: Implement cnnConvolve in cnnConvolve.m first!
convolvedFeatures = cnnConvolve(patchDim, hiddenSize, convImages, W, b, ZCAWhite, meanPatch);

% %% STEP 2b: Checking your convolution
% %  To ensure that you have convolved the features correctly, we have
% %  provided some code to compare the results of your convolution with
% %  activations from the sparse autoencoder

% % For 1000 random points
% for i = 1:1000    
%     featureNum = randi([1, hiddenSize]);
%     imageNum = randi([1, 8]);
%     imageRow = randi([1, imageDim - patchDim + 1]);
%     imageCol = randi([1, imageDim - patchDim + 1]);    
   
%     patch = convImages(imageRow:imageRow + patchDim - 1, imageCol:imageCol + patchDim - 1, :, imageNum);
%     patch = patch(:);            
%     patch = patch - meanPatch;
%     patch = ZCAWhite * patch;
    
%     features = feedForwardAutoencoder(optTheta, hiddenSize, visibleSize, patch); 

%     if abs(features(featureNum, 1) - convolvedFeatures(featureNum, imageNum, imageRow, imageCol)) > 1e-9
%         fprintf('Convolved feature does not match activation from autoencoder\n');
%         fprintf('Feature Number    : %d\n', featureNum);
%         fprintf('Image Number      : %d\n', imageNum);
%         fprintf('Image Row         : %d\n', imageRow);
%         fprintf('Image Column      : %d\n', imageCol);
%         fprintf('Convolved feature : %0.5f\n', convolvedFeatures(featureNum, imageNum, imageRow, imageCol));
%         fprintf('Sparse AE feature : %0.5f\n', features(featureNum, 1));       
%         error('Convolved feature does not match activation from autoencoder');
%     end 
% end

% disp('Congratulations! Your convolution code passed the test.');

%% STEP 2c: Implement pooling
%  Implement pooling in the function cnnPool in cnnPool.m

% NOTE: Implement cnnPool in cnnPool.m first!
pooledFeatures = cnnPool(poolDim, convolvedFeatures);

% %% STEP 2d: Checking your pooling
% %  To ensure that you have implemented pooling, we will use your pooling
% %  function to pool over a test matrix and check the results.

% testMatrix = reshape(1:64, 8, 8);
% expectedMatrix = [mean(mean(testMatrix(1:4, 1:4))) mean(mean(testMatrix(1:4, 5:8))); ...
%                   mean(mean(testMatrix(5:8, 1:4))) mean(mean(testMatrix(5:8, 5:8))); ];
            
% testMatrix = reshape(testMatrix, 1, 1, 8, 8);
        
% pooledFeatures = squeeze(cnnPool(4, testMatrix));

% if ~isequal(pooledFeatures, expectedMatrix)
%     disp('Pooling incorrect');
%     disp('Expected');
%     disp(expectedMatrix);
%     disp('Got');
%     disp(pooledFeatures);
% else
%     disp('Congratulations! Your pooling code passed the test.');
% end

%%======================================================================
%% STEP 3: Convolve and pool with the dataset
%  In this step, you will convolve each of the features you learned with
%  the full large images to obtain the convolved features. You will then
%  pool the convolved features to obtain the pooled features for
%  classification.
%
%  Because the convolved features matrix is very large, we will do the
%  convolution and pooling 50 features at a time to avoid running out of
%  memory. Reduce this number if necessary

stepSize = 50;
assert(mod(hiddenSize, stepSize) == 0, 'stepSize should divide hiddenSize');

load stlTrainSubset.mat % loads numTrainImages, trainImages, trainLabels
load stlTestSubset.mat  % loads numTestImages,  testImages,  testLabels

pooledFeaturesTrain = zeros(hiddenSize, numTrainImages, ...
    floor((imageDim - patchDim + 1) / poolDim), ...
    floor((imageDim - patchDim + 1) / poolDim) );
pooledFeaturesTest = zeros(hiddenSize, numTestImages, ...
    floor((imageDim - patchDim + 1) / poolDim), ...
    floor((imageDim - patchDim + 1) / poolDim) );

tic();

for convPart = 1:(hiddenSize / stepSize)
    
    featureStart = (convPart - 1) * stepSize + 1;
    featureEnd = convPart * stepSize;
    
    fprintf('Step %d: features %d to %d\n', convPart, featureStart, featureEnd);  
    Wt = W(featureStart:featureEnd, :);
    bt = b(featureStart:featureEnd);    
    
    fprintf('Convolving and pooling train images\n');
    convolvedFeaturesThis = cnnConvolve(patchDim, stepSize, ...
        trainImages, Wt, bt, ZCAWhite, meanPatch);
    pooledFeaturesThis = cnnPool(poolDim, convolvedFeaturesThis);
    pooledFeaturesTrain(featureStart:featureEnd, :, :, :) = pooledFeaturesThis;   
    toc();
    clear convolvedFeaturesThis pooledFeaturesThis;
    
    fprintf('Convolving and pooling test images\n');
    convolvedFeaturesThis = cnnConvolve(patchDim, stepSize, ...
        testImages, Wt, bt, ZCAWhite, meanPatch);
    pooledFeaturesThis = cnnPool(poolDim, convolvedFeaturesThis);
    pooledFeaturesTest(featureStart:featureEnd, :, :, :) = pooledFeaturesThis;   
    toc();

    clear convolvedFeaturesThis pooledFeaturesThis;

end


% You might want to save the pooled features since convolution and pooling takes a long time
% save('cnnPooledFeatures.mat', 'pooledFeaturesTrain', 'pooledFeaturesTest');
% load('cnnPooledFeatures.mat');
toc();

%%======================================================================
%% STEP 4: Use pooled features for classification
%  Now, you will use your pooled features to train a softmax classifier,
%  using softmaxTrain from the softmax exercise.
%  Training the softmax classifer for 1000 iterations should take less than
%  10 minutes.

% Add the path to your softmax solution, if necessary
% addpath /path/to/solution/

% Setup parameters for softmax
softmaxLambda = 1e-4;
numClasses = 4;
% Reshape the pooledFeatures to form an input vector for softmax
softmaxX = permute(pooledFeaturesTrain, [1 3 4 2]);
softmaxX = reshape(softmaxX, numel(pooledFeaturesTrain) / numTrainImages,...
    numTrainImages);
softmaxY = trainLabels;

options = struct;
options.MaxIter = 200;
softmaxModel = softmaxTrain(numel(pooledFeaturesTrain) / numTrainImages,...
    numClasses, softmaxLambda, softmaxX, softmaxY, options);

%%======================================================================
%% STEP 5: Test classifer
%  Now you will test your trained classifer against the test images

softmaxX = permute(pooledFeaturesTest, [1 3 4 2]);
softmaxX = reshape(softmaxX, numel(pooledFeaturesTest) / numTestImages, numTestImages);
softmaxY = testLabels;

[pred] = softmaxPredict(softmaxModel, softmaxX);
acc = (pred(:) == softmaxY(:));
acc = sum(acc) / size(acc, 1);
fprintf('Accuracy: %2.3f%%\n', acc * 100);

% You should expect to get an accuracy of around 80% on the test images.
