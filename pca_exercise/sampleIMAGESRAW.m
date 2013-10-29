function patches = sampleIMAGESRAW

% sampleIMAGESRAW
% Returns 10000 "raw" unwhitened  patches


load IMAGES_RAW;
IMAGES = IMAGESr;

patchSize = 12;
numPatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchSize*patchSize, numPatches);

p = 0;
for im = 1:size(IMAGES, 3)
    
    % Sample Patches
    numsamples = numPatches / size(IMAGES, 3);
    for s = 1:numsamples
        y = randi(size(IMAGES,1)-patchSize+1);
        x = randi(size(IMAGES,2)-patchSize+1);
        sample = IMAGES(y:y+patchSize-1, x:x+patchSize-1,im);
        p = p + 1;
        patches(:, p) = sample(:); 
    end
   
end

end
