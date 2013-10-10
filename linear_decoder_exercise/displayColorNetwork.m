function displayColorNetwork(A)

% display receptive field(s) or basis vector(s) for image patches
%
% A         the basis, with patches as column vectors

% In case the midpoint is not set at 0, we shift it dynamically
if min(A(:)) >= 0
    A = A - mean(A(:));
end

cols = round(sqrt(size(A, 2)));

channel_size = size(A,1) / 3;
dim = sqrt(channel_size);
dimp = dim+1;
rows = ceil(size(A,2)/cols);
B = A(1:channel_size,:);
C = A(channel_size+1:channel_size*2,:);
D = A(2*channel_size+1:channel_size*3,:);
B=B./(ones(size(B,1),1)*max(abs(B)));
C=C./(ones(size(C,1),1)*max(abs(C)));
D=D./(ones(size(D,1),1)*max(abs(D)));
% Initialization of the image
I = ones(dim*rows+rows-1,dim*cols+cols-1,3);

%Transfer features to this image matrix
for i=0:rows-1
  for j=0:cols-1
      
    if i*cols+j+1 > size(B, 2)
        break
    end
    
    % This sets the patch
    I(i*dimp+1:i*dimp+dim,j*dimp+1:j*dimp+dim,1) = ...
         reshape(B(:,i*cols+j+1),[dim dim]);
    I(i*dimp+1:i*dimp+dim,j*dimp+1:j*dimp+dim,2) = ...
         reshape(C(:,i*cols+j+1),[dim dim]);
    I(i*dimp+1:i*dimp+dim,j*dimp+1:j*dimp+dim,3) = ...
         reshape(D(:,i*cols+j+1),[dim dim]);

  end
end

I = I + 1;
I = I / 2;
imagesc(I); 
axis equal
axis off

end


