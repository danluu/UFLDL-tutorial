function [params, netconfig] = stack2params(stack)

% Converts a "stack" structure into a flattened parameter vector and also
% stores the network configuration. This is useful when working with
% optimization toolboxes such as minFunc.
%
% [params, netconfig] = stack2params(stack)
%
% stack - the stack structure, where stack{1}.w = weights of first layer
%                                    stack{1}.b = weights of first layer
%                                    stack{2}.w = weights of second layer
%                                    stack{2}.b = weights of second layer
%                                    ... etc.


% Setup the compressed param vector
params = [];
for d = 1:numel(stack)
    
    % This can be optimized. But since our stacks are relatively short, it
    % is okay
    params = [params ; stack{d}.w(:) ; stack{d}.b(:) ];
    
    % Check that stack is of the correct form
    assert(size(stack{d}.w, 1) == size(stack{d}.b, 1), ...
        ['The bias should be a *column* vector of ' ...
         int2str(size(stack{d}.w, 1)) 'x1']);
    if d < numel(stack)
        assert(size(stack{d}.w, 1) == size(stack{d+1}.w, 2), ...
            ['The adjacent layers L' int2str(d) ' and L' int2str(d+1) ...
             ' should have matching sizes.']);
    end
    
end

if nargout > 1
    % Setup netconfig
    if numel(stack) == 0
        netconfig.inputsize = 0;
        netconfig.layersizes = {};
    else
        netconfig.inputsize = size(stack{1}.w, 2);
        netconfig.layersizes = {};
        for d = 1:numel(stack)
            netconfig.layersizes = [netconfig.layersizes ; size(stack{d}.w,1)];
        end
    end
end

end