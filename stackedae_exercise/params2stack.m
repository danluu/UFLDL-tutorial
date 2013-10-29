function stack = params2stack(params, netconfig)

% Converts a flattened parameter vector into a nice "stack" structure 
% for us to work with. This is useful when you're building multilayer
% networks.
%
% stack = params2stack(params, netconfig)
%
% params - flattened parameter vector
% netconfig - auxiliary variable containing 
%             the configuration of the network
%


% Map the params (a vector into a stack of weights)
depth = numel(netconfig.layersizes);
stack = cell(depth,1);
prevLayerSize = netconfig.inputsize; % the size of the previous layer
curPos = double(1);                  % mark current position in parameter vector

for d = 1:depth
    % Create layer d
    stack{d} = struct;

    % Extract weights
    wlen = double(netconfig.layersizes{d} * prevLayerSize);
    stack{d}.w = reshape(params(curPos:curPos+wlen-1), netconfig.layersizes{d}, prevLayerSize);
    curPos = curPos+wlen;

    % Extract bias
    blen = double(netconfig.layersizes{d});
    stack{d}.b = reshape(params(curPos:curPos+blen-1), netconfig.layersizes{d}, 1);
    curPos = curPos+blen;
    
    % Set previous layer size
    prevLayerSize = netconfig.layersizes{d};
end

end