function [label, e] = predict(example, weight, include_loss, inference_algo, normalize_featuremap, loss_type, weight_hamming, weight_count, normalize_loss)

load(example);

% energy minimization for inference -> flip sign
weight = -weight;

% setup edges
num_nodes = size(example.features_unary,2);
num_edges = size(example.edges,2);

% feature dimensions
num_unary = size(example.features_unary,1);
num_features_pair = size(example.features_pairwise,1);

% unary costs
w = weight(1:num_unary);
w = w(:);
D = w'*example.features_unary;
D = [zeros(1,size(D,2)); D];

% pairwise costs (have different edge types with corresponding weights)
V = [];
edges_temp = [];
types = unique(example.edges_type);
idx_weight = num_unary;
for type_idx=1:numel(types)
    w = weight((idx_weight+1):(idx_weight+num_features_pair));
    w = w(:);
    v = w'*example.features_pairwise;
    Vtemp = [zeros(1,size(v,2)); v; v; zeros(1,size(v,2))];
    idx_weight = idx_weight + num_features_pair;

    idx = find(example.edges_type == types(type_idx));
    edges_temp = [edges_temp example.edges(:, idx)];
    V = [V Vtemp(:,idx)];
end
edges = edges_temp;

if (normalize_featuremap)
    D = D./num_nodes;
    V = V./num_edges;
end

% add the loss term
if (include_loss)
    num_pixels = numel(example.label);
    if (isequal(loss_type, 'hamming') || isequal(loss_type, 'combined'))
        idx = sub2ind(size(D), (~example.label(:)')+1, 1:num_pixels);
        sc = weight_hamming;
        if (normalize_loss)
            sc = sc/num_pixels;
        end
        D(idx) = D(idx)-sc;
    end
    if (isequal(loss_type, 'count') || isequal(loss_type, 'combined'))
        sc = weight_count;
        if (normalize_loss)
            sc = sc/num_pixels;
        end
        c = sum(example.label(:));
        D = [D sc*[-c; +c]];
        e = [repmat(num_pixels+1, [num_pixels 1]) [1:num_pixels]'];
        edges = [edges e'];
        v = repmat(sc*[0 1 0 -1], [num_pixels 1]);
        V = [V v'];
    end
end

% do the inference
if (isequal(inference_algo, 'ibfs'))
    [segm, e] = mex_gc_ibfs(D, edges, V);
elseif (isequal(inference_algo, 'bk'))
    [segm, e] = mex_gc_bk(D, edges, V);
else
    error('No solver with this name!');
end
% TODO: also implement a version that simply does twice graph-cut!

if (include_loss)
    z = 0; % higher-order variable
    if (isequal(loss_type, 'count') || isequal(loss_type, 'combined'))
        z = segm(end);
        segm = segm(1:(end-1));
    end
    
    label = segm;
    
    % check that higher-order variable has the right state
    if (isequal(loss_type, 'count') || isequal(loss_type, 'combined')) 
        assert((numel(find(label)) > numel(find(example.label(:))))==z);
    end
else
    label = segm;
end

label = reshape(label, size(example.label));
