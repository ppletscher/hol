function s = featuremap(example, label, normalize_featuremap)

load(example);

if (numel(label) < 1)
    label = example.label;
end

y = label(:);

% assert that it is a binary label
assert(max(y) <= 1 && min(y) <= 1);
assert(max(y) >= 0 && min(y) >= 0);

% unary
s = [];
idx = find(y==1);
for i=1:size(example.features_unary,1)
    unary = example.features_unary(i,:)';
    s = [s; sum(unary(idx))];
end
num_unary = numel(s);
if (normalize_featuremap)
    s = s./numel(y);
end

% pairwise
edges_type = example.edges_type;
types = unique(edges_type);
edges = example.edges;
t = [];
for type_idx=1:numel(types)
    idx_edge = find(edges_type == types(type_idx));

    idx_diff = find(abs(y(edges(1,idx_edge)) - y(edges(2,idx_edge))) == 1);
    for i=1:size(example.features_pairwise,1)
        weight = example.features_pairwise(i,idx_edge)';
        t = [t; sum(weight(idx_diff))];
    end
end
num_pair = numel(t);
if (normalize_featuremap)
    t = t./size(example.edges,2);
end

s = [s; t];
s = full(s);

end % featuremap
