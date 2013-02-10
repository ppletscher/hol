function phi = multiclass_featuremap(example, label, num_states)

% TODO: documentation

% label starts with 0!

if (numel(label) < 1)
    label = example.label;
end

num_dims = numel(example.data);

phi = zeros(num_states*num_dims,1);
idx = label*num_dims;
phi((idx+1):(idx+num_dims)) = example.data(:);
