function grabcutMain(run_idx)

run_idx = str2num(run_idx);

% library to do the classification
if (~isdeployed())
    addpath('helpers');
    addpath('matluster');
    addpath('svmstruct');
    addpath('svmstruct/helpers');
    addpath('mosek');
    addpath('mpe_inference/ibfs');
    addpath('mpe_inference/bk');
end

% load the options and dataset
load(sprintf('local/options_%d.mat', run_idx));
load(sprintf('local/%s/filenames.mat', options.featureset));

% prediction function
normalize_featuremap = 0;
normalize_loss = 0;
loss_type = options.loss;
inference_algo = 'ibfs';
if (isequal(options.loss, 'combined'))
    weight_hamming = 0.5;
    weight_count = 0.5;
else
    weight_hamming = 1;
    weight_count = 1;
end
p = @(a,b)predict(a, b, 1, inference_algo, normalize_featuremap, loss_type, weight_hamming, weight_count, normalize_loss);

% feature map
f = @(a,b)featuremap(a, b, normalize_featuremap);

% loss
l = @(a,b)loss(a, b, loss_type, normalize_loss, weight_hamming, weight_count);

% data split
rand('seed', options.seed);
idx = randperm(numel(example_filenames));
example_filenames_train = example_filenames(idx(1:60));
example_filenames_val = example_filenames(idx(61:80));
example_filenames_test = example_filenames(idx(81:end));

% options based on arguments to function
opts = [];
opts.lambda = options.lambda;
opts.max_iter = 50;
opts.logging = 0;
opts.runtime_limit = 24000; % ~7hrs
% submodularity constraint
load(example_filenames_train{1});
num_unary = size(example.features_unary,1);
types = unique(example.edges_type);
num_pair = numel(types)*size(example.features_pairwise,1);
opts.custom_constraint = @(a)addBinarySubmodularityConstraint(a, num_unary, num_pair);

% train
weight = trainMaxMarginCuttingPlanes(example_filenames_train, f, p, l, [], opts);

% prediction
[prediction_results_train, err_train] = predictDataset(example_filenames_train, weight);
[prediction_results_val, err_val] = predictDataset(example_filenames_val, weight);
[prediction_results_test, err_test] = predictDataset(example_filenames_test, weight);

result = [];
E = mean(err_train,1);
result.err_train_hamming = E(1);
result.err_train_count = E(2);
result.err_train_combined = E(3);
E = mean(err_val,1);
result.err_val_hamming = E(1);
result.err_val_count = E(2);
result.err_val_combined = E(3);
E = mean(err_test,1);
result.err_test_hamming = E(1);
result.err_test_count = E(2);
result.err_test_combined = E(3);

% save all the results
if (~exist('output', 'dir'))
    mkdir('output');
end
conf_str = matluster_generateStringFromOptions(options);
filename = sprintf('output/%s.mat', conf_str);
save(filename, 'result', 'prediction_results_*', 'example_filenames*', 'err_*');
