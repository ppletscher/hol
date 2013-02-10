clear all;
rand('seed', 0);
randn('seed', 0);

addpath('../../');
addpath('mosek');

% load the mnist dataset
load('mnist_all.mat');

% convert the dataset
XTrain = [];
YTrain = [];
XTest = [];
YTest = [];
for k=0:9
    % train
    eval(sprintf('A = train%d;', k));
    A = double(A);
    XTrain = [XTrain; im2double(A)];
    YTrain = [YTrain; repmat(k, [size(A,1) 1])];

    % test
    eval(sprintf('A = test%d;', k));
    XTest = [XTest; im2double(A)];
    YTest = [YTest; repmat(k, [size(A,1) 1])];
end
clear test* train*;

% adding a generalized component
XTrain = [XTrain repmat(1, [size(XTrain,1) 1])];
XTest = [XTest repmat(1, [size(XTest,1) 1])];

% random subset of the data

idx_train = randperm(numel(YTrain));
idx_train = idx_train(1:100);
idx_test = randperm(numel(YTest));
idx_test = idx_test(1:100);
XTrain = XTrain(idx_train,:);
YTrain = YTrain(idx_train);
XTest = XTest(idx_test,:);
YTest = YTest(idx_test);

% build the train and test datasets
examples_train = {};
for i=1:size(XTrain,1)
    examples_train{i} = [];
    examples_train{i}.data = XTrain(i,:);
    examples_train{i}.label = YTrain(i);
end
examples_test = {};
for i=1:size(XTest,1)
    examples_test{i} = [];
    examples_test{i}.data = XTest(i,:);
    examples_test{i}.label = YTest(i);
end
clear XTest XTrain YTest YTrain;


num_states = 10;
f = @(a,b) multiclass_featuremap(a, b, num_states);
p = @(a,b) multiclass_predict(a, b, 1, num_states);

options = [];
options.lambda = 1e2;
options.max_iter = 10;
%options.pruning_enable = 1;
%options.pruning_slack_tol = inf;
%options.pruning_max_constraints_factor = 2;
weight = trainMaxMarginCuttingPlanes(examples_train, f, p, @multiclass_loss, [], options);

% loss on train set
avg_loss = 0;
for i=1:numel(examples_train)
    label = multiclass_predict(examples_train{i}, weight, 0, num_states);
    avg_loss = avg_loss + multiclass_loss(examples_train{i}, label);
end
avg_loss = avg_loss / numel(examples_train);
fprintf('average loss on the training set: %f.\n', avg_loss);

% loss on test set
avg_loss = 0;
for i=1:numel(examples_test)
    label = multiclass_predict(examples_test{i}, weight, 0, num_states);
    avg_loss = avg_loss + multiclass_loss(examples_test{i}, label);
end
avg_loss = avg_loss / numel(examples_test);
fprintf('average loss on the test set: %f.\n', avg_loss);
