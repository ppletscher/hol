function [label, e] = multiclass_predict(example, weight, include_loss, num_states)

% TODO: review documentation!
% predicts the MAP label for a given example. If include_errorterm is set to
% 1, then loss-augmented inference is performed.

% problem dimensions
num_dims = numel(example.data);

% map current w to a matrix which is easier to deal with
weight = reshape(weight, [num_dims num_states]);

% build score for the different classes
score = example.data(:)'*weight;
score = score';

% loss-augmentated prediction (0/1 loss)
if (nargin > 2 && include_loss)
    score = score + 1;
    idx = example.label+1;
    score(idx) = score(idx) - 1;
end

% solve inference problem
[e, label] = max(score);
label = label-1;
e = -e;
