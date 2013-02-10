function [weight, svm, history] = trainMaxMarginCuttingPlanes(examples, featuremap, predict, loss, weight0, options)

% TODO: help/documentation
% The objective is:
% 1/N*\sum_{n=1}^N \xi^n + 0.5*\lambda*\|w\|^2 .

% REMARK: Make sure that the loss function handle and the loss-augmented
% version of the predict function handle implement the same form of the loss.
% trainMaxMarginCuttingPlanes performes some simple checks, but it probably
% dies horribly if you have non-matching loss terms!


options_default = defaultOptions();
if (nargin >= 6)
    options = processOptions(options, options_default);
else
    options = options_default;
end
options

% dimensionality of the problem
num_examples = numel(examples);
s = featuremap(examples{1}, []);
num_dims = numel(s);

% initial parameter guess
if (nargin>4 && numel(weight0) > 0)
    if (num_dims ~= numel(weight0))
        error('dimension of input weight seems wrong!');
    end
    weight = weight0;
else
    weight = zeros(size(s));
end

% initialize the SVM
svm = setupSVM(num_examples, num_dims, options);
objective_old = 0;

printInformationHeader();
time_total = 0;
timer_start = tic;

history = [];
history.weight = weight(:);

% cutting-planes algorithm (n-slack)
for iter=1:options.max_iter
    for example_id=1:num_examples
        [g, h] = generateConstraint(featuremap, predict, loss, ...
                    example_id, weight, ...
                    examples{example_id}, num_examples, ...
                    iter, options);
        svm = addConstraintsSVM(svm, g, h);
        
        % solve the updated QP
        [weight, slacks, objective] = solveSVM(svm, options);
        svm.slacks = slacks;
    
        % timer
        timer_elapsed = toc(timer_start);
        time_total = time_total + timer_elapsed;
        timer_start = tic;

        % enforce a runtime limit
        if (time_total > options.runtime_limit)
            break;
        end
    end

    rel_difference = (objective - objective_old) / (objective_old + eps);
    if (rel_difference < 0)
        display('Numerical imprecision! Could be due to pruning.');
    end
    rel_difference = max(rel_difference,0);

    % print progress information
    printInformation(iter, objective, time_total, rel_difference);
    
    svm = pruneConstraints(svm, weight, options);

    % show the current constraints matrix and the weight
    if (options.verbose)
        clf;
        subplot(1,2,1)
        colormap jet;
        imagesc(svm.qp.G)
        subplot(1,2,2)
        plot(weight)
        drawnow
    end

    history.weight = [history.weight weight(:)];

    % convergence checks
    % TODO: improve this! (especially if we prune or approximate inference is used!)
    if (iter==options.max_iter)
        fprintf('Maximum number of iterations reached.\n');
    elseif (time_total > options.runtime_limit)
        fprintf('runtime limit reached.\n');
        break;
    elseif ( (options.convergence_check) &&  (rel_difference < options.convergence_tol) )
        fprintf('Relative convergence reached.\n');
        break;
    end
    objective_old = objective;
end

end % trainMaxMarginCuttingPlanes


%----------------------------------------------------------------------------
function [w, slacks, objective] = solveSVM(svm, options)

% for better readability
num_vars = svm.dim + numel(svm.slacks);
P = svm.qp.P;
q = svm.qp.q;
G = svm.qp.G;
h = svm.qp.h;

% solve the standard SVM QP
if (isequal(options.solver, 'cvx'))
    % TODO: did not check the cvx version in quite some time, please
    % double-check the code below before you use it!
    cvx_begin
        cvx_precision high
        cvx_quiet(true)
        variable x(num_vars)
        minimize(0.5*x'*P*x + q'*x);
        subject to
            G*x <= h;
    cvx_end
    objective = 0.5*x'*P*x + q'*x;
elseif (isequal(options.solver, 'mosek'))
    % NOTE: 0.5 is already taken care of by MOSEK QP
    param = [];
    param.MSK_IPAR_LOG = 0;
    prob = [];
    prob.c = q;
    [indI,indJ,v] = find(tril(P));
    prob.qosubi = indI;
    prob.qosubj = indJ;
    prob.qoval = v;
    prob.a = sparse(G);
    prob.blc = [];
    prob.buc = h;
    [r, res] = mosekopt('minimize echo(0)', prob, param);

    if (r > 0)
        res
    end

    x = res.sol.itr.xx;
    objective = res.sol.itr.pobjval;
end

% read out solution
slacks = x(svm.dim+1:end);
w = x(1:svm.dim);

end % function solveSVM


%----------------------------------------------------------------------------
function [g, h] = generateConstraint(featuremap, predict, loss, index, weight, example, num_examples, iter, options)

g = [];
h = [];

% featuremap of ground-truth and loss-augmented max label
phi_true = featuremap(example, []);
[label, e] = predict(example, weight);
phi_map = featuremap(example, label);

% compute the error term
delta = loss(example, label);

% energy checks: energy returned by predict should agree with the inner
% product!
if ((options.energy_check) && (abs(dot(phi_map, weight)+delta+e) > 1e-2))
    dot(phi_map, weight)+delta
    e
    error(['The energy returned by your predict function seems to be wrong!']);
end

% generate the linear constraint
t = zeros(num_examples, 1);
t(index) = -1;
s = phi_map-phi_true;
g = [g; s' t'];
h = [h; -delta];

end % generateConstraint


%----------------------------------------------------------------------------
function svm = setupSVM(num_examples, num_dims, options)

svm = [];
svm.dim = num_dims;
svm.slacks = zeros(num_examples, 1);
svm.qp = [];

% constraint on positive slack variables
svm.qp.G = [zeros(num_examples, num_dims) -diag(ones(num_examples,1))];
svm.qp.G = sparse(svm.qp.G);
svm.qp.h = zeros(num_examples,1);

% objective: quadratic norm of weights and summing up the structured loss
svm.qp.P = [ ...
            diag(ones(num_dims,1)), zeros(num_dims, num_examples); ...
            zeros(num_examples, num_dims), zeros(num_examples, num_examples)...
];
svm.qp.P = sparse(options.lambda*svm.qp.P);
svm.qp.q = [zeros(num_dims,1); 1/(num_examples)*ones(num_examples,1)];

svm = options.custom_constraint(svm);

svm.num_fixed_constraints = numel(svm.qp.h);

end % setupSVM


%----------------------------------------------------------------------------
function svm = addConstraintsSVM(svm, G, h)

svm.qp.G = [svm.qp.G; G];
svm.qp.h = [svm.qp.h; h];

end % addConstraintsSVM


%----------------------------------------------------------------------------
function options = defaultOptions()

options = [];
options.lambda = 1;
options.max_iter = 200;
options.runtime_limit = inf;
options.convergence_check = 1;
options.convergence_tol = 1e-5;
options.solver = 'mosek';
options.custom_constraint = @emptyCustomConstraint;
options.verbose = 0;
options.logging = 0;
options.log_dir = 'log';
options.pruning_enable = 0;
options.pruning_slack_tol = 1e-4; % TODO: misleading name, the bigger, the more constraints are kept!
options.pruning_max_constraints_factor = 5;
options.energy_check = 1;

end % defaultOptions


%----------------------------------------------------------------------------
function printInformationHeader()

fprintf('%5s | %8s | %20s | %10s\n', 'iter', 'time', 'objective', 'conv');

end % printInformationHeader


%----------------------------------------------------------------------------
function printInformation(iter, objective, time, rel_difference)

fprintf('%5d   %8.1f   %20.6g   %10.3e\n', iter, time, objective, rel_difference);

end % printInformation


%----------------------------------------------------------------------------
function svm = emptyCustomConstraint(svm)

end % emptyCustomConstraint


%----------------------------------------------------------------------------
function svm = pruneConstraints(svm, weight, options)

if (options.pruning_enable)
    svm = pruneConstraintsTol(svm, weight, options);
    svm = pruneConstraintsMax(svm, weight, options);
end

end % function pruneConstraints


%----------------------------------------------------------------------------
function svm = pruneConstraintsTol(svm, weight, options)

% for better readability
G = svm.qp.G(svm.num_fixed_constraints+1:end,:);
h = svm.qp.h(svm.num_fixed_constraints+1:end);
x = [weight(:); svm.slacks];

% find binding constraints
slack = G*x-h;
idx = find(slack > -options.pruning_slack_tol);

% only keep the binding constraints
svm.qp.G = [svm.qp.G(1:svm.num_fixed_constraints,:); G(idx,:)];
svm.qp.h = [svm.qp.h(1:svm.num_fixed_constraints); h(idx,:)];

end % function pruneConstraintsTol


%----------------------------------------------------------------------------
function svm = pruneConstraintsMax(svm, weight, options)

% for better readability
G = svm.qp.G(svm.num_fixed_constraints+1:end,:);
h = svm.qp.h(svm.num_fixed_constraints+1:end);
x = [weight(:); svm.slacks];

% find least binding constraints
slack = G*x-h;
idx_keep = [];
S = G(:,(svm.dim+1):end);
for i=1:size(S,2)
    s = S(:,i);
    idx = find(s < 0);
    if (numel(idx) > options.pruning_max_constraints_factor)
        [~, idx2] = sort(slack(idx), 'descend');
        idx2 = idx2(1:min(options.pruning_max_constraints_factor, numel(idx)));
        idx_keep = [idx_keep; idx(idx2)];
    else
        idx_keep = [idx_keep; idx];
    end
end

% remove non-binding constraints
svm.qp.G = [svm.qp.G(1:svm.num_fixed_constraints,:); G(idx_keep,:)];
svm.qp.h = [svm.qp.h(1:svm.num_fixed_constraints); h(idx_keep,:)];

end % function pruneConstraintsMax
