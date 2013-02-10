function grabcutPrepare()

addpath('matluster');

run_idx = 0;

% initialization
if (~exist('local', 'dir'))
    mkdir('local');
end
fid = fopen('submit_grabcut.sh', 'w');

% formatting
format = [];
format.featureset = '%s';
format.loss = '%s';
format.lambda = '%f';
format.seed = '%d';

% reporting
reporting = [];
reporting.groupby = {'featureset', 'loss'};

% init options
options = [];
options.format = format;
options.reporting = reporting;

% grid search
featuresets = {'4conn_allsameparam', '4conn_differentparam', '8conn_allsameparam', '8conn_differentparam'};
losses = {'hamming', 'count'};
lambdas = 10.^[-3,-2,-1, 0, 1 2];
for featureset_idx=1:numel(featuresets)
    options.featureset = featuresets{featureset_idx};
    for loss_idx=1:numel(losses)
        options.loss = losses{loss_idx};
        for seed=[1,7,13,31]
            options.seed = seed;
            for lambda=lambdas
                options.lambda = lambda;

                filename = sprintf('local/options_%d.mat', run_idx);
                save(filename, 'options');
                matluster_addJobToQueue(fid, options, run_idx, './run_grabcutMain.sh /cluster/apps/matlab/7.14/', '08:00');
        
                run_idx = run_idx+1;
            end
        end
    end
end

num_runs = run_idx;
save('local/num_runs.mat', 'num_runs');

fclose(fid);
unix('chmod +x submit_grabcut.sh');
