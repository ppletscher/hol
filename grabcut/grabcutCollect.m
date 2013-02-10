function grabcutCollect()

addpath('matluster');

% collect results
load('local/num_runs.mat');
collection = matluster_collect(num_runs);

evaluateRatio(collection, 'count');
evaluateRatio(collection, 'hamming');

% save the segmentations
lambda = 1.0;
featureset = '4conn_allsameparam';
seed = 13;
for run_idx=0:(num_runs-1)
    % options
    load(sprintf('local/options_%d.mat', run_idx));
    
    if ((options.lambda == lambda) && (isequal(options.featureset,featureset)) && (options.seed == seed))
        saveSegmentations(options);
    end
end



function evaluateRatio(collection, eval_loss)
% for each split and loss/dataset combination find the best lambda, then
% report the ratio of them!

% transform to tables and show relevant information
for i=0:3
    % evaluate for count loss
    fprintf('evaluate the %s loss.\n', eval_loss);

    table_hamming = matluster_reshapeResults(collection{2*i+1}, sprintf('err_val_%s', eval_loss)); % Hamming trained
    table_count = matluster_reshapeResults(collection{2*i+2},  sprintf('err_val_%s', eval_loss)); % count trained

    [~,idx_hamming] = min(table_hamming.results, [], 2);
    [~,idx_count] = min(table_count.results, [], 2);
    idx_hamming = sub2ind(size(table_hamming.results), [1:size(table_hamming.results,1)]', idx_hamming);
    idx_count = sub2ind(size(table_count.results), [1:size(table_count.results,1)]', idx_count);

    table_hamming = matluster_reshapeResults(collection{2*i+1}, sprintf('err_test_%s', eval_loss)); % Hamming trained
    table_count = matluster_reshapeResults(collection{2*i+2}, sprintf('err_test_%s', eval_loss)); % count trained
    
    ratio = table_hamming.results(idx_hamming)./table_count.results(idx_count)

    idx = [1:numel(ratio)];
    A = [idx(:) ratio(:)];
    filename = sprintf('output/grabcut_%s_%s.csv', eval_loss, collection{2*i+1}.name{1});
    csvwrite(filename, A);
end



function saveSegmentations(options)

conf_str = matluster_generateStringFromOptions(options);
filename = sprintf('output/%s.mat', conf_str);
load(filename);

mkdir('output/predictions/');
str = sprintf('featureset=%s_seed=%d_lambda=%f', options.featureset, options.seed, options.lambda);
path = sprintf('output/predictions/%s', str); 
mkdir(path);

loss = options.loss;
saveSegmentationsInner(loss, [path, '/train/'], prediction_results_train, example_filenames_train);
saveSegmentationsInner(loss, [path, '/val/'], prediction_results_val, example_filenames_val);
saveSegmentationsInner(loss, [path, '/test/'], prediction_results_test, example_filenames_test);


function saveSegmentationsInner(eval_loss, path, predictions, filenames)

addpath('../helpers/');

mkdir(path);

for i=1:numel(predictions)
    [~,fname_base,~] = fileparts(filenames{i});

    % image itself
    filename = ['data/images/' fname_base, '.jpg'];
    if (exist(filename, 'file'))
        img = imread(filename);
    else
        filename = ['data/images/' fname_base, '.bmp'];
        if (exist(filename, 'file'))
            img = imread(filename);
        else
            filename = ['data/images/' fname_base, '.png'];
            img = imread(filename);
        end
    end
    img = im2double(img);
    filename_out = [path, fname_base, '.png'];
    imwrite(img, filename_out);

    % ground-truth
    filename = ['data/images-gt/' fname_base, '.png'];
    label_gt = imread(filename);
    label_gt = label_gt(:,:,1) > 0;
    filename_out = [path, fname_base, '_gt.png'];
    saveSegmentation(filename_out, img, label_gt);
    
    % prediction
    filename_out = [path, fname_base, '_', eval_loss, '_pred.png'];
    saveSegmentation(filename_out, img, predictions{i});

    % save raw loss numbers
    filename_out = [path, fname_base, '_', eval_loss, '_losses.txt'];
    fid=fopen(filename_out, 'w');
    loss_count = loss(filenames{i}, predictions{i}, 'count',1,1,1);
    fprintf(fid, 'count: %f\n', loss_count);
    loss_ham = loss(filenames{i}, predictions{i}, 'hamming',1,1,1);
    fprintf(fid, 'Hamming: %f\n', loss_ham);
end
