function [prediction, err] = predictDataset(example_filenames, weight)

prediction = cell(numel(example_filenames), 1);
err = zeros(numel(example_filenames), 3);

for idx=1:numel(example_filenames)
    % predict
    label = predict(example_filenames{idx}, weight, 0, 'ibfs', 0);
    prediction{idx} = label;

    % Hamming loss
    loss_type = 'hamming';
    normalize_loss = 1;
    weight_hamming = 1;
    weight_count = 1;
    delta_hamming = loss(example_filenames{idx}, label, loss_type, normalize_loss, weight_hamming, weight_count);
    
    % count loss
    loss_type = 'count';
    normalize_loss = 1;
    weight_hamming = 1;
    weight_count = 1;
    delta_count = loss(example_filenames{idx}, label, loss_type, normalize_loss, weight_hamming, weight_count);
    
    % combined loss
    loss_type = 'combined';
    normalize_loss = 1;
    weight_hamming = 0.5;
    weight_count = 0.5;
    delta_combined = loss(example_filenames{idx}, label, loss_type, normalize_loss, weight_hamming, weight_count);

    err(idx, 1) = delta_hamming;
    err(idx, 2) = delta_count;
    err(idx, 3) = delta_combined;
end
