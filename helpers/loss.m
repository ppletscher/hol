function delta = loss(example, label_pred, loss_type, normalize_loss, weight_hamming, weight_count)

load(example);
label_gt = example.label;

delta = 0;
if (isequal(loss_type, 'hamming') || isequal(loss_type, 'combined'))
    sc = weight_hamming;
    if (normalize_loss)
        sc = sc/numel(label_gt);
    end
    delta = numel(find(label_pred~=label_gt))*sc;
end
if (isequal(loss_type, 'count') || isequal(loss_type, 'combined'))
    sc = weight_count;
    if (normalize_loss)
        sc = sc/numel(label_gt);
    end
    delta = delta + abs(sum(label_pred(:)) - sum(label_gt(:)))*sc;
end
