function delta = multiclass_loss(example, y_pred)

% TODO: documentation

delta = double(example.label ~= y_pred);
