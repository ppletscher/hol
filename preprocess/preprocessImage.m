function example = preprocessImage(filename, data_dir, options)

options_default = defaultOptions();
if (nargin >= 3)
    options = processOptions(options, options_default);
else
    options = options_default;
end

example = [];

fname = [data_dir 'images/' filename];
img = imread(fname);
img = im2double(img);
fname = [data_dir 'images-gt/' filename(1:end-4) '.png'];
label = imread(fname);
label = label > 0;
fname = [data_dir 'images-labels/' filename(1:end-4) '-anno.png'];
label_seed = imread(fname);

example.image = img;
example.label = label;
example.label_seed = label_seed;


opts = [];
opts.posteriorMethod = 'gmm_bs_mixtured';
opts.gmmNmix_fg=5;
opts.gmmNmix_bg=5;
opts.gmmUni_value=1; % assuming features in [0,1]
opts.gmmLikeli_gamma=0.05;
opts.featureSpace = 'rgb';
opts.gcGamma = 150;
opts.gcSigma_c = 'auto';
opts.gcScale = 50;
%opts.gcNbrType = 'colourGradient';
%opts.gcNbrType = 'colourGradient8';
opts.gcNbrType = options.connectivity;

[features, edges, edge_weights, edge_types] = extractFeatures(img, opts);
example.features = features;
example.edge_weights = edge_weights;

posterior_image = getPosteriorImage(features, label_seed, opts);
example.posterior_image = posterior_image;

posterior_image = posterior_image(:)';
bg_clamp=(label_seed(:)==2);
fg_clamp=(label_seed(:)==1);

prob_densities=[-log(1-posterior_image); -log(posterior_image)];
prob_densities(prob_densities>100)=100;

% TODO: check again which values to use??
%prob_densities(2,bg_clamp)=inf;prob_densities(1,bg_clamp)=0;
%prob_densities(1,fg_clamp)=inf;prob_densities(2,fg_clamp)=0;
prob_densities(2,bg_clamp)=2000;prob_densities(1,bg_clamp)=0;
prob_densities(1,fg_clamp)=2000;prob_densities(2,fg_clamp)=0;

prob_densities=prob_densities*opts.gcScale;
example.prob_densities = prob_densities;

opts=gscSeq.segOpts();
segH=gscSeq.segEngine(0,opts);
segH.preProcess(im2double(img));
ok=segH.start(example.label_seed);
gscSeq_label = double(segH.seg>0);

% TODO: check whether we should reenable this!
%example.features_unary = [example.features; double(example.prob_densities); gscSeq_label(:)'; ones(1, size(example.features,2))];
example.features_unary = [example.features; double(example.prob_densities); ones(1, size(example.features,2))];

example.edges = edges;
example.edges_type = edge_types;
if (~options.different_edge_type)
    example.edges_type = repmat(1, [1 size(example.edges,2)]);
end
example.features_pairwise = [double(edge_weights(:)'); ones(1,numel(edge_weights))];


function options = defaultOptions()

options = [];
options.connectivity = 'colourGradient';
options.different_edge_type = 0;

