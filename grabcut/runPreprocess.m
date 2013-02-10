addpath('preprocess');
addpath('gsc-1.2');

cd gsc-1.2;
setup();
cd ..;

data_dir = 'data/';

%local_dir = 'local/';
options = [];

% preprocess all the images and store preprocessed energies in local/
files = dir([data_dir 'images/*.jpg']);
files = [files; dir([data_dir 'images/*.bmp'])];
files = [files; dir([data_dir 'images/*.png'])];
example_filenames = cell(numel(files),1);
h = waitbar(0, 'Preprocessing images...');
for f_idx = 1:numel(files) 
    %waitbar(f_idx/numel(files), h);
    fprintf('%f.\n', f_idx/numel(files))
    example = preprocessImage(files(f_idx).name, data_dir, options);
    fname = files(f_idx).name(1:end-4);
    fname = [local_dir fname '.mat'];
    save(fname, 'example');
    example_filenames{f_idx} = fname;
end
close(h);

% TODO: we get better results when including the gsc-prediction, however then
% definitely no difference between count and hamming loss!

save(sprintf('%sfilenames.mat', local_dir), 'example_filenames');
