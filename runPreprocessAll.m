options = [];

options.connectivity = 'colourGradient';
options.different_edge_type = 0;
local_dir = 'local/4conn_allsameparam/';
mkdir(local_dir);
runPreprocess;

options.connectivity = 'colourGradient';
options.different_edge_type = 1;
local_dir = 'local/4conn_differentparam/';
mkdir(local_dir);
runPreprocess;

options.connectivity = 'colourGradient8';
options.different_edge_type = 0;
local_dir = 'local/8conn_allsameparam/';
mkdir(local_dir);
runPreprocess;

options.connectivity = 'colourGradient8';
options.different_edge_type = 1;
local_dir = 'local/8conn_differentparam/';
mkdir(local_dir);
runPreprocess;

