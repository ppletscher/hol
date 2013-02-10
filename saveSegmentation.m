function saveSegmentation(filename, image, segmentation)

perim = bwperim(segmentation);
perim = double(perim);

perim_small = perim;

se = strel(ones(5,5));
perim_large = imdilate(perim, se, 'same');

show = image;

% outer
perim = find(perim_large);
show(perim) = 1;
dim = size(image,1)*size(image,2);
show(perim+dim) = 0;
show(perim+dim*2) = 0;

% inner
perim = find(perim_small);
show(perim) = 1;
dim = size(image,1)*size(image,2);
show(perim+dim) = 1;
show(perim+dim*2) = 1;

imwrite(show, filename);

% TODO: thicker, nicer colors!
