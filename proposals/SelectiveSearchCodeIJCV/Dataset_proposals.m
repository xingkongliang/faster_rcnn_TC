% This demo shows how to use the software described in our IJCV paper: 
%   Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
%%
function Dataset_proposals(dataset_name)
% dataset_name = 'PL_Pizza';
% 
addpath('Dependencies');

fprintf('Demo of how to run the code for:\n');
fprintf('   J. Uijlings, K. van de Sande, T. Gevers, A. Smeulders\n');
fprintf('   Segmentation as Selective Search for Object Recognition\n');
fprintf('   IJCV 2013\n\n');

% Compile anisotropic gaussian filter
if(~exist('anigauss'))
    fprintf('Compiling the anisotropic gauss filtering of:\n');
    fprintf('   J. Geusebroek, A. Smeulders, and J. van de Weijer\n');
    fprintf('   Fast anisotropic gauss filtering\n');
    fprintf('   IEEE Transactions on Image Processing, 2003\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://staff.science.uva.nl/~mark/downloads.html#anigauss\n\n');
    mex Dependencies/anigaussm/anigauss_mex.c Dependencies/anigaussm/anigauss.c -output anigauss
end

if(~exist('mexCountWordsIndex'))
    mex Dependencies/mexCountWordsIndex.cpp
end

% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('mexFelzenSegmentIndex'))
    fprintf('Compiling the segmentation algorithm of:\n');
    fprintf('   P. Felzenszwalb and D. Huttenlocher\n');
    fprintf('   Efficient Graph-Based Image Segmentation\n');
    fprintf('   International Journal of Computer Vision, 2004\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://www.cs.brown.edu/~pff/segment/\n');
    fprintf('Note: A small Matlab wrapper was made. See demo.m for usage\n\n');
%     fprintf('   
    mex Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
end

%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
ks = [50 100 150 300]; % controls size of segments of initial segmentation. 
sigma = 0.8;

% After segmentation, filter out boxes which have a width/height smaller
% than minBoxWidth (default = 20 pixels).
minBoxWidth = 20;
maxBoxWidth = 200;
% Comment the following three lines for the 'quality' version
colorTypes = colorTypes(1:2); % 'Fast' uses HSV and Lab
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies
ks = ks(1:2);

% Test the boxes
if nargin < 1
    dataset_name = 'PNNLParkingLot2';
end
root_dir = ['./datasets/' dataset_name '/' dataset_name];
VOCopts = get_voc_opts(root_dir);
VOCopts.testset = dataset_name;
year = '2007';
image_sets = {'trainval', 'test', 'val'};

for s = 1:length(image_sets)
    image_set = image_sets{s};
    if strcmp(dataset_name,'voc')
        imdb.name = ['voc_' year '_' image_set];
    else   
        imdb.name = [dataset_name '_' image_set];
    end
    imdb.image_dir = fileparts(VOCopts.imgpath);
    imdb.image_ids = textread(sprintf(VOCopts.imgsetpath, image_set), '%s');
    imdb.extension = 'jpg';

    image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
    fprintf('After box extraction, boxes smaller than %d pixels and larger than %d will be removed\n', minBoxWidth, maxBoxWidth);
    fprintf('Obtaining boxes for %s set:\n', dataset_name);
    totalTime = 0;

    for i = 1:length(imdb.image_ids)
        im = imread(image_at(i));
        fprintf('%d ', i);  
        % VOCopts.img
        idx = 1;
        for j=1:length(ks)
            k = ks(j); % Segmentation threshold k
            minSize = k; % We set minSize = k
            for n = 1:length(colorTypes)
                colorType = colorTypes{n};
                tic;
                [boxesT{idx} blobIndIm blobBoxes hierarchy priorityT{idx}] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
                totalTime = totalTime + toc;
                idx = idx + 1;
            end
        end
        boxes{i} = cat(1, boxesT{:}); % Concatenate boxes from all hierarchies
        priority = cat(1, priorityT{:}); % Concatenate priorities

        % Do pseudo random sorting as in paper
        priority = priority .* rand(size(priority));
        [priority sortIds] = sort(priority, 'ascend');
        boxes{i} = boxes{i}(sortIds,:);
        boxes{i} = FilterBoxesWidth(boxes{i}, maxBoxWidth, minBoxWidth);
        boxes{i} = BoxRemoveDuplicates(boxes{i});
        images{i} = imdb.image_ids{i};
    %     visual boxes and image
    %     box = boxes{i};
    %     imshow(im);
    %     for i=1:length(box)
    %         rectangle('Position',[box(i,2),box(i,1),box(i,4)-box(i,2)+1,box(i,3)-box(i,1)+1],'EdgeColor','r','LineWidth',1);
    %     end    
    end
    save(fullfile('./data/selective_search_data',[dataset_name '_' image_set]), 'boxes', 'images');
    fprintf('\n');
    fprintf('%s is done!\n', image_set);
end

