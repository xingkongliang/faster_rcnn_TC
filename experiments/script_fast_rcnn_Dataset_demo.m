function script_fast_rcnn_Dataset_demo()
close all;
clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

opts.per_nms_topN           = 6000;
opts.nms_overlap_thres      = 0.3;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;

opts.test_scales            = 600;
dataset_name = 'TownCentre';
% model
model                       = Model.VGG16_for_Fast_RCNN_VOC2007();
% cache name
opts.cache_name             = ['fast_rcnn_' dataset_name '_VGG16'];
% config
conf                        = fast_rcnn_config('image_means', model.mean_image);
% train/test data
dataset                     = [];
dataset                     = Dataset.Dataset_trainval_ss(dataset, dataset_name, 'train', conf.use_flipped);
dataset                     = Dataset.Dataset_test_ss(dataset, dataset_name, 'test', false);

% do validation, or not
opts.do_val                 = false; 

%% -------------------- TRAINING --------------------

% opts.fast_rcnn_model        = fast_rcnn_train(conf, dataset.imdb_train, dataset.roidb_train, ...
%                                 'do_val',           opts.do_val, ...
%                                 'imdb_val',         dataset.imdb_test, ...
%                                 'roidb_val',        dataset.roidb_test, ...
%                                 'solver_def_file',  model.solver_def_file, ...
%                                 'net_file',         model.net_file, ...
%                                 'cache_name',       opts.cache_name);
                            
opts.fast_rcnn_model        = ['/root/Workspace/Caffe_projects/faster_rcnn_TC/output/'...
                               'fast_rcnn_cachedir/fast_rcnn_PL_Pizza_VGG16/PL_Pizza_trainval/final'];

assert(exist(opts.fast_rcnn_model, 'file') ~= 0, 'not found trained model');

                                
%% -------------------- TESTING --------------------
                              fast_rcnn_test_show(conf, dataset_name, dataset.imdb_test, dataset.roidb_test, ...
                                    'net_def_file',     model.test_net_def_file, ...
                                    'net_file',         opts.fast_rcnn_model, ...
                                    'cache_name',       opts.cache_name);


%% -------------------- TESTING --------------------


end


function fast_rcnn_test_show(conf, dataset_name, imdb, roidb, varargin)
% mAP = fast_rcnn_test(conf, imdb, roidb, varargin)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb',                              @isstruct);
    ip.addRequired('roidb',                             @isstruct);
    ip.addParamValue('net_def_file',    '', 			@isstr);
    ip.addParamValue('net_file',        '', 			@isstr);
    ip.addParamValue('cache_name',      '', 			@isstr);                                         
    ip.addParamValue('suffix',          '',             @isstr);
    ip.addParamValue('ignore_cache',    false,          @islogical);
    
    ip.parse(conf, imdb, roidb, varargin{:});
    opts = ip.Results;
    

%%  set cache dir
    cache_dir = fullfile(pwd, 'output', 'fast_rcnn_cachedir', opts.cache_name, imdb.name);
    mkdir_if_missing(cache_dir);

%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
    diary(log_file);
    
    num_images = length(imdb.image_ids);
    num_classes = imdb.num_classes;
    
    try
      aboxes = cell(num_classes, 1);
      if opts.ignore_cache
          throw('');
      end
      for i = 1:num_classes
        load(fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]));
        aboxes{i} = boxes;
      end
    catch    
%%      testing 
        % init caffe net
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = caffe.Net(opts.net_def_file, 'test');
        caffe_net.copy_from(opts.net_file);

        % set random seed
        prev_rng = seed_rand(conf.rng_seed);
        caffe.set_random_seed(conf.rng_seed);

        % set gpu/cpu
        if conf.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end             

        % determine the maximum number of rois in testing 
        max_rois_num_in_gpu = check_gpu_memory(conf, caffe_net);

        disp('opts:');
        disp(opts);
        disp('conf:');
        disp(conf);
        
        %heuristic: keep an average of 40 detections per class per images prior to NMS
        max_per_set = 400 * num_images;
        % heuristic: keep at most 100 detection per class per image prior to NMS
        max_per_image = 400;
        % detection thresold for each class (this is adaptively set based on the max_per_set constraint)
        thresh = -inf * ones(num_classes, 1);
        % top_scores will hold one minheap of scores per class (used to enforce the max_per_set constraint)
        top_scores = cell(num_classes, 1);
        % all detections are collected into:
        %    all_boxes[cls][image] = N x 5 array of detections in
        %    (x1, y1, x2, y2, score)
        aboxes = cell(num_classes, 1);
        box_inds = cell(num_classes, 1);
        for i = 1:num_classes
            aboxes{i} = cell(length(imdb.image_ids), 1);
            box_inds{i} = cell(length(imdb.image_ids), 1);
        end

        count = 0;
        for i = 1:num_images
            count = count + 1;
            fprintf('%s: test (%s) %d/%d \n', procid(), imdb.name, count, num_images);
            d = roidb.rois(i);
            im = imread(imdb.image_at(i));

            [boxes, scores] = fast_rcnn_im_detect(conf, caffe_net, im, d.boxes, max_rois_num_in_gpu);
            % visualize
            class = 'person';
            classes = cell(1);
            classes{1} = class;
            boxes_cell = cell(length(classes), 1);
            thres = 0.9;
            nms_thr = 0.4;
            for j = 1:length(boxes_cell)
                boxes_cell{j} = [boxes(:, (1+(j-1)*4):(j*4)), scores(:, j)];
                boxes_cell{j} = boxes_cell{j}(nms(boxes_cell{j}, nms_thr), :);

                I = boxes_cell{j}(:, 5) >= thres;
                boxes_cell{j} = boxes_cell{j}(I, :);
            end
            figure(1)
            imshow(im);
            bb = boxes_cell{1}(:,1:4);
            for k = 1:length(bb)
                rectangle('Position',[bb(k,1),bb(k,2),bb(k,3)-bb(k,1)+1,bb(k,4)-bb(k,2)+1],'EdgeColor','r','LineWidth',2);
            end
            % showboxes(im, boxes_cell, classes, 'voc');
            SaveImagePath = sprintf('Result_imgs/%s_Fast_rcnn_vgg16_nms%.2f_thres%.2f_OnPL_PizzaModel', dataset_name, nms_thr, thres);
            
            if ~exist(SaveImagePath) 
                mkdir(SaveImagePath
            end
            Result_imagepath = [SaveImagePath '/Fast_RCNN_%s.jpg'];
            saveas(gca, sprintf(Result_imagepath, imdb.image_ids{i}), 'jpg');
            pause(0.1);
 

        end

        caffe.reset_all(); 
        rng(prev_rng);
    end

end

function max_rois_num = check_gpu_memory(conf, caffe_net)
%%  try to determine the maximum number of rois

    max_rois_num = 0;
    for rois_num = 500:500:5000
        % generate pseudo testing data with max size
        im_blob = single(zeros(conf.max_size, conf.max_size, 3, 1));
        rois_blob = single(repmat([0; 0; 0; conf.max_size-1; conf.max_size-1], 1, rois_num));
        rois_blob = permute(rois_blob, [3, 4, 1, 2]);

        net_inputs = {im_blob, rois_blob};

        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);

        caffe_net.forward(net_inputs);
        gpuInfo = gpuDevice();

        max_rois_num = rois_num;
            
        if gpuInfo.FreeMemory < 2 * 10^9  % 2GB for safety
            break;
        end
    end

end


function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
    end
end
