function dataset = Dataset_trainval(dataset, dataset_name, usage, use_flip)
% Pascal voc 2007 trainval set
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install
devkit                      = ['./datasets/' dataset_name '/' dataset_name];

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_voc(devkit, dataset_name, 'trainval', '2007', use_flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_voc(devkit, dataset_name, 'trainval', '2007', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    otherwise
        error('usage = ''train'' or ''test''');
end

end