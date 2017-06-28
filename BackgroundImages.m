function BackgroundImages()
%BACKGROUNDIMAGES �˴���ʾ�йش˺����ժҪ
%   �˴���ʾ��ϸ˵��
datasetname = 'PNNLParkingLot2';
paras.ShowFg = false;
devkit = ['./datasets/' datasetname '/' datasetname];
background_image_file = [devkit '/background_image.png'];

use_flip = false;
VOCopts = get_voc_opts(devkit);
VOCopts.bgimpath = [VOCopts.datadir '/' VOCopts.dataset '/Bg_JPEGImages/Bg_%s.jpg'];
dataset = [];
dataset.imdb_train = imdb_from_voc(devkit, 'trainval', '2007', use_flip);
dataset.imdb_test  = imdb_from_voc(devkit, 'test', '2007', use_flip) ;

if ~exist([devkit '/' datasetname '/Bg_JPEGImages'])
    mkdir([devkit '/' datasetname '/Bg_JPEGImages']);
end

%% produce a background image
if exist(background_image_file)
    sumFrame = imread(background_image_file);   
    sumFrame = double(sumFrame);
else
    for i = 1:length(dataset.imdb_train.image_ids)
        frame = imread(dataset.imdb_train.image_at(i));
        if(i==1)  sumFrame = double(frame); 
        else sumFrame = sumFrame+double(frame); end        
        fprintf('background modeling.. the %d th frame\n',i);         
    end
    sumFrame = sumFrame/length(dataset.imdb_train.image_ids); 
    imwrite(uint8(sumFrame), background_image_file);
end

%% produce background probability images for every origin image

for i = 1:length(dataset.imdb_train.image_ids)
    frame = imread(dataset.imdb_train.image_at(i));
    diffImage  =  abs(sumFrame - double(frame)); 
    %sumarizing multiple channels
    if(size(diffImage,3)>1)
        fgMask = sum(diffImage,3)/3; 
    else
        fgMask = diffImage; 
    end
    %foreground detection   
    if (paras.ShowFg)  
        figure(1),imshow(uint8(fgMask)); 
    end   
    imwrite(uint8(fgMask),sprintf(VOCopts.bgimpath, dataset.imdb_train.image_ids{i}));
    fprintf('produce background probalility image ... the %d th frame\n',i);  
end
           
end

