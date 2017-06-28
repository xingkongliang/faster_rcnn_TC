im = imread(imdb.image_at(1));
figure(1)
imshow(im);

for k = 1:17
    bb = aboxes{1}{1,1}(:,1:4);
    rectangle('Position',[bb(k,1),bb(k,2),bb(k,3)-bb(k,1)+1,bb(k,4)-bb(k,2)+1]);
end

figure(i)
imshow(im);
bb = boxes_cell{1}(:,1:4);
for k = 1:length(bb)
    rectangle('Position',[bb(k,1),bb(k,2),bb(k,3)-bb(k,1)+1,bb(k,4)-bb(k,2)+1],'EdgeColor','r','LineWidth',2);
end

figure(1)
imshow(im);
for k=1:length(boxes)
    rectangle('Position',[boxes(k,1),boxes(k,2),boxes(k,3)-boxes(k,1)+1,boxes(k,4)-boxes(k,2)+1]);
end

/root/Workspace/Caffe_projects/faster_rcnn_TC/output/fast_rcnn_cachedir/fast_rcnn_TownCentre_VGG16/TownCentre_trainval/final