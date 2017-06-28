function vis_RPN_Results(imdb, roidb, aboxes)

    rois = roidb.rois;
    for iIM = 1:length(rois)
        im = imread(imdb.image_at(iIM));
        % boxes = arrayfun(@(x) rois(iIM).boxes(rois(iIM).class == x, :), 1:length(imdb.classes), 'UniformOutput', false);
        boxes = aboxes{iIM};
        pick = nms(boxes, 0.3);
        boxes = boxes(pick, :);
        idx = find(boxes(:,5)>0.98);
        boxes = boxes(idx, 1:4);
        boxes = {boxes};
        legends = imdb.classes;
        showboxes(im, boxes, legends);
        pause(0.1);
    end
end
