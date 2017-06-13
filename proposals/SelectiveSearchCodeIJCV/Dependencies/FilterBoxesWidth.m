function [outBoxes idsGood]= FilterBoxesWidth(inBoxes, maxLen, minLen)
% [outBoxes idsGood]= FilterBoxesWidth(inBoxes, minLen)
%
% Filters out small boxes. Boxes have to have a width and height 
% larger than minLen
%
% inBoxes:       M x 4 array of boxes
% minLen:   Minimum width and height of boxes
%
% outBoxes:      N x 4 array of boxes, N < M
% idsGood:       M x 1 logical array denoting boxes kept
%
%     Jasper Uijlings - 2013

[nr nc] = BoxSize(inBoxes);

idsGood = (nr >= minLen) & (nc >= minLen) & (maxLen >= nr) & (maxLen >= nc) & (nr./nc > 1.5);
outBoxes = inBoxes(idsGood,:);


