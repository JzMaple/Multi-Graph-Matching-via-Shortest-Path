% a simple function to crop the a dataset of a specific cluster
function datasetCrop = cropTarget(tmpClusterPosition)
    global dataset
    datasetCrop = dataset;
    tmp = double([1:length(tmpClusterPosition)]);
    datasetCrop.config.selectGraphMask = {tmp};
    datasetCrop.config.graphCnt = length(tmpClusterPosition);
    tmpMask = dataset.pairwiseMask{1};
    nDivide = ones([1 dataset.config.graphCnt])*datasetCrop.config.nodeCnt;
    tmpMaskCell = mat2cell(tmpMask,nDivide,nDivide);
    for i = 1:length(tmpClusterPosition)
        for j = 1:length(tmpClusterPosition)
            tmpMaskCellCrop(i,j) = tmpMaskCell(tmpClusterPosition(i),tmpClusterPosition(j));
        end
    end
    maskCrop = cell2mat(tmpMaskCellCrop);
    datasetCrop.pairwiseMask = {maskCrop};
end