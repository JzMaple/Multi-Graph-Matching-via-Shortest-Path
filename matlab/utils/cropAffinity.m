% a simple function to crop the affinity of a specific cluster
function affinityCrop = cropAffinity (tmpClusterPosition)
    global affinity
    affinityCrop.BiDir = affinity.BiDir;
    affinityCrop.edgeAffinityWeight = affinity.edgeAffinityWeight;
    affinityCrop.angleAffinityWeight = affinity.angleAffinityWeight;
    affinityCrop.graphCnt = length(tmpClusterPosition);
    affinityCrop.nodeCnt = affinity.nodeCnt;
    affinityCrop.EG = affinity.EG(tmpClusterPosition,1);
    affinityCrop.nP = affinity.nP(1,tmpClusterPosition);
    affinityCrop.edge = affinity.edge(1,tmpClusterPosition);
    affinityCrop.edgeRaw = affinity.edgeRaw(1,tmpClusterPosition);
    affinityCrop.adj = affinity.adj(1,tmpClusterPosition);
%     affinityCrop.pointFeat = affinity.pointFeat(1,tmpClusterPosition);
%     affinityCrop.nE = affinity.nE(1,tmpClusterPosition);
%     affinityCrop.edgeFeat = affinity.edgeFeat(1,tmpClusterPosition);
    affinityCrop.G = affinity.G(1,tmpClusterPosition);
    affinityCrop.H = affinity.H(1,tmpClusterPosition);
    affinityCrop.KP = affinity.KP(tmpClusterPosition,tmpClusterPosition);
    affinityCrop.KQ = affinity.KQ(tmpClusterPosition,tmpClusterPosition);
    affinityCrop.K = affinity.K(tmpClusterPosition,tmpClusterPosition);
    nDivide = ones([1 affinity.graphCnt])*affinity.nodeCnt;
    cellGT = mat2cell(affinity.GT, nDivide, nDivide);
    for i = 1:length(tmpClusterPosition)
        for j = 1:length(tmpClusterPosition)
            cellGTcrop(i,j)=cellGT(tmpClusterPosition(i),tmpClusterPosition(j));
        end
    end
    affinityCrop.GT = cell2mat(cellGTcrop);
end