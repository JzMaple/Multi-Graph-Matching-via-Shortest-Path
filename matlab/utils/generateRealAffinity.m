function [affinity]= generateRealAffinity(testk,permutation)
global dataset
bUnaryEnable = dataset.config.bUnaryEnable;
bEdgeEnable = dataset.config.bEdgeEnable;
affinity.BiDir = dataset.config.affinityBiDir;
if strcmp(dataset.config.connect,'fc'), adjFlag=1; else, adjFlag=5; end
affinity.edgeAffinityWeight = dataset.config.edgeAffinityWeight;
affinity.angleAffinityWeight = dataset.config.angleAffinityWeight;
Sacle_2D = dataset.config.Sacle_2D;
nodeCnt = dataset.config.nodeCnt;
graphCnt = dataset.config.graphCnt;
affinity.graphCnt = graphCnt;
affinity.nodeCnt = nodeCnt;

if dataset.config.complete<1
    dataset.pairwiseMask{testk} = generateRandomCompleteMask(nodeCnt,graphCnt,dataset.config.complete);
else
    dataset.pairwiseMask{testk} = ones(graphCnt*nodeCnt,graphCnt*nodeCnt);
end

affinity.GT = cell(graphCnt,graphCnt);
for vi = 1:graphCnt
    for vj = 1:graphCnt
        gtMat = zeros(nodeCnt, nodeCnt);
        for vx = 1:nodeCnt
            for vy = 1:nodeCnt
                if dataset.data{vi}.shuffle(vx) == dataset.data{vj}.shuffle(vy)
                    gtMat(vx, vy) = 1;
                    break;
                end
            end
        end
        affinity.GT{permutation(vi), permutation(vj)} = gtMat;
    end
end
affinity.GT = cell2mat(affinity.GT);

Data = cell(graphCnt,1);
for viewk = 1:graphCnt
    vk = permutation(viewk);
    Data{viewk}.nP = size(dataset.data{vk}.point,1);
    Data{viewk}.edge = zeros(nodeCnt,nodeCnt);
    Data{viewk}.point = dataset.data{vk}.point;
    Data{viewk}.angle = zeros(Data{viewk}.nP,Data{viewk}.nP); 
    for r = 1:nodeCnt
        for c = r+1:nodeCnt
            Data{viewk}.edge(r,c) = sqrt((dataset.data{vk}.point(r,1)-dataset.data{vk}.point(c,1))^2+(dataset.data{vk}.point(r,2)-dataset.data{vk}.point(c,2))^2);
            Data{viewk}.angle(r,c) = 180/pi*atan((dataset.data{vk}.point(r,2)-dataset.data{vk}.point(c,2))/(dataset.data{vk}.point(r,1)-dataset.data{vk}.point(c,1)));
        end
    end
    Data{viewk}.edge = Data{viewk}.edge/max(Data{viewk}.edge(:));
    Data{viewk}.edge = Data{viewk}.edge + Data{viewk}.edge';
    Data{viewk}.angle = Data{viewk}.angle/90;
    Data{viewk}.angle = Data{viewk}.angle + Data{viewk}.angle';
            
    Data{viewk}.edgeRaw = Data{viewk}.edge;
    Data{viewk}.angleRaw = Data{viewk}.angle;
    
    if strcmp(dataset.config.connect,'delaunay')
        tri = delaunay(Data{viewk}.point(:,1),Data{viewk}.point(:,2));
        triNum=size(tri,1);
        Data{viewk}.adjMatrix = zeros( Data{viewk}.nP, Data{viewk}.nP);
        for i=1:triNum
            Data{viewk}.adjMatrix(tri(i,1),tri(i,2))=1;
            Data{viewk}.adjMatrix(tri(i,2),tri(i,1))=1;
            Data{viewk}.adjMatrix(tri(i,2),tri(i,3))=1;
            Data{viewk}.adjMatrix(tri(i,3),tri(i,2))=1;
            Data{viewk}.adjMatrix(tri(i,1),tri(i,3))=1;
            Data{viewk}.adjMatrix(tri(i,3),tri(i,1))=1;
        end
    else
        Data{viewk}.adjMatrix = ones(Data{viewk}.nP, Data{viewk}.nP);
    end
end

for viewk=1:graphCnt
    Data{viewk}.adjMatrix = logical(Data{viewk}.adjMatrix);
    Data{viewk}.nE = sum(Data{viewk}.adjMatrix(:));
end

affinity.EG = cell(graphCnt,1);

for viewk=1:graphCnt
    vk = permutation(viewk);
    Data{viewk}.adjMatrix = logical(Data{viewk}.adjMatrix);
    Data{viewk}.edge(~Data{viewk}.adjMatrix) = NaN;
    Data{viewk}.angle(~Data{viewk}.adjMatrix) = NaN;
    Data{viewk}.nE = sum(sum(Data{viewk}.adjMatrix));
            
    [r,c]=find(~isnan(Data{viewk}.edge));
    affinity.EG{viewk}=[r,c]';
    Data{viewk}.edgeFeat = Data{viewk}.edge(~isnan(Data{viewk}.edge))';
    Data{viewk}.angleFeat = Data{viewk}.angle(~isnan(Data{viewk}.angle))';
    if bUnaryEnable
        Data{viewk}.pointFeat = dataset.data{vk}.feat';
    end
    affinity.nP{viewk} = Data{viewk}.nP;
    % incidence matrix
    affinity.G{viewk} = zeros(Data{viewk}.nP, Data{viewk}.nE);
    for c = 1 : Data{viewk}.nE
        affinity.G{viewk}(affinity.EG{viewk}(:, c), c) = 1;
    end
    % augumented adjacency
    affinity.H{viewk} = [affinity.G{viewk}, eye(Data{viewk}.nP)];
    affinity.edge{viewk} = Data{viewk}.edge;
    affinity.edgeRaw{viewk} = Data{viewk}.edgeRaw;
    affinity.angleRaw{viewk} = Data{viewk}.angleRaw;
    affinity.adj{viewk} = Data{viewk}.adjMatrix;
    
end

for xview = 1:graphCnt
    if affinity.BiDir
        yviewSet = [1:xview-1,xview+1:graphCnt];
    else
        yviewSet = xview+1:graphCnt;
    end
    for yview = yviewSet    
        if bUnaryEnable
            featAffinity = conDst(Data{xview}.pointFeat, Data{yview}.pointFeat,0)/10000/128;
            affinity.KP{xview,yview} = exp(-featAffinity/ Sacle_2D);
        else
            affinity.KP{xview,yview} = zeros(Data{xview}.nP, Data{yview}.nP);
        end
        dq = zeros(length(Data{xview}.edgeFeat),length(Data{yview}.edgeFeat));
        if bEdgeEnable
            if isfield(Data{xview},'edgeFeat') && affinity.edgeAffinityWeight>0
                dq = dq + affinity.edgeAffinityWeight*conDst(Data{xview}.edgeFeat, Data{yview}.edgeFeat,0);
            end
            if isfield(Data{xview},'angleFeat') && affinity.angleAffinityWeight>0
                dq = dq + affinity.angleAffinityWeight*conDst(Data{xview}.angleFeat, Data{yview}.angleFeat,1);
            end
            affinity.KQ{xview,yview} = exp(-dq / Sacle_2D);
        else
            affinity.KQ{xview,yview} = dq;
        end
        affinity.K{xview,yview} = conKnlGphKU(affinity.KP{xview,yview}, affinity.KQ{xview,yview}, affinity.EG{xview},affinity.EG{yview});
    end
end
