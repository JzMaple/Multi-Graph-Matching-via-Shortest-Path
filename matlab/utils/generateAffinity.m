function [affinity]= generateAffinity(testk,permutation)
global dataset GT
bUnaryEnable = dataset.config.bUnaryEnable;
bEdgeEnable = dataset.config.bEdgeEnable;
affinity.BiDir = dataset.config.affinityBiDir;
% affinity.Factorize = dataset.config.affinityFGM;
% adjFlag = dataset.config.adjFlag;
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
else % fully complete, hence all are ones
    dataset.pairwiseMask{testk} = ones(graphCnt*nodeCnt,graphCnt*nodeCnt);
end

affinity.GT = cell(graphCnt,graphCnt);
for vii = 1:graphCnt
    for vjj = 1:graphCnt
        vi = permutation(vii); vj = permutation(vjj);
        s1 = dataset.data{vi}.shuffle;
        s2 = dataset.data{vj}.shuffle;
        gtMat = zeros(nodeCnt, nodeCnt);
        for i = 1:nodeCnt
            for j = 1:nodeCnt
                if s1(i) == s2(j)
                    gtMat(i, j) = 1; break;
                end
            end
        end
        affinity.GT{vii, vjj} = gtMat;
    end
end
affinity.GT = cell2mat(affinity.GT);

adjlen = zeros(graphCnt,1);
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
    % ��ת��[-1,1]
    Data{viewk}.angle = Data{viewk}.angle/90;
    Data{viewk}.angle = Data{viewk}.angle + Data{viewk}.angle';
            
    Data{viewk}.edgeRaw = Data{viewk}.edge;
    Data{viewk}.angleRaw = Data{viewk}.angle;
    
    if adjFlag>1%1��full connect
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
    else%1��full connect
        Data{viewk}.adjMatrix = ones(Data{viewk}.nP, Data{viewk}.nP);
    end
    adjlen(viewk) = sum(Data{viewk}.adjMatrix(:));
end
switch adjFlag
    case {1,5}% 1:full connected,5:���ܸ�
        for viewk=1:graphCnt
            Data{viewk}.adjMatrix = logical(Data{viewk}.adjMatrix);
            Data{viewk}.nE = sum(Data{viewk}.adjMatrix(:));
        end
    case {2,3,4}% 2:����,3:���,4:����,��˳�򱻴��ҵ�ʱ�򣬵���ȡ��������work ��Ҫ���ض��� ��ȡ������
        maxid = find(adjlen==max(adjlen));reference=maxid(1);
        refAdj = Data{reference}.adjMatrix;
        
        if adjFlag==2||adjFlag==4%2�ǲ�����4�ǽ�������Ҫ��һ����ʵ����ϡ��reference adj
            for viewk=1:graphCnt
                if viewk~=reference
                    %����adjMatrix: reference (r) vs. viewk (v)
                    permGT = GT((reference-1)*nodeCnt+1:reference*nodeCnt,(viewk-1)*nodeCnt+1:viewk*nodeCnt);
                    permgt = mat2perm(permGT);%permgt(r)=v
                    if adjFlag==2%���� ���ӷ�
                        for i=1:nodeCnt
                            for j=i+1:nodeCnt
                                refAdj(i,j) = refAdj(i,j)+Data{viewk}.adjMatrix(permgt(i),permgt(j));
                            end
                        end
                    elseif adjFlag==4%���� ���˷�
                        for i=1:nodeCnt
                            for j=i+1:nodeCnt
                                refAdj(i,j) = refAdj(i,j)*Data{viewk}.adjMatrix(permgt(i),permgt(j));
                            end
                        end
                    end
                    refAdj = refAdj + refAdj';
                end
            end
        end
        refAdj = logical(refAdj);
        %������reference adj����������adj
        for viewk=1:graphCnt
            if viewk~=reference
                %����adjMatrix: viewk (v) vs. reference (r)
                permGT = GT((viewk-1)*nodeCnt+1:viewk*nodeCnt,(reference-1)*nodeCnt+1:reference*nodeCnt);
                permgt = mat2perm(permGT);%permgt(v)=r
                adj = zeros(nodeCnt,nodeCnt);
                for i=1:nodeCnt
                    for j=i+1:nodeCnt
                        adj(i,j) = refAdj(permgt(i),permgt(j));
                    end
                end
                Data{viewk}.adjMatrix = adj + adj';
            end
        end
end

affinity.EG = cell(graphCnt,1);

for viewk=1:graphCnt
    vk = permutation(viewk);
    Data{viewk}.adjMatrix = logical(Data{viewk}.adjMatrix);
    Data{viewk}.edge(~Data{viewk}.adjMatrix) = NaN;
    Data{viewk}.angle(~Data{viewk}.adjMatrix) = NaN;
    Data{viewk}.nE = sum(sum(Data{viewk}.adjMatrix));
            
    [r,c]=find(~isnan(Data{viewk}.edge));
    affinity.EG{viewk}=[r,c]';%2*Data.nE{1} ��2*�ߵ���Ŀ ��һ������� �ڶ������յ�
    Data{viewk}.edgeFeat = Data{viewk}.edge(~isnan(Data{viewk}.edge))';%edgeFeat��һ��1*�����ľ���,edge��triangle��mask
    Data{viewk}.angleFeat = Data{viewk}.angle(~isnan(Data{viewk}.angle))';
    if bUnaryEnable
        Data{viewk}.pointFeat = dataset.data{vk}.feat';%һ����һ�����feature @todo
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
    affinity.edgeRaw{viewk} = Data{viewk}.edgeRaw;%�������ֱ��������admm
    affinity.angleRaw{viewk} = Data{viewk}.angleRaw;
    affinity.adj{viewk} = Data{viewk}.adjMatrix;
    
end
%�������affinity����
for xview = 1:graphCnt
%     for yview = xview+1:graphCnt
    if affinity.BiDir
        yviewSet = [1:xview-1,xview+1:graphCnt];
    else
        yviewSet = xview+1:graphCnt;
    end
    for yview = yviewSet
        % % % ����һ���������ƶ�     
        if bUnaryEnable%@todo
            featAffinity = conDst(Data{xview}.pointFeat, Data{yview}.pointFeat,0)/10000/128;
            affinity.KP{xview,yview} = exp(-featAffinity/ Sacle_2D);%��ô����Ȩ�أ���
        else
            affinity.KP{xview,yview} = zeros(Data{xview}.nP, Data{yview}.nP);
        end
        % % %
        dq = zeros(length(Data{xview}.edgeFeat),length(Data{yview}.edgeFeat));
        if bEdgeEnable
            if isfield(Data{xview},'edgeFeat') && affinity.edgeAffinityWeight>0
                dq = dq + affinity.edgeAffinityWeight*conDst(Data{xview}.edgeFeat, Data{yview}.edgeFeat,0);%��������ͼ��߱�֮��ľ���ƽ�����γɱߵľ������n1*n2
            end
            if isfield(Data{xview},'angleFeat') && affinity.angleAffinityWeight>0
                dq = dq + affinity.angleAffinityWeight*conDst(Data{xview}.angleFeat, Data{yview}.angleFeat,1);
            end
    %         affinity.DQ{xview,yview} = dq;
            affinity.KQ{xview,yview} = exp(-dq / Sacle_2D);%���ߺͱ�֮����affinity Kq����ָ����ʽ�Ƚ�ƽ�������Ż�,�γ�ָ������ı߱߾���n1*n2
    %         affinity.Ct{xview,yview} = ones(size(affinity.KP{xview,yview}));
        else
            affinity.KQ{xview,yview} = dq;
        end
        affinity.K{xview,yview} = conKnlGphKU(affinity.KP{xview,yview}, affinity.KQ{xview,yview}, affinity.EG{xview},affinity.EG{yview});%EG�ǱߵĶ˵��������2*n1,2*n2
    end
end

%������affinity������һЩfactorized����Ϣ
% ��factMultiGraph����������
% if affinity.Factorize
%     for i=1:graphCnt
%         affinity.HH{i} = affinity.H{i}' * affinity.H{i};
%         affinity.IndG{i} = mat2ind(affinity.G{i});
%         affinity.IndGT{i} = mat2ind(affinity.G{i}');
%         affinity.IndH{i} = mat2ind(affinity.H{i});
%         affinity.IndHT{i} = mat2ind(affinity.H{i}');
%     end
%     
%     affinity.L = cell(graphCnt,graphCnt);
%     affinity.UU = cell(graphCnt,graphCnt);
%     affinity.VV = cell(graphCnt,graphCnt);
%     affinity.UUHH = cell(graphCnt,graphCnt);
%     affinity.VVHH = cell(graphCnt,graphCnt);
%     affinity.HUH = cell(graphCnt,graphCnt);
%     affinity.HVH = cell(graphCnt,graphCnt);
%     affinity.GKGP = cell(graphCnt,graphCnt);
%     
%     
%     
%     for i=1:graphCnt
%         if affinity.BiDir
%             jset = [1:i-1,i+1:graphCnt];
%         else
%             jset = i+1:graphCnt;
%         end
%         for j=jset
%             % L
% %             fprintf('i=%d,j=%d\n',i,j);
%             affinity.L{i,j} = [affinity.KQ{i,j}, -affinity.KQ{i,j} * affinity.G{j}'; 
%                 -affinity.G{i} * affinity.KQ{i,j}, affinity.G{i} * affinity.KQ{i,j} * affinity.G{j}' + affinity.KP{i,j}];
%             % SVD
%             [U, S, V] = svd(affinity.L{i,j});
%             s = diag(S);
%             idx = 1 : length(s);
%             %idx = find(s > 0);
%     %         pr('svd: %d/%d', length(idx), length(s));
%     %         k = length(idx);
%             U = U(:, idx);
%             V = V(:, idx);
%             s = s(idx);
% 
%             U = multDiag('col', U, real(sqrt(s)));
%             V = multDiag('col', V, real(sqrt(s)));
% 
%             % the following decomposition works very badly
%             % U = eye(size(L, 1));
%             % V = L;
% 
%             % auxiliary variables that will be frequently used in the optimization
%             affinity.UU{i,j} = U * U';
%             affinity.VV{i,j} = V * V';
% 
%             affinity.UUHH{i,j} = affinity.UU{i,j} .* affinity.HH{i};
%             affinity.VVHH{i,j} = affinity.VV{i,j} .* affinity.HH{j};
%             affinity.HUH{i,j} = affinity.H{i} * affinity.UUHH{i,j} * affinity.H{i}';
%             affinity.HVH{i,j} = affinity.H{j} * affinity.VVHH{i,j} * affinity.H{j}';
%             affinity.GKGP{i,j} = -affinity.G{i} * affinity.KQ{i,j} * affinity.G{j}' + affinity.KP{i,j};
%         end
%     end
% end

