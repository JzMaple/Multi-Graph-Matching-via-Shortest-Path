% Incremental multi-graph matching
% rawMat: initial global matching n(N+1)*n(N+1), n: node size, N: graph
% rawMat contains only the initial matching within each cluster, and it's
% the composition of previous IMGM result and the new graph
% affMat: edge affinity/similarity
% number, the N+1th is new graph
% affScore: affinity score matrix for each pair of graphs, size N*N
% param: structure storing extra parameters
%        param.method: partition method (1: AP, 2: DPP, 3: random)

function M = IMGM(affScore, rawMat, param)
    global affinity
    [a,b]=size(affScore);
    if a~=b
        error('The similarity matrix must be square!');
    end
    isDPP = 1; isAP = 2; isRand = 3; isTIP = 4;% DPP, AP or random clustering
    n=param.n; N=param.N; 
    massOutlierMode = 0;
    inlierMask =  zeros(n,N);
    %%%%%%%%%%%%%%%%% calculating graph consistency is time consuming %%%
    singleCon = cal_single_graph_consistency(rawMat(1:n*N,1:n*N),n,N,massOutlierMode,inlierMask); % calculate the unary consistency
    [maxCon, indMaxCon] = max(singleCon); % find the graph with maximal consistency
    nDivide = ones([1 N+1])*n;
    
    rawMatCell = mat2cell(rawMat,nDivide,nDivide); % divide rawMat into blocks
    maxScore = max(affScore(:));
    Similarity = arrayfun(@(x) x/maxScore,affScore);
    % Similarity = Similarity(1:end-1, 1:end-1);
    Similarity = Similarity([1:indMaxCon-1,indMaxCon+1:end],[1:indMaxCon-1,indMaxCon+1:end]);
    Similarity = (Similarity*1).^2;
    medianSimilarity = median(Similarity(:));
    p = ones([1 N])*medianSimilarity;
    nCluster = 2;
    %%%%%%%%%%%% visualize graphs if specified %%%%%%%%%%%%%%%%
    if param.visualization == 1
        maxSim = max(Similarity(:));
        tmpSimilarity = Similarity + eye(N)*1.05*maxSim;
        tmpSimilarity = ones(N,N)*1.05*maxSim - tmpSimilarity;
        pointMDS = mdscale(tmpSimilarity.^2, 2);
    end
    %%%%%%%%%%%%%%%%% generating the topology of hypergraph %%%%%%%%%%%%%%%
    if param.method == isAP
        %tic;
        [idx,netsim,dpsim,expref]=apcluster(-Similarity,-p); % perform AP-clustering
        % [idxCluster]=apcluster(Similarity,p);
        idx(find(idx>=indMaxCon)) = idx(find(idx>=indMaxCon)) + 1;
        C_index = unique(idx);
        idx = [idx(1:indMaxCon-1);-1;idx(indMaxCon:end)];
        %toc
    end
    
    if param.method == isDPP
        %tic;
        affMax = max(affScore(:));
        affScoreTmp = affScore + eye(a)*(affMax*1.1);
        affMin = min(affScoreTmp(:));
        affScoreTmp = affScoreTmp - affMin*0.9;
        affScoreTmp = affScoreTmp/max(affScoreTmp(:));
        [idx, partition] = dpp_graph_partition(affScoreTmp,nCluster);
        C_index = unique(idx);
        idx(indMaxCon) = -1;
        %toc
    end
    
    if param.method == isRand
        %tic;
        tmpPartitionSize = floor(a/nCluster);
        res = a - nCluster*tmpPartitionSize;
        if res == 0
            PartitionSize = ones(nCluster,1)*tmpPartitionSize;
        else
            PartitionSize = ones(nCluster,1)*tmpPartitionSize;
            for i=1:res
                PartitionSize(i) = PartitionSize(i)+1;
            end
        end
        aPerm = randperm(a);
        idx = zeros(a,1);
        for i = 1:nCluster
            idx(aPerm(1:PartitionSize(i))) = i;
            aPerm(1:PartitionSize(i)) = [];
        end
        C_index = unique(idx);
        idx(indMaxCon) = -1;
        %toc
    end
    
    if param.method == isTIP
        C_index = [1:a];
        idx = C_index;
        C_index(indMaxCon) = [];
        idx(indMaxCon) = -1;
    end
    
    %display(idx);
    %fprintf('size of idx : %d, N : %d\n', length(idx), N);
    
    ClusterSize = []; % storing the size of each cluster
%     fprintf('cluster number: %d, method:%d\n',nCluster,param.method);
    for i=1:length(C_index)
        tmpClusterPosition = find(idx==C_index(i) | idx == -1); % find the indices of graphs that fall into i cluster
        % tmpClusterPosition = [tmpClusterPosition; N+1]; % combine new graph to current cluster, this is the index of ith cluster in all graphs
        ClusterPosition{i} = tmpClusterPosition;
        iClusterSize = length(tmpClusterPosition); % size of each cluster 
        ClusterSize = [ClusterSize iClusterSize]; % sizes of all clusters
        cropMat = cell(iClusterSize,iClusterSize); % the cropped rawMat for cluster i, +1 as new graph
        for j=1:iClusterSize
            for k=1:iClusterSize
                cropMat{j,k}=rawMatCell{tmpClusterPosition(j),tmpClusterPosition(k)}; % crop the block matrix of initial matching
            end
        end
        %%%%%%%%%%%%%%%% crop the affinity %%%%%%%%%%%%%%%%%%%%%
        affinityCrop = cropAffinity(tmpClusterPosition);
        %%%%%%%%%%%%%%%% crop the target %%%%%%%%%%%%%%%%%%%%%%%
        targetCrop = cropTarget(tmpClusterPosition);
        
        tmpRawMat = cell2mat(cropMat);
        affScoreCurrent = affScore(tmpClusterPosition,tmpClusterPosition);
        scrDenom = max(max(affScoreCurrent(1:end,1:end)));
        iXmatching = CAO_local(tmpRawMat,n,iClusterSize, scrDenom,affinityCrop, targetCrop,'pair',1); % perform CAO_pc
%         iXmatching = CAO_Floyd(tmpRawMat,n,iClusterSize, scrDenom,affinityCrop, targetCrop,'pair',1);
        iDivide = ones([1 iClusterSize])*n;
        tmpMatching{i}=mat2cell(iXmatching,iDivide, iDivide);
    end
    Matching=cell(N+1,N+1);
    for i=1:length(C_index) % This loop copies the existing matchings
        tmp = tmpMatching{i};
        tmpClusterPosition = ClusterPosition{i};
        for j=1:ClusterSize(i)
            for k=1:ClusterSize(i)
                Matching{tmpClusterPosition(j),tmpClusterPosition(k)}=tmp{j,k}; % copy the sub-matrix to final matching matrix
            end
        end
    end
    for i = 1:N+1 % This for loop generates the un-linked matchings
        for j = 1:N+1
            if idx(i)~=idx(j) && i~=j && idx(i)~=-1 && idx(j)~=-1
                Matching{i,j}=Matching{i,indMaxCon}*Matching{indMaxCon,j}; % generate matching through new graph
            end
        end
    end
    M = cell2mat(Matching);
end