function M = FastSPFA(affScore,rawMat,nodeCnt,graphCnt,scrDenom,affinity,target,optType)
    inlierMask =  zeros(nodeCnt,graphCnt);
    if strcmp(target.config.testType,'massOutlier'),massOutlierMode = 1;else,massOutlierMode = 0;end
    curIdx = 1;lastIdx = 2;
    clusterCnt = max(floor(graphCnt / 15),1);
    clusterSize = floor(graphCnt / clusterCnt);
    cluster = cell(1,clusterCnt);
    temp = 1:graphCnt-1; cnt = graphCnt-1;
    position = zeros(1,graphCnt);
    for i = 1:clusterCnt-1
        tmpCluster = zeros(1,clusterSize);
        for j = 1:clusterSize
            id = ceil(rand()*cnt);           
            tmpCluster(j) = temp(id);            
            position(temp(id)) = i;
            temp = [temp(1:id-1) temp(id+1:end)];
            cnt = cnt - 1;
        end
        cluster{i} = [tmpCluster,graphCnt];
    end
    cluster{clusterCnt} = [temp,graphCnt];
    position(temp) = clusterCnt;
    
    matchMat = rawMat;
    for i = 1:clusterCnt
        tmpCluster = cluster{i};
        size = length(tmpCluster);
        matCrop = cropMatching(tmpCluster,matchMat,nodeCnt);        
        affinityCrop = cropAffinity(tmpCluster);
        targetCrop = cropTarget(tmpCluster);
        affScoreCrop = affScore(tmpCluster,tmpCluster);
        scrDenomCrop = max(max(affScoreCrop(1:end,1:end)));
        tempMat = CAO_SPFA(matCrop, nodeCnt, size, scrDenomCrop, affinityCrop, targetCrop, optType, 1, size, 1, 0.3);
        
        [X,Y] = meshgrid(1:size,1:size);
        X=X(:); Y=Y(:);
        for vk = 1:size^2
            iview = X(vk); jview = Y(vk);
            if iview>=jview, continue; end
            iscope = (iview-1)*nodeCnt+1:iview*nodeCnt;
            jscope = (jview-1)*nodeCnt+1:jview*nodeCnt;
            xview = tmpCluster(iview); yview = tmpCluster(jview);
            xscope = (xview-1)*nodeCnt+1:xview*nodeCnt;
            yscope = (yview-1)*nodeCnt+1:yview*nodeCnt;
            matchMat(xscope,yscope) = tempMat(iscope,jscope);
            matchMat(yscope,xscope) = tempMat(jscope,iscope);
        end 
    end
    
    unaryListConsistency{curIdx} = cal_single_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
    unaryListConsistency{lastIdx} = unaryListConsistency{curIdx};
    pairListConsistency{curIdx} = cal_pair_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
    pairListConsistency{lastIdx} = pairListConsistency{curIdx};
    
    constWeight = 0.4;
    [X,Y] = meshgrid(1:graphCnt,1:graphCnt);
    X=X(:); Y=Y(:);
    for vk = 1:graphCnt^2
        xview = X(vk); yview = Y(vk);
        if xview >= yview, continue; end
        if position(xview) == position(yview) || xview == graphCnt || yview == graphCnt, continue; end
        [~, matchMat(xscope,yscope)] = update(matchMat,xview,yview,graphCnt,affinity,unaryListConsistency{curIdx},pairListConsistency{curIdx},nodeCnt,graphCnt,constWeight,scrDenom,optType,massOutlierMode,inlierMask);
        matchMat(yscope,xscope) = matchMat(xscope,yscope)';
    end
    M = matchMat;
end

function M = cropMatching(tmpCluster,matchMat,nodeCnt)
    size = length(tmpCluster);
    matchingCrop = zeros(size*nodeCnt);
    [X,Y] = meshgrid(1:size,1:size);
    X=X(:); Y=Y(:);
    for vk = 1:size^2
        iview = X(vk); jview = Y(vk);
        if iview>=jview, continue; end
        iscope = (iview-1)*nodeCnt+1:iview*nodeCnt;
        jscope = (jview-1)*nodeCnt+1:jview*nodeCnt;
        xview = tmpCluster(iview); yview = tmpCluster(jview);
        xscope = (xview-1)*nodeCnt+1:xview*nodeCnt;
        yscope = (yview-1)*nodeCnt+1:yview*nodeCnt;
        matchingCrop(iscope,jscope) = matchMat(xscope,yscope);
    end
    matchingCrop = matchingCrop + matchingCrop' + eye(size*nodeCnt);
    M = matchingCrop;
end

function [replace, P] = update(matchMat,xview,yview,kview,affinity,unaryListConsistency,pairListConsistency,nodeCnt,graphCnt,constWeight,scrDenom,metricType,massOutlierMode,inlierMask)
    xscope = (xview-1)*nodeCnt+1:xview*nodeCnt;
    yscope = (yview-1)*nodeCnt+1:yview*nodeCnt;
    kscope = (kview-1)*nodeCnt+1:kview*nodeCnt;
    P1 = matchMat(xscope,kscope); P2 = matchMat(kscope,yscope); 
    Y1 = sparse(mat2vec(matchMat(xscope,yscope)));
    Y2 = sparse(mat2vec(P1*P2));
    afntyScr1 = Y1'*affinity.K{xview,yview}*Y1/scrDenom;
    afntyScr2 = Y2'*affinity.K{xview,yview}*Y2/scrDenom;
    if strcmp(metricType,'pair')
        xyCon = pairListConsistency(xview,yview);
        pairCon1 = sqrt(xyCon);
        xkCon = pairListConsistency(xview,kview);
        kyCon = pairListConsistency(kview,kview);
        pairCon2 = sqrt(xkCon*kyCon);       
    end
    if strcmp(metricType,'unary')
        a = unaryListConsistency(xview);
        b = unaryListConsistency(yview);
        unaCon1 = max(a,b);
        unaCon2 = cal_exact_single_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask,kscope);
    end
    switch metricType
        case 'unary'% CAO-UC
            res1 = (1-constWeight)*afntyScr1 + constWeight*unaCon1;
            res2 = (1-constWeight)*afntyScr2 + constWeight*unaCon2;
        case 'pair'% CAO-PC
            res1 = (1-constWeight)*afntyScr1 + constWeight*pairCon1;
            res2 = (1-constWeight)*afntyScr2 + constWeight*pairCon2;
        case 'exact'% CAO-C
            res1 = (1-constWeight)*afntyScr1 + constWeight*exaCon1;
            res2 = (1-constWeight)*afntyScr2 + constWeight*exaCon2;
        case 'afnty'% CAO
            res1 = afntyScr1;
            res2 = afntyScr2;
    end
    if res1 >= res2
    	P = vec2mat(Y1,nodeCnt,nodeCnt);
        replace = false;
    else
    	P = vec2mat(Y2,nodeCnt,nodeCnt);
        replace = true;
    end
end