function P = CAO_SPFA(rawMat,nodeCnt,graphCnt,scrDenom,affinity, target, optType, useCstInlier, x, k, c)
    inlierMask =  zeros(nodeCnt,graphCnt);
    if strcmp(target.config.testType,'massOutlier'),massOutlierMode = 1;else,massOutlierMode = 0;end
%     constStep = target.config.constStep;
%     initConstWeight = target.config.initConstWeight;% initial consistency regularizer weight, e.g 0.2-0.25
%     constWeightMax = target.config.constWeightMax;% the upperbound, always set to 1
    if massOutlierMode    
        if useCstInlier
            inlierMask = cal_node_consistency_mask(rawMat,nodeCnt,graphCnt,inCnt);
        else
            inlierMask = cal_node_affinity_mask(rawMat,nodeCnt,graphCnt,inCnt);
        end
    end
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    
    matchMat = rawMat;
    constWeight = c; 
    [X,Y] = meshgrid(1:graphCnt,1:graphCnt);
    X = X(:);Y = Y(:);
    l = 1;
    r = graphCnt-1;
    scope = (x-1)*nodeCnt+1:x*nodeCnt;
    node = [1:x-1,x+1:graphCnt];
    b = ones(graphCnt,1);
    
    unaryListConsistency = cal_single_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
    pairListConsistency = cal_pair_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
    
    cnt = 0;
    while l <= r
        b(node(l))=0;
        for i = 1:graphCnt
            if i==x || i==node(l) , continue; end
            iscope = (i-1)*nodeCnt+1:i*nodeCnt;
            [replace, matchMat(scope,iscope)] = update(matchMat,x,i,node(l),affinity,unaryListConsistency,pairListConsistency,nodeCnt,constWeight,scrDenom,optType);
            matchMat(iscope,scope) = matchMat(scope,iscope)';
            if replace && b(i)==0
                r = r+1; node(r) = i; b(i)=1; cnt = cnt + 1;
            end
        end
        l = l+1;
        if mod(l,k*graphCnt) == 0
            pairListConsistency = cal_pair_part_consistency(pairListConsistency,x,matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
        end        
        if l > graphCnt^2
            l = r+1;
        end
    end
%     fprintf('%d %d\n',cnt,graphCnt^2);
    unaryListConsistency = cal_single_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
    pairListConsistency = cal_pair_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
    
    for vk = 1:graphCnt^2
        if X(vk) >= Y(vk), continue; end
        xview = X(vk); yview = Y(vk);
        xscope = (xview-1)*nodeCnt+1:xview*nodeCnt;
        yscope = (yview-1)*nodeCnt+1:yview*nodeCnt;
        [~, matchMat(xscope,yscope)] = update(matchMat,xview,yview,x,affinity,unaryListConsistency,pairListConsistency,nodeCnt,constWeight,scrDenom,optType);
        matchMat(yscope,xscope) = matchMat(xscope,yscope)';
    end
    P = matchMat;
end

function [replace, P] = update(matchMat,xview,yview,kview,affinity,unaryListConsistency,pairListConsistency,nodeCnt,constWeight,scrDenom,metricType)
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
        unaCon2 = unaryListConsistency(kview);
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
