function [P, mat_list] = CAO_Floyd(rawMat,nodeCnt,graphCnt,scrDenom,affinity,target,optType,useCstInlier,c) 
    inlierMask =  zeros(nodeCnt,graphCnt);
    if strcmp(target.config.testType,'massOutlier'),massOutlierMode = 1;else,massOutlierMode = 0;end
    if massOutlierMode
        if useCstInlier
            inlierMask = cal_node_consistency_mask(rawMat,nodeCnt,graphCnt,inCnt);
        else
            inlierMask = cal_node_affinity_mask(rawMat,nodeCnt,graphCnt,inCnt);
        end
    end
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    mat_list = cell(graphCnt+1, 1);
    matchMat = rawMat;    
    mat_list{1} = rawMat;
    constWeight = c;
    unaryListConsistency = cal_single_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
    pairListConsistency = cal_pair_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);      
    [X,Y] = meshgrid(1:graphCnt,1:graphCnt);
    X = X(:);Y = Y(:);
    for kview = 1:graphCnt
        for vk = 1:graphCnt^2
            xview = X(vk); yview = Y(vk);
            if X(vk) >= Y(vk), continue; end
            xscope = (xview-1)*nodeCnt+1:xview*nodeCnt;
            yscope = (yview-1)*nodeCnt+1:yview*nodeCnt;
            [replace, matchMat(xscope,yscope)] = update(matchMat,xview,yview,kview,affinity,unaryListConsistency,pairListConsistency,nodeCnt,graphCnt,constWeight,scrDenom,'afnty');
            if replace, matchMat(yscope,xscope) = matchMat(xscope,yscope)'; end
        end
    end
    for kview = 1:graphCnt
        for vk = 1:graphCnt^2
            xview = X(vk); yview = Y(vk);
            if X(vk) >= Y(vk), continue; end
            xscope = (xview-1)*nodeCnt+1:xview*nodeCnt;
            yscope = (yview-1)*nodeCnt+1:yview*nodeCnt;
            [replace, matchMat(xscope,yscope)] = update(matchMat,xview,yview,kview,affinity,unaryListConsistency,pairListConsistency,nodeCnt,graphCnt,constWeight,scrDenom,optType);
            if replace, matchMat(yscope,xscope) = matchMat(xscope,yscope)'; end
        end
        if strcmp(optType,'unary')            
            unaryListConsistency = cal_single_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
        elseif strcmp(optType,'pair')
            pairListConsistency = cal_pair_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
        end
        mat_list{kview+1} = matchMat;
    end
    P = matchMat;
end

function [replace, P] = update(matchMat,xview,yview,kview,affinity,unaryListConsistency,pairListConsistency,nodeCnt,graphCnt,constWeight,scrDenom,metricType)
    xscope = (xview-1)*nodeCnt+1:xview*nodeCnt;
    yscope = (yview-1)*nodeCnt+1:yview*nodeCnt;
    kscope = (kview-1)*nodeCnt+1:kview*nodeCnt;
    P1 = matchMat(xscope,kscope); P2 = matchMat(kscope,yscope); P3 = matchMat(xscope,yscope);
    Y1 = sparse(mat2vec(P3));
    Y2 = sparse(mat2vec(P1*P2));
    afntyScr1 = Y1'*affinity.K{xview,yview}*Y1/scrDenom;
    afntyScr2 = Y2'*affinity.K{xview,yview}*Y2/scrDenom;
    if strcmp(metricType,'pair')
        xyCon = pairListConsistency(xview,yview);
        pairCon1 = sqrt(xyCon);
        xkCon = pairListConsistency(xview,kview);
        kyCon = pairListConsistency(kview,kview);
        pairCon2 = sqrt(xkCon*kyCon);       
    elseif strcmp(metricType,'unary')
        xCon = unaryListConsistency(xview);
        yCon = unaryListConsistency(yview);
%         unaCon1 = min(xCon,yCon);
        unaCon1 = xCon;
        unaCon2 = unaryListConsistency(kview);
    elseif strcmp(metricType,'exact')
        extCon1 = cal_single_pair_consistency(matchMat,P3,xview,yview,nodeCnt,graphCnt,0,0);
        extCon2 = cal_single_pair_consistency(matchMat,P1*P2,xview,yview,nodeCnt,graphCnt,0,0);
    end
    switch metricType
        case 'unary'% CAO-UC
            res1 = (1-constWeight)*afntyScr1 + constWeight*unaCon1;
            res2 = (1-constWeight)*afntyScr2 + constWeight*unaCon2;
        case 'pair'% CAO-PC
            res1 = (1-constWeight)*afntyScr1 + constWeight*pairCon1;
            res2 = (1-constWeight)*afntyScr2 + constWeight*pairCon2;
        case 'exact'
            res1 = (1-constWeight)*afntyScr1 + constWeight*extCon1;
            res2 = (1-constWeight)*afntyScr2 + constWeight*extCon2;
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
