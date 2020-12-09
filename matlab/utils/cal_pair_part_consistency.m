function cstAdj = cal_pair_part_consistency(preCst,node,X,nodeCnt,graphCnt,massOutlierMode,inlierMask)
% \sum_k=1^N ||Pij-PikPkj||_F
if nargin<4
    massOutlierMode = 0;
end
% estOutCnt = nodeCnt - estInCnt;
if massOutlierMode
    estInCnt = sum(inlierMask(:,1));
    b = zeros(nodeCnt^2,graphCnt);
    for i=1:graphCnt
        b(:,i) = mat2vec(repmat(inlierMask(:,i)',nodeCnt,1));
    end
end
cstAdj = zeros(graphCnt,graphCnt);
for x = 1:graphCnt
    xscope = (x-1)*nodeCnt+1:x*nodeCnt;
    for y = x+1:graphCnt
        yscope = (y-1)*nodeCnt+1:y*nodeCnt;
        if y ~= node, cstAdj(x,y) = preCst(x,y); continue; end
        Xij = X(xscope,yscope);
        err = 0; 
        for k=1:graphCnt
            if k==x || k==y, continue;end
            kscope = (k-1)*nodeCnt+1:k*nodeCnt;
            % Xij=Xik*Xkj
            aggX = X(xscope,kscope)*X(kscope,yscope);
            if massOutlierMode
%             err = err + sum(sum(abs(Xij - aggX),2)/(2*nodeCnt));
                err = err + sum(abs(Xij(:) - aggX(:)).*b(:,x));
            else
                err = err + sum(abs(Xij(:) - aggX(:)));
            end
        end
%         cstAdj(x,y) = 1-err/graphCnt;
        if massOutlierMode
            cstAdj(x,y) = 1-err/(2*graphCnt*estInCnt);
        else
            cstAdj(x,y) = 1-err/(2*graphCnt*nodeCnt);
        end
    end
end
cstAdj = cstAdj + cstAdj'+eye(graphCnt);