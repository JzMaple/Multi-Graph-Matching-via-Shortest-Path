function cstAdj = cal_single_graph_consistency(X,nodeCnt,graphCnt,massOutlierMode,inlierMask)
% global inlierMask
if nargin<4
    massOutlierMode = 0;
end
% cstAdj is of size graphCnt \times 1, include the unary consitency for all graphs  
% inlierMask is of size nodeCnt \times graphCnt
if massOutlierMode
    estInCnt = sum(inlierMask(:,1));% total number of inliers, without loss of generality, use graph 1 is O.K.
    b = zeros(nodeCnt^2,graphCnt);% the actual mask
    for i=1:graphCnt
        b(:,i) = mat2vec(repmat(inlierMask(:,i)',nodeCnt,1));
    end
end
cstAdj = zeros(graphCnt,1);
for ref = 1:graphCnt %for each graph ref, compute its unary consistency by Definition (1) in the PAMI paper
    viewk = 1; err = zeros((graphCnt+1)*(graphCnt-2)/2,1);
    rscope = (ref-1)*nodeCnt+1:ref*nodeCnt;
    for i = 1:graphCnt
        iscope = (i-1)*nodeCnt+1:i*nodeCnt;
        for j = i+1:graphCnt
            jscope = (j-1)*nodeCnt+1:j*nodeCnt;
            Xirj=X(iscope,rscope)*X(rscope,jscope);
%             err(viewk) = sum(sum(abs(Xirj-X(iscope,jscope)),2))/(2*nodeCnt)-estOutCnt;
             if massOutlierMode
                 err(viewk) = sum(abs(Xirj(:) - mat2vec(X(iscope,jscope))).*b(:,i));
             else
                 err(viewk) = sum(sum(abs(Xirj-X(iscope,jscope)),2));
             end
            viewk = viewk + 1;
        end
    end 
    if massOutlierMode
        cstAdj(ref) = 1-mean(err)/(2*estInCnt);
    else
        cstAdj(ref) = 1-mean(err)/(2*nodeCnt);
    end
end
% ok = 1;