function rawScorePair = cal_pair_graph_inlier_score(X,GT,nodeCnt,graphCnt,inCnt)
global affinity
% scorePair = zeros(graphCnt,graphCnt);
rawScorePair = zeros(graphCnt,graphCnt);
for viewx=1:graphCnt
    xscope = (viewx-1)*nodeCnt+1:(viewx-1)*nodeCnt+inCnt;
    for viewy = viewx+1:graphCnt
        yscope = (viewy-1)*nodeCnt+1:(viewy-1)*nodeCnt+inCnt;
        x = zeros(nodeCnt,nodeCnt);
        x(1:inCnt,1:inCnt) = X(xscope,yscope);
        gt = zeros(nodeCnt,nodeCnt);
        gt(1:inCnt,1:inCnt) = GT(xscope,yscope);
        [~, rawScorePair(viewx,viewy)] = cal_score(mat2vec(x),affinity.K{viewx,viewy},mat2vec(gt));
    end
end
rawScorePair = rawScorePair + rawScorePair';