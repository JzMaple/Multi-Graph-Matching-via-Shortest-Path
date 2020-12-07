function [scorePair rawScorePair] = cal_pair_graph_score(X,GT,nodeCnt,graphCnt)
global affinity
scorePair = zeros(graphCnt,graphCnt);
rawScorePair = zeros(graphCnt,graphCnt);
rawInlierScorePair = zeros(graphCnt,graphCnt);
for viewx=1:graphCnt
    xscope = (viewx-1)*nodeCnt+1:viewx*nodeCnt;
    for viewy = viewx+1:graphCnt
        yscope = (viewy-1)*nodeCnt+1:viewy*nodeCnt;
        [scorePair(viewx,viewy) rawScorePair(viewx,viewy) gt] = cal_score(mat2vec(X(xscope,yscope)),affinity.K{viewx,viewy},mat2vec(GT(xscope,yscope)));
        [~, rawInlierScorePair(viewx,viewy)] = cal_score(mat2vec(X(xscope,yscope)),affinity.K{viewx,viewy},mat2vec(GT(xscope,yscope)));
    end
end
scorePair = scorePair + scorePair' + eye(graphCnt);
rawScorePair = rawScorePair + rawScorePair' + eye(graphCnt);