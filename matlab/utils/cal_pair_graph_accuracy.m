function accPair = cal_pair_graph_accuracy(X,GT,nOutlier,nodeCnt,graphCnt)
accPair = zeros(graphCnt,graphCnt);
 for viewx=1:graphCnt
    xscope = (viewx-1)*nodeCnt+1:viewx*nodeCnt;
    for viewy = viewx+1:graphCnt        
        yscope = (viewy-1)*nodeCnt+1:viewy*nodeCnt;
        accPair(viewx,viewy) = cal_acc(X(xscope,yscope),nOutlier,GT(xscope,yscope));
    end
 end
 accPair = accPair + accPair' + eye(graphCnt);