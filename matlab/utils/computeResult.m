function [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,matching,alg,i,iTest,End,graphCnt)
    global affinity dataset args

    acc = cal_pair_graph_accuracy(matching,affinity.GT,dataset.config.nOutlier,dataset.config.nodeCnt,graphCnt);
    scr = cal_pair_graph_score(matching,affinity.GT,dataset.config.nodeCnt,graphCnt);
    con = cal_pair_graph_consistency(matching,dataset.config.nodeCnt,graphCnt,0);
    
    if args.metric_eye == 0
        nPair = graphCnt * (graphCnt - 1);
        accResult(alg,i,iTest)=sum(acc - eye(graphCnt), 'all') / nPair;
        scrResult(alg,i,iTest)=sum(scr - eye(graphCnt), 'all') / nPair;
        conResult(alg,i,iTest)=sum(con - eye(graphCnt), 'all') / nPair;
        timResult(alg,i,iTest)=End;
    else
        accResult(alg,i,iTest)=mean(acc(:));
        scrResult(alg,i,iTest)=mean(scr(:));
        conResult(alg,i,iTest)=mean(con(:));
        timResult(alg,i,iTest)=End;
    end
end