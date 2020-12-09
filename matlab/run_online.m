clear *; clear -global *; clear all; clc; close all;
global affinity dataset args

args = setOnlineArgs;
dataset = setDataset(args);
algpar = setPairwiseSolver(args);
% mpmAlgPar = setMPMAlgPar(args);

testCnt = args.test_cnt;
nodeCnt = dataset.config.nodeCnt;
graphCnt = dataset.config.graphCnt;

% Parameter for algorithms
nMethods = 6;
algSet.algNameSet = {'IMGM-D','IMGM-R','CAO-PC','Floyd-PC','MGM-SPFA', 'FastSPFA'};
accResult = zeros(nMethods,graphCnt,testCnt);
conResult = zeros(nMethods,graphCnt,testCnt);
scrResult = zeros(nMethods,graphCnt,testCnt);
timResult = zeros(nMethods,graphCnt,testCnt);

for iTest = 1:testCnt %for each indepent test
    permutation = randperm(dataset.config.totalCnt);
    switch dataset.config.datasetType
        case 'sync'
            affinity = generateRandomAffinity(nInlier,iTest);
        case 'real'
            affinity = generateAffinity(iTest,permutation);
    end
    
    switch dataset.config.inCntType
        case 'exact'
            dataset.config.inCnt = nodeCnt - nOutlier;
        case 'all'
            dataset.config.inCnt = nodeCnt;
        case 'spec'
            dataset.config.inCnt = specNodeCnt;
    end
    
    % rrwm pairwise match, once for all graph pairs
    pairStart = tic;
    rawMat = generatePairAssignment(algpar,nodeCnt,graphCnt,iTest);     %generate matchings by pairwise matching solver
    pairEnd = toc(pairStart);
    scrDenomMatInCnt = cal_pair_graph_inlier_score(rawMat,affinity.GT,nodeCnt,graphCnt,dataset.config.inCnt);
    scrDenomMatInCntGT = cal_pair_graph_inlier_score(affinity.GT,affinity.GT,nodeCnt,graphCnt,dataset.config.inCnt);
   
    fprintf('--------------------------------------------------------------test %02d performance-------------------------------------------------------------------\n',iTest);
    for algk=1:nMethods
        fprintf('%25s',algSet.algNameSet{algk});
    end
    fprintf('\ngrh# itr#  ');
    for i = 1:nMethods
        fprintf(' acc   scr   con   tim   ');
    end
    
    baseGraphCnt = args.baseGraphCnt;
    scrDenomCurrent = max(max(scrDenomMatInCnt(1:baseGraphCnt,1:baseGraphCnt)));
    baseMat = rawMat(1:nodeCnt*baseGraphCnt,1:nodeCnt*baseGraphCnt);
    baseCAOMat = CAO_local(baseMat,nodeCnt, baseGraphCnt,scrDenomCurrent,affinity,dataset,'pair',1);
    baseFloydMat = MGM_Floyd(baseMat,nodeCnt,baseGraphCnt,scrDenomCurrent,affinity,dataset,'pair',1, args.floydconfig.alpha);    
    imgmrBase = baseCAOMat; imgmdBase = baseCAOMat; caoBase = baseCAOMat;
    spfaBase = baseFloydMat; fspfaBase = baseFloydMat; floydBase = baseFloydMat;
    
    for i = args.baseGraphCnt+1:graphCnt        
        scrDenomCurrent = max(max(scrDenomMatInCnt(1:i,1:i)));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%CAO-PC%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        caoMat = rawMat(1:nodeCnt*i,1:nodeCnt*i); 
        caoMat(1:end-nodeCnt,1:end-nodeCnt) = caoBase;    
        caoStart = tic;
        caoMat = CAO(caoMat,nodeCnt,i,scrDenomCurrent,'pair',1);
        caoEnd = toc(caoStart);   
        caoBase = caoMat;
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,caoMat,3,i,iTest,caoEnd,i);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%MGM-Floyd%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        flodMat = rawMat(1:nodeCnt*i,1:nodeCnt*i); 
        flodMat(1:end-nodeCnt,1:end-nodeCnt) = floydBase;       
        floydStart = tic;
        floydMat = MGM_Floyd(flodMat,nodeCnt,i,scrDenomCurrent,affinity,dataset,'pair',1,args.floydconfig.alpha);
        floydEnd = toc(floydStart);
        floydBase = floydMat;
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,floydMat,4,i,iTest,floydEnd,i);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%IMGM-D%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        args.imgmconfig.method = 1;
        args.imgmconfig.N = i - 1;
        imgmdMat = rawMat(1:nodeCnt*i,1:nodeCnt*i); 
        imgmdMat(1:end-nodeCnt,1:end-nodeCnt) = imgmdBase; 
        
        imgmStart = tic;        
        sigma = 0;
        scrDenomMatInCntDPPTmp = cal_pair_graph_inlier_score(imgmdMat,affinity.GT(1:nodeCnt*i,1:nodeCnt*i),nodeCnt,i,nodeCnt);
        conDenomMatInCntDPPTmp = cal_pair_graph_consistency(imgmdMat,nodeCnt,i,0);
        simDPP = (1-sigma)*scrDenomMatInCntDPPTmp + sigma*conDenomMatInCntDPPTmp;
        
        imgmdMat = IMGM(simDPP, imgmdMat, args.imgmconfig);
        imgmEnd = toc(imgmStart);
        imgmdBase = imgmdMat;
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,imgmdMat,1,i,iTest,imgmEnd,i);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%IMGM-R%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        args.imgmconfig.method = 3;
        args.imgmconfig.N = i - 1;
        imgmrMat = rawMat(1:nodeCnt*i,1:nodeCnt*i);
        imgmrMat(1:end-nodeCnt,1:end-nodeCnt) = imgmrBase; 
        
        imgmStart = tic;        
        sigma = 0;
        scrDenomMatInCntRndTmp = cal_pair_graph_inlier_score(imgmrMat,affinity.GT(1:nodeCnt*i,1:nodeCnt*i),nodeCnt,i,nodeCnt);
        conDenomMatInCntRndTmp = cal_pair_graph_consistency(imgmrMat,nodeCnt,i,0);
        simDPP = (1-sigma)*scrDenomMatInCntRndTmp + sigma*conDenomMatInCntRndTmp;
        
        imgmrMat = IMGM(simDPP, imgmrMat, args.imgmconfig);
        imgmEnd = toc(imgmStart);
        imgmrBase = imgmrMat;
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,imgmrMat,2,i,iTest,imgmEnd,i);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%MGM-SPFA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        spfaMat = rawMat(1:nodeCnt*i,1:nodeCnt*i);
        spfaMat(1:end-nodeCnt,1:end-nodeCnt) = spfaBase; 
        spfaStart = tic;
        spfaMat = CAO_SPFA(spfaMat, nodeCnt, i, scrDenomCurrent, affinity, dataset, 'pair', 1, i, 2, 0.3);
        spfaEnd = toc(spfaStart);
        spfaBase = spfaMat;
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,spfaMat,5,i,iTest,spfaEnd,i);   
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%FastSPFA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        fspfaMat = rawMat(1:nodeCnt*i,1:nodeCnt*i);
        fspfaMat(1:end-nodeCnt,1:end-nodeCnt) = fspfaBase; 
        
        imgmStart = tic;        
        sigma = 0;
        scrDenomMatInCntSPFATmp = cal_pair_graph_inlier_score(fspfaMat,affinity.GT(1:nodeCnt*i,1:nodeCnt*i),nodeCnt,i,nodeCnt);
        conDenomMatInCntSPFATmp = cal_pair_graph_consistency(fspfaMat,nodeCnt,i,0);
        simAP = (1-sigma)*scrDenomMatInCntSPFATmp + sigma*conDenomMatInCntSPFATmp;
        
        fspfaMat = FastSPFA(simAP, fspfaMat, nodeCnt, i, scrDenomCurrent, affinity, dataset, 'pair');
        imgmEnd = toc(imgmStart);
        fspfaBase = fspfaMat;
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,fspfaMat,6,i,iTest,imgmEnd,i);    
        
        fprintf('\n %02d,  %02d ',i,iTest);
        for algk=1:nMethods
            fprintf('| %.3f %.3f %.3f %.3f',accResult(algk,i,iTest),scrResult(algk,i,iTest),conResult(algk,i,iTest),timResult(algk,i,iTest));
        end
    end    
    T = clock;
    fprintf('\ntime: %d : %d : %f\n', T(4), T(5), T(6));
end

fprintf('--------------------------------------------------------------overall performance-------------------------------------------------------------------\n');
for algk=1:nMethods
    fprintf('%25s',algSet.algNameSet{algk});
end
fprintf('\n grh# itr#  ');
for i = 1:nMethods
    fprintf(' acc   scr   con   tim   ');
end
fprintf('\n');
for i = args.baseGraphCnt+1:graphCnt
    fprintf(' %02d,  all ',i);
    for algk=1:nMethods
        acc = mean(accResult(algk,i,:));
        scr = mean(scrResult(algk,i,:));
        con = mean(conResult(algk,i,:));
        tim = mean(timResult(algk,i,:));
        fprintf('| %.3f %.3f %.3f %.3f',acc,scr,con,tim);
    end
    fprintf('\n');
end