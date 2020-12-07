clear *; clear -global *; clear all; clc; close all;
global affinity dataset args

args = setArgs;
dataset = setDataset(args);
algpar = setPairwiseSolver(args);
% mpmAlgPar = setMPMAlgPar(args);

testCnt = args.test_cnt;
nodeCnt = dataset.config.nodeCnt;
graphCnt = dataset.config.graphCnt;

% Parameter for algorithms
nMethods = 6;
algSet.algNameSet = {'RRWM','Spectral','MatchLift','MatchALS','CAO_PC','Floyd_PC'};
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
    
    for i = 4:4:graphCnt        
        scrDenomCurrent = max(max(scrDenomMatInCnt(1:i,1:i)));
        dim = ones(i, 1) * nodeCnt;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%RRWM%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rawMatTmp = rawMat(1:end-nodeCnt*(graphCnt - i),1:end-nodeCnt*(graphCnt - i));        
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,rawMatTmp,1,i,iTest,pairEnd,i);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%Spectral%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rawMatTmp = rawMat(1:end-nodeCnt*(graphCnt - i),1:end-nodeCnt*(graphCnt - i));        
        spectralStart = tic;
        spectralMat = Spectral(rawMatTmp,dim,nodeCnt);
        spectralEnd = toc(spectralStart);
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,spectralMat,2,i,iTest,spectralEnd+pairEnd,i);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%MatchLift%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rawMatTmp = rawMat(1:end-nodeCnt*(graphCnt - i),1:end-nodeCnt*(graphCnt - i));          
        liftStart = tic;
        liftMat = MatchLift(rawMatTmp,dim,nodeCnt);
        liftEnd = toc(liftStart);
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,liftMat,3,i,iTest,pairEnd+liftEnd,i);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%MatchALS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rawMatTmp = rawMat(1:end-nodeCnt*(graphCnt - i),1:end-nodeCnt*(graphCnt - i));          
        alsStart = tic;
        alsMat = MatchALS(rawMatTmp,dim,'univsize',nodeCnt,'pSelect',1,'tol',5e-4,'beta',0);
        alsEnd = toc(alsStart);        
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,alsMat,4,i,iTest,pairEnd+alsEnd,i);       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%CAO-pc%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rawMatTmp = rawMat(1:end-nodeCnt*(graphCnt - i),1:end-nodeCnt*(graphCnt - i));  
        caoStart = tic;
        caoMat = CAO(rawMatTmp,nodeCnt,i,scrDenomCurrent,'pair',1);
        caoEnd = toc(caoStart);
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,caoMat,5,i,iTest,caoEnd+pairEnd,i);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%Floyd-pc%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rawMatTmp = rawMat(1:end-nodeCnt*(graphCnt - i),1:end-nodeCnt*(graphCnt - i));        
        floydStart = tic;
        floydMat = CAO_Floyd(rawMatTmp,nodeCnt,i,scrDenomCurrent,affinity,dataset,'pair',1,0.3);
        floydEnd = toc(floydStart);
        [accResult,scrResult,conResult,timResult] = computeResult(accResult,scrResult,conResult,timResult,floydMat,6,i,iTest,floydEnd+pairEnd,i); 
        
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
for i = 4:4:graphCnt
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