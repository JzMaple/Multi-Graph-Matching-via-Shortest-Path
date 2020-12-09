% Input:
% rawMat is a matrix of size nodeCnt*graphCnt \times nodeCnt*graphCnt.
% rawMat contains the pairwise matchings for all graph pairs computed by a
% pairwise matching solver, such as RRWM.
% The values of optType include (refer to PAMI paper): 'unary' (CAO-UC),
% 'afnty' (CAO), 'pair' (CAO-PC), 'exact' (CAO-C).
% scrDenom is used to normalize affinity score, in order to be comparable
% to consistency metrics in [0,1].
%
% Output:
% P is the processed rawMat
function P = CAO_local(rawMat,nodeCnt,graphCnt,scrDenom,affinity,dataset,optType,useCstInlier)
    global matchMat %  inlierMask %used in the sub-calling function in this file 
    global args
%     useCstInlier = dataset.config.useCstInlier;%set to 1 if use consistency to mask inliers, otherwise use affinity metrics
    inCnt = dataset.config.inCnt;% the number of inliers, can be specified to different values by manual or automatically 
    inlierMask =  zeros(nodeCnt,graphCnt); % node-wise consistency for each node in each graph
    if strcmp(dataset.config.testType,'massOutlier'),massOutlierMode = 1;else,massOutlierMode = 0;end
    constIterImmune = args.caoconfig.constIterImmune;% immune from consistency regularization, only use affinity in earl
    curIdx = 1;lastIdx = 2;
    constStep = args.caoconfig.constStep;% the inflate parameter, e.g. 1.05-1.1
    initConstWeight = args.caoconfig.initConstWeight;% initial consistency regularizer weight, e.g 0.2-0.25
    constWeightMax = args.caoconfig.constWeightMax;% the upperbound, always set to 1
    iterMax = args.caoconfig.iterRange;
    if massOutlierMode % if in the presence of many outliers, need to compute mask to identify inliers
        % identified inliers' correspondences are kept, others (outliers) are zeroed, to make the score function more discriminative    
        if useCstInlier
            inlierMask = cal_node_consistency_mask(rawMat,nodeCnt,graphCnt,inCnt);
        else
            inlierMask = cal_node_affinity_mask(rawMat,nodeCnt,graphCnt,inCnt);
        end
    end
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    iter = 0;
    err = ones(iterMax,1);
    I = eye(nodeCnt*graphCnt,nodeCnt*graphCnt);
    matchMat = rawMat;
    constWeight = initConstWeight; % intialize weight (const means consistency rather than const)
    % Compute the unary consistency of each graph, see Definition (1) in the PAMI paper
    unaryListConsistency{curIdx} = cal_single_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
    unaryListConsistency{lastIdx} = unaryListConsistency{curIdx};
    % Compute the pairwise consistency of each graph pair, see Definition (2) in the PAMI paper
    pairListConsistency{curIdx} = cal_pair_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);
    pairListConsistency{lastIdx} = pairListConsistency{curIdx};

    [X,Y] = meshgrid(1:graphCnt,1:graphCnt);
    X = X(:);Y = Y(:);
    while iter<iterMax % composition driven iteration
        tempMat = zeros(nodeCnt*graphCnt,nodeCnt*graphCnt);
        for vk = 1:graphCnt^2% for each graph pair, update their matching by evaluating the fitness function as stated in Eq.7 in the PAMI paper
            xview = X(vk); yview = Y(vk);
            if mod(vk-1,graphCnt) == 0,xscope = (xview-1)*nodeCnt+1:xview*nodeCnt;end
            if X(vk)>= Y(vk),continue;end% only need to compute a half of pairs due to redudency
            yscope = (yview-1)*nodeCnt+1:yview*nodeCnt;	
            % different fitness function desgin based on setting optType and iteration stage
            if iter<constIterImmune&&(strcmp(optType,'exact')||strcmp(optType,'pair')||strcmp(optType,'unary'))% for CAO-C, CAO-PC, CAO-UC, in early immune stage, only affinity is boosted
                tempMat(xscope,yscope) = find1stOrderPathByScoreUnaryPairConsistency(...
                    xview,yview,affinity, unaryListConsistency{curIdx},pairListConsistency{curIdx},nodeCnt,graphCnt,constWeight,scrDenom,'afnty',massOutlierMode,inlierMask);
            else% CAO or its consistency version (replace affinity score by consistency CAO^{cst})
                tempMat(xscope,yscope) = find1stOrderPathByScoreUnaryPairConsistency(...
                    xview,yview,affinity, unaryListConsistency{curIdx},pairListConsistency{curIdx},nodeCnt,graphCnt,constWeight,scrDenom,optType,massOutlierMode,inlierMask);
            end
        end%for vk
        matchMatOld = matchMat;% keep the solution of last iteration for debug
        matchMat = tempMat+tempMat'+I;% fill the full solution matrix
        if iter>=constIterImmune% inflate the weight parameter when exceeds the immune stage
            constWeight = min([constWeightMax,constStep*constWeight]);
        else
            % Otherwise in the presence of massive outliers i.e. massOutlierMode is set to 1, node-wise consistency need be computed
            if massOutlierMode
                if useCstInlier% if use node-wise consistency to identify inliers, see Definition 4 in PAMI paper
                    inlierMask = cal_node_consistency_mask(matchMat,nodeCnt,graphCnt,inCnt);
                else% otherwise use node-wise affinity to identify inliers, see Definition 5 in PAMI paper
                    inlierMask = cal_node_affinity_mask(matchMat,nodeCnt,graphCnt,inCnt);
                end
            end
        end
        % update unary consistency for each graph(Definition 1) and pairwise consistency for each graph pair (Definition 2)
        unaryListConsistency{lastIdx} = unaryListConsistency{curIdx};
        unaryListConsistency{curIdx} = cal_single_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);%0.001秒的运行时间
        pairListConsistency{lastIdx} = pairListConsistency{curIdx};
        pairListConsistency{curIdx} = cal_pair_graph_consistency(matchMat,nodeCnt,graphCnt,massOutlierMode,inlierMask);

        iter = iter + 1;
        err(iter) = sum(abs(matchMatOld(:) - matchMat(:)));% check if converge 
        if err(iter)==0
            break;
        end
    end%for iter
    % return the final solution
    P = matchMat;

function P = find1stOrderPathByScoreUnaryPairConsistency(xview,yview,affinity, unaryListConsistency,pairListConsistency,...
    nodeCnt,graphCnt,constWeight,scrDenom,metricType,massOutlierMode,inlierMask)
    % global affinity %inlierMask
    global matchMat
    pairCon = zeros(graphCnt,1);% it is dynamically generated, shall not use the input pairListConsistency, see Eq.10 in the PAMI paper,
    xscope = (xview-1)*nodeCnt+1:xview*nodeCnt;% the xview-th graph
    yscope = (yview-1)*nodeCnt+1:yview*nodeCnt;% the yview-th graph
    Y = zeros(nodeCnt*nodeCnt,graphCnt);% anchor graph candidate set
    for anchor=1:graphCnt
        ascope = (anchor-1)*nodeCnt+1:anchor*nodeCnt;
        P1 = matchMat(xscope,ascope);P2 = matchMat(ascope,yscope);
        Y(:,anchor) = mat2vec(P1*P2);% store P_{xy} = P_{xa}*P_{ay}
        if strcmp(metricType,'pair')% see Eq.10 in the PAMI paper, compute the pairwise consistency
            pairCon(anchor) = sqrt(pairListConsistency(xview,anchor)*pairListConsistency(anchor,yview));
        end
    end
	% reduce the problem size by reducing duplicate generated solutions
    [~,m1,n1] = unique(Y','rows','first');
    uniLen = length(m1);
    uniCon = zeros(uniLen,1);uniAfty = zeros(uniLen,1);
    % only need to take care of unique solutions
    
    for i=1:uniLen
        a = m1(i);
        p = sparse(Y(:,a));
		% in massOutlierMode, mask the matching Y(:,a) to keep only identified inlier matchings, outliers matchings are set to 0
        if massOutlierMode,b=repmat(inlierMask(:,a)',nodeCnt,1);p = p.*b(:);end
        if ~strcmp(metricType,'cstcy')% consistency mode need no affinity score, and keep initial value 0
            uniAfty(i) = p'*affinity.K{xview,yview}*p/scrDenom;
        end
        % for CAO-C and CAO^{cst}, compute the exact consistency for the candidate solution Y(:,a)
        if strcmp(metricType,'exact')||strcmp(metricType,'cstcy')
			uniCon(i) = cal_single_pair_consistency(matchMat,vec2mat(Y(:,a),nodeCnt,nodeCnt),xview,yview,nodeCnt,graphCnt,massOutlierMode,inlierMask);
		end
    end
    % now repopulate the affinity score for all candidate middle graphs which were reduced by the unique function above
    afntyScr = uniAfty(n1);% affinity score for each middle graph
    if strcmp(metricType,'exact')||strcmp(metricType,'cstcy')
        exaCon = uniCon(n1);% exact (not unary or pariwse) consistency score for each middle graph
    end
    
    switch metricType
        case 'unary'% CAO-UC
            fitness = (1-constWeight)*afntyScr + constWeight*unaryListConsistency;
        case 'pair'% CAO-PC
            fitness = (1-constWeight)*afntyScr + constWeight*pairCon;
        case 'exact'% CAO-C
            fitness = (1-constWeight)*afntyScr + constWeight*exaCon;
        case 'afnty'% CAO
            fitness = afntyScr;
        case 'cstcy'% used in Fig.1 in PAMI paper for CAO^{cst}
            fitness = exaCon;
    end
	% return the maximum fitness solution	 
	[~, idx] = max(fitness);
	P = vec2mat(Y(:,idx(1)),nodeCnt,nodeCnt);
