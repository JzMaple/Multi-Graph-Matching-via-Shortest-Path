%% Implementaion of PageRank matching algortihm
function [ X ] = RRWM( M, nP1, nP2, algpar)
% MATLAB demo code of Reweighted Random Walks Graph Matching of ECCV 2010
%
% Minsu Cho, Jungmin Lee, and Kyoung Mu Lee, 
% Reweighted Random Walks for Graph Matching, 
% Proc. European Conference on Computer Vision (ECCV), 2010
% http://cv.snu.ac.kr/research/~RRWM/
%
% Please cite our work if you find this code useful in your research. 
%
% written by Minsu Cho & Jungmin Lee, 2010, Seoul National University, Korea
% http://cv.snu.ac.kr/~minsucho/
% http://cv.snu.ac.kr/~jungminlee/
%
%
% Date: 01/11/2011
% Version: 1.2
%
% * Update report
%  This code is updated to be easily applied to any affinity matrix of candidate matches. 
%  For image matching demo, combine this with some additional codes and data provided in our site.
% ==================================================================================================
%
% input
%       M: affinity matrix
%       group1: conflicting match groups in domain 1 (size(M,1) x nGroup1)
%       group2: conflicting match groups in domain 2 (size(M,1) x nGroup2)
%                 
%       e.g. find(group1(:,3)) represents the third goup of matches  
%                               sharing the same feature in domain1   
%
% output
%       X: steady state distribution of RRWM

%% Default Parameters
param = struct( ...
    'c', 0.2, ...                   % prob. for walk or reweighted jump?
    'amp_max', 30, ...              % maximum value for amplification procedure
    'thresConvergence', 1e-25, ...  % convergence threshold for random walks
    'tolC', 1e-3 ...                % convergence threshold for the Sinkhorn method
);
E12 = ones(nP1,nP2);
[L12(:,1) L12(:,2)] = find(E12);
[group1 group2] = make_group12(L12);

%% parameter structure -> parameter value
strField = fieldnames(param);
for i = 1:length(strField), eval([strField{i} '=param.' strField{i} ';']); end
% % % % % % % % % % % % % % % % % % % % 
% get groups for bistochastic normalization
[idx1 ID1] = make_groups(group1);
[idx2 ID2] = make_groups(group2);
% 
if ID1(end) < ID2(end)
    [idx1 ID1 idx2 ID2 dumVal dumSize] = make_groups_slack(idx1, ID1, idx2, ID2);
    dumDim = 1;
elseif ID1(end) > ID2(end)
    [idx2 ID2 idx1 ID1 dumVal dumSize] = make_groups_slack(idx2, ID2, idx1, ID1);
    dumDim = 2;
else
    dumDim = 0; dumVal = 0; dumSize = 0;
end
idx1 = idx1-1; idx2 = idx2-1;

% eliminate conflicting elements to prevent conflicting walks
% conf1 = zeros(size(M));
% conf2 = zeros(size(M));
% for i = 1:size(group1,2)
%     idx = find(group1(:,i));
%     for j = 1:length(idx)
%         for k = 1:length(idx)
%             conf1(idx(j),idx(k)) = 1;
%         end
%     end
% end
% for i = 1:size(group2,2)
%     idx = find(group2(:,i));
%     for j = 1:length(idx)
%         for k = 1:length(idx)
%             conf1(idx(j),idx(k)) = 1;
%         end
%     end
% end
% conf = conf1 | conf2;
% M = M.*(~conf);
% % % % % % % % % % % % % % % % % % % % % % % 
% note that this matrix is column-wise stochastic
d = sum(M, 1); % degree : col sum
maxD = max(d);
Mo = M ./ maxD; % nomalize by the max degree

% initialize answer
nMatch = length(M);
prev_score = ones(nMatch,1)/nMatch; % buffer for the previous score
prev_score2 = prev_score;         % buffer for the two iteration ahead
prev_assign = ones(nMatch,1)/nMatch; % buffer for Sinkhorn result of prev_score
% prev_assign = rand(nMatch,1);
% prev_assign = BregmanBiStoch(reshape(prev_assign,[40 40]),50);
% prev_assign = prev_assign(:);

bCont = 1;  iter_i = 0;

% for convergence check of power iteration
% thresConvergence2 = length(prev_score)*norm(M,1)*eps;
% la = prev_score'*M*prev_score;

%% start main iteration
while bCont && iter_i < algpar.iterMax1
    
    iter_i = iter_i + 1;
    
    %% random walking with reweighted jumps
    cur_score = Mo * ( c*prev_score + (1-c)*prev_assign );
    
    sumCurScore = sum(cur_score); % normalization of sum 1
    if sumCurScore>0, cur_score = cur_score./sumCurScore; end
    
    %% update reweighted jumps
    cur_assign = cur_score;
    % attenuate small values and amplify large values
    amp_value = amp_max/ max(cur_assign);  % compute amplification factor
    cur_assign = exp( amp_value*cur_assign );  % amplification 
    
    % Sinkhorn method of iterative bistocastic normalizations
    X_slack = [cur_assign; dumVal*ones(dumSize,1)];
    X_slack = mexBistocNormalize_match_slack(X_slack, int32(idx1), int32(ID1), int32(idx2), int32(ID2), tolC, dumDim, dumVal, int32(1000));
    cur_assign = X_slack(1:nMatch);   
    
    sumCurAssign = sum(cur_assign); % normalization of sum 1
    if sumCurAssign>0, cur_assign = cur_assign./sumCurAssign; end
    
    %% Check the convergence of random walks
    if 1
        diff1 = sum((cur_score-prev_score).^2);
        diff2 = sum((cur_score-prev_score2).^2); % to prevent oscillations
        diff_min = min(diff1, diff2);
        if diff_min < thresConvergence
            bCont = 0;
        end
    else
        normed_cur_score = cur_score/norm(cur_score);
        if norm(M*normed_cur_score - la*normed_cur_score,1) < thresConvergence2
            bCont = 0;
        end
        la = normed_cur_score'*M*normed_cur_score;
    end

    prev_score2 = prev_score;
    prev_score = cur_score;
    % cur_score'*Mo*cur_score
    prev_assign = cur_assign;
 
end % end of main iteration

X = cur_score;
end