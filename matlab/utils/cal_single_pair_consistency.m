function pairConst = cal_single_pair_consistency(X,Xij,i,j,nodeCnt,graphCnt,massOutlierMode,inlierMask)
% 给定Xij，和大X，计算其consistency
% \sum_k=1^N ||Pij-PikPkj||_F
% global inlierMask
if nargin<7
    massOutlierMode = 0;
end
is = (i-1)*nodeCnt+1:i*nodeCnt;
js = (j-1)*nodeCnt+1:j*nodeCnt;
errX = 0;
if massOutlierMode
    estInCnt = sum(inlierMask(:,1));
%     b = zeros(nodeCnt^2,graphCnt);
%     for i=1:graphCnt
%         b(:,i) = mat2vec(repmat(nodeConsistencyMask(:,i)',nodeCnt,1));
%     end
    b = mat2vec(repmat(inlierMask(:,i)',nodeCnt,1));
end
for k=1:graphCnt
    if k==i || k==j, continue;end%直接忽略，因为err不会增加
    ks = (k-1)*nodeCnt+1:k*nodeCnt;
    % Xij=Xik*Xkj
    aggX = X(is,ks)*X(ks,js);
%     errWeight = sqrt((1-outlierScore(:,k)).*(1-outlierScore(:,i)));
%     errX = errX + sum(errWeight.*sum(abs(Xij - aggX),2)/(2*sum(errWeight)));%除以2是因为排列阵相减误差会被放大2倍
    if massOutlierMode
        errX = errX + sum(abs(Xij(:) - aggX(:)).*b);
    else
%         errX = errX + sum(sum(abs(Xij - aggX),2)/(2*nodeCnt));%不要用之前的sqrt操作，会慢非常多！！！也别用sparse，反而会更慢
        errX = errX + sum(abs(Xij(:) - aggX(:)));
    end
end
if massOutlierMode
    pairConst = 1-errX/(graphCnt*2*estInCnt);
else
    pairConst = 1-errX/(graphCnt*2*nodeCnt);
end