function pairConst = cal_single_pair_consistency(X,Xij,i,j,nodeCnt,graphCnt,massOutlierMode,inlierMask)
% ����Xij���ʹ�X��������consistency
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
    if k==i || k==j, continue;end%ֱ�Ӻ��ԣ���Ϊerr��������
    ks = (k-1)*nodeCnt+1:k*nodeCnt;
    % Xij=Xik*Xkj
    aggX = X(is,ks)*X(ks,js);
%     errWeight = sqrt((1-outlierScore(:,k)).*(1-outlierScore(:,i)));
%     errX = errX + sum(errWeight.*sum(abs(Xij - aggX),2)/(2*sum(errWeight)));%����2����Ϊ������������ᱻ�Ŵ�2��
    if massOutlierMode
        errX = errX + sum(abs(Xij(:) - aggX(:)).*b);
    else
%         errX = errX + sum(sum(abs(Xij - aggX),2)/(2*nodeCnt));%��Ҫ��֮ǰ��sqrt�����������ǳ��࣡����Ҳ����sparse�����������
        errX = errX + sum(abs(Xij(:) - aggX(:)));
    end
end
if massOutlierMode
    pairConst = 1-errX/(graphCnt*2*estInCnt);
else
    pairConst = 1-errX/(graphCnt*2*nodeCnt);
end