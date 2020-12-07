function rawMat = generatePairAssignment(algpar,nodeCnt,graphCnt,testk)
global dataset
rawMat = zeros(nodeCnt*graphCnt,nodeCnt*graphCnt);
for i = 1:graphCnt-1
    iscope = (i-1)*nodeCnt+1:i*nodeCnt;
    for j = i+1:graphCnt
        jscope = (j-1)*nodeCnt+1:j*nodeCnt;
        mask = dataset.pairwiseMask{testk}(iscope,jscope);
        if max(mask(:))>0
            rawMat(iscope,jscope) = pairMatch(i,j,algpar);
        else
            rawMat(iscope,jscope) = generateRandomPermute(nodeCnt);
        end
    end
end
rawMat = rawMat + rawMat' + eye(nodeCnt*graphCnt,nodeCnt*graphCnt);