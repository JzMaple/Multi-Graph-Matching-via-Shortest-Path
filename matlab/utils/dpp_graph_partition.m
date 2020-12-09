% To partition hyper-graph into n clusters using dpp sampling
% p: similarity/gram matrix
% n: number of the partitions

function [Idx, Partition] = dpp_graph_partition(p,n)
    [nSize, ~]=size(p);
    tmpSim = p;
    similarity = decompose_kernel(p);
    tmpPartitionSize = floor(nSize/n);
    res = nSize - n*tmpPartitionSize;
    if res == 0
        PartitionSize = ones(n,1)*tmpPartitionSize;
    else
        PartitionSize = ones(n,1)*tmpPartitionSize;
        for i=1:res
            PartitionSize(i) = PartitionSize(i)+1;
        end
    end
    tmpIdx = [1:nSize];
    for i=1:n
        if i==n
            Partition{n}=tmpIdx;
            break;
        end
        currentIdx=sample_dpp(similarity,PartitionSize(i));
        Partition{i} = tmpIdx(currentIdx);
        tmpIdx(currentIdx)=[];
        tmpSim(currentIdx,:)=[]; tmpSim(:,currentIdx)=[];
        similarity = decompose_kernel(tmpSim);
    end
    Idx = zeros(nSize,1);
    for i=1:n
        Idx(Partition{i})=i;
    end
end