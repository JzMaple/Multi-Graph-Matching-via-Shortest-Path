function acc = cal_acc(P,nOutlier,GT)
%P: Nx*Ny, GT: Nx*Ny
% nOutlier = 0;
[Nx Ny]=size(P);
if nargin<3
    GT = zeros(Nx,Ny);
    GT(:,1:Nx) = diag(ones(1,Nx));
end
if Ny==Nx%两侧有outlier
    P = P(1:end-nOutlier,1:end-nOutlier);%只看Nx 内点，忽略外点
    Nx = size(P,1);
    GT = GT(1:end-nOutlier,1:end-nOutlier);
end
% acc = (Nx-sum(sum(abs(P-GT)/2)))/Nx;
acc = sum(sum(abs(P-GT),2)==0)/Nx;%对每行求和,只有每行所有元素都相同，才认为匹配准确，防止有外点的情况下，P(1:nInlier)里没有1的情况