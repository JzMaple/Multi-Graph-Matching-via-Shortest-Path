function [score rawScore gtScore]= cal_score(P, K, GT)
% P: (Nx*Ny) * 1
% GT: (Nx*Ny) * 1
if nargin<3
    n = int32(sqrt(size(P)));
    GT = eye(n(1),n(1));
    GT = mat2vec(GT);
end
P = round(P);
% n = int32(sqrt(size(GT)));
% P = mat2vec(eye(n(1),n(1)));

rawScore = P'*K*P;
gtScore = GT'*K*GT;
score = rawScore/gtScore;
