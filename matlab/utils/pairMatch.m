function [rawP rawScore]= pairMatch(viewx,viewy,algpar)
global affinity
if strcmpi(algpar.algMethod,'FGM')% not supported in the demo code
    P = pairFGM(algpar,viewx,viewy);
end
if strcmpi(algpar.algMethod,'RRWM')
    nP1 = affinity.nP{viewx};
    nP2 = affinity.nP{viewy};
    E12 = ones(nP1,nP2);
    n12 = nnz(E12);
    [L12(:,1) L12(:,2)] = find(E12);
    [group1 group2] = make_group12(L12);
    P = RRWM(affinity.K{viewx,viewy}, group1, group2);
else
    exeString = ['P=',algpar.algMethod,'(affinity.K{viewx,viewy},affinity.nP{viewx},affinity.nP{viewy},algpar);'];
    eval(exeString);
end
if algpar.bDisc
    E12 = ones(affinity.nP{viewx},affinity.nP{viewy}); 
    P = convert2Discrete(E12,P);
end
rawScore = P'*affinity.K{viewx,viewy}*P;
rawP = vec2mat(P,affinity.nP{viewx},affinity.nP{viewy});
