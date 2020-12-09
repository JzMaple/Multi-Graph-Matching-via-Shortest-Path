function [rawP rawScore]= pairMatch(viewx,viewy,algpar)
global affinity
if strcmpi(algpar.algMethod,'FGM')% not supported in the demo code
    P = pairFGM(algpar,viewx,viewy);
else
    exeString = ['P=',algpar.algMethod,'(affinity.K{viewx,viewy},affinity.nP{viewx},affinity.nP{viewy},algpar);'];
%     disp(exeString);
    eval(exeString);
end
if algpar.bDisc
    E12 = ones(affinity.nP{viewx},affinity.nP{viewy}); 
    P = convert2Discrete(E12,P);
end
rawScore = P'*affinity.K{viewx,viewy}*P;
rawP = vec2mat(P,affinity.nP{viewx},affinity.nP{viewy});
