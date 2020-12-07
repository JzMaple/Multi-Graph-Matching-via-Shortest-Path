function [Xab] = convert2Discrete(E12,XabRaw,bDiscrete)
if ~exist('bDiscrete','var'),bDiscrete=1;end    
if bDiscrete
        Xab = zeros(size(E12)); Xab(find(E12)) = XabRaw;
        Xab = discretisationMatching_hungarian(Xab,E12);
        Xab = Xab(find(E12));
    else
        Xab = XabRaw;
end
