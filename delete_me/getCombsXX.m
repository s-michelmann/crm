function [same_idc, diff_idc] = getCombsXX(labelsx)    

    Nx = size(labelsx,1);
   
    [A, B] = meshgrid(1:Nx, 1:Nx) ;
    A(tril(ones(size(A)))==1) = nan;
    B(tril(ones(size(B)))==1) = nan;
    c = cat(2,A',B');
    d  = reshape(c,[],2);
    d(isnan(d(:,1)),:) = [];
    same_idc = d(labelsx(d(:,1)) == labelsx(d(:,2)),:);
    diff_idc = d(labelsx(d(:,1)) ~= labelsx(d(:,2)),:);  
end
