function [same_idc, diff_idc] = getCombsXY(labelsx, labelsy)    

    Nx = size(labelsx,1);
    Ny = size(labelsy,1);
    
    [A, B] = meshgrid(1:Nx, 1:Ny) ;
    c =cat(2,A',B');
    d  = reshape(c,[],2);
    same_idc = d(labelsx(d(:,1)) == labelsy(d(:,2)),:);
    diff_idc = d(labelsx(d(:,1)) ~= labelsy(d(:,2)),:);
    
    
end