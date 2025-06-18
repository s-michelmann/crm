function [w_x, w_y, lbd3i] = compute_weights(C_xx,C_yy,C_xy, D_xy,f, gamma)
%@Cxx covariance matrix of data at time X
%@Cyy covariance matrix of data at time Y
%@Cxy covariance matrix between data at time X and y
%@Dxy covariance matrix between data at time X and Y, where the Data has
%been shuffled

% computes the weights w_x and w_y for Dimension f such that w_x*Data_x is maximally
% correlated with Data_y * w_y, while keeping the correlation between
% w_x*Data_Shuffled and Data_y*w_y at zero.
if nargin>5
    I = eye(size(C_xx,1));
    C_xx = (1-gamma)* C_xx + gamma*I*trace(C_xx);
    I = eye(size(C_yy,1));
    C_yy = (1-gamma)* C_yy + gamma*I*trace(C_yy);
end
if nargin < 5
    fun = @(l) foo2(C_xx,C_yy,C_xy, D_xy,l, 1);
else
    fun = @(l) foo2(C_xx,C_yy,C_xy, D_xy,l, f);
end
lbd3i = fminsearch(fun,0);

M = inv(C_xx)*(C_xy+lbd3i*D_xy)*inv(C_yy)* ((C_xy+lbd3i*D_xy)');
[W,D] = eig(M);
w_x = W(:,f);
w_x = w_x./sqrt(w_x'*C_xx*w_x);
lbd = sqrt(D(f,f));
w_y = -inv(C_yy)*(C_xy+lbd3i*D_xy)'/lbd *w_x;
if w_x'*C_xy*w_y < 0
    w_y = -w_y; % The sign of "lbd" is not constrained. Make sure we maximize correlation
end

end
function [tst] = foo2(C_xx,C_yy,C_xy, D_xy,lbd3, f)

% calculate the f largest eigenvalues only!
M = inv(C_xx)*(C_xy+lbd3*D_xy)*inv(C_yy)* ((C_xy+lbd3*D_xy)');
[W,D] = eig(M);
w_x = W(:,f);
w_x = w_x./sqrt(w_x'*C_xx*w_x);
w_y = inv(C_yy)*(C_xy+lbd3*D_xy)'*w_x;
tst = abs(w_x'*D_xy*w_y);

end

