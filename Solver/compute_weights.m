function [w_x, w_y, lambda3] = compute_weights(C_xx, C_yy,C_xy, D_xy, f, gamma)

    if nargin == 5       %   Apply regularization on covariance matrices.
        I = eye(size(C_xx,1));
        C_xx = (1-gamma)* C_xx + gamma*I*trace(C_xx);
        I = eye(size(C_yy,1));
        C_yy = (1-gamma)* C_yy + gamma*I*trace(C_yy);
        fun = @(l) foo2(C_xx,C_yy,C_xy, D_xy,l, f);
    
    elseif nargin == 4   %   Find "f" solutions.
        fun = @(l) foo2(C_xx,C_yy,C_xy, D_xy,l, f);
    else
        fun = @(l) foo2(C_xx,C_yy,C_xy, D_xy,l, 1);
    end
    
    lambda3 = fminsearch(fun,0);
    
    M = inv(C_xx)*(C_xy+lambda3*D_xy)*inv(C_yy)* ((C_xy+lambda3*D_xy)');
    [W,D] = eigs(M,f);
    w_x = W(:,f);
    w_x = w_x./sqrt(w_x'*C_xx*w_x);
    lbd = sqrt(D(f,f));
    w_y = -inv(C_yy)*(C_xy+lambda3*D_xy)'/lbd *w_x;
    if w_x'*C_xy*w_y < 0
        w_y = -w_y; % The sign of "lbd" is not constrained. Make sure we maximize correlation
    end

end

function [tst] = foo2(C_xx, C_yy, C_xy, D_xy, lbd3, f)
    % calculate the f largest eigenvalues only!
    M = inv(C_xx)*(C_xy+lbd3*D_xy)*inv(C_yy)* ((C_xy+lbd3*D_xy)');
    [W,D] = eigs(M,f);
    w_x = W(:,f);
    w_x = w_x./sqrt(w_x'*C_xx*w_x);
    w_y = inv(C_yy)*(C_xy+lbd3*D_xy)'*w_x;
    tst = abs(w_x'*D_xy*w_y);
end

