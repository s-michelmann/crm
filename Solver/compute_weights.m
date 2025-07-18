function [w_x, w_y, lambda3] = compute_weights(C_xx, C_yy,C_xy, D_xy, f, gamma)


    if nargin == 5       %   Apply regularization on covariance matrices.

        % Base gamma value
        gamma_base = gamma;
        
        % Compute Frobenius norms
        norm_Cxx = norm(C_xx, 'fro');
        norm_Cyy = norm(C_yy, 'fro');
        norm_Cxy = norm(C_xy, 'fro');
        norm_Dxy = norm(D_xy, 'fro');
        
        % Compute average norm for reference
        norm_max = max([norm_Cxx, norm_Cyy, norm_Cxy, norm_Dxy]);
        
        % Scale gamma for each matrix
        gamma_Cxx = gamma_base * (norm_max / norm_Cxx);
        gamma_Cyy = gamma_base * (norm_max / norm_Cyy);
        gamma_Cxy = gamma_base * (norm_max / norm_Cxy);
        gamma_Dxy = gamma_base * (norm_max / norm_Dxy);

        I = eye(size(C_xx,1));

        % Regularize C_xy and D_xy if square
        if size(C_xy,1) == size(C_xy,2)
            C_xy = (1-gamma_Cxy) * C_xy + gamma_Cxy * I * trace(C_xy);
            C_xx = (1-gamma_Cxx) * C_xx + gamma_Cxx * I * trace(C_xx);
            C_yy = (1-gamma_Cyy) * C_yy + gamma_Cyy * I * trace(C_yy);
            D_xy = (1-gamma_Dxy) * D_xy + gamma_Dxy * I * trace(D_xy);

        else % Shrink toward zero if non-square
            C_xy = (1-gamma_Cxy) * C_xy;  
            D_xy = (1-gamma_Dxy) * D_xy;
            C_xx = (1-gamma_Cxx) * C_xx;
            C_yy = (1-gamma_Cyy) * C_yy;
        end

        if size(D_xy,1) == size(D_xy,2)
        else
        end


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
    [W,D] = eigs(M,f,'lr');
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
