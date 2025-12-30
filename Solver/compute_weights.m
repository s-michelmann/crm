function [w_x, w_y, lambda3] = compute_weights(C_xx, C_yy,C_xy, D_xy, f, gamma, chlsky)


    if nargin > 5 & ~isempty(gamma)      %   Apply regularization on covariance matrices.
    
        I = eye(size(C_xx,1));
        C_xx = C_xx + gamma*I; %Tuzhilina, Tozzi, Hastie (2021)
        I = eye(size(C_yy,1));
        C_yy = C_yy + gamma*I;
    
    end
    
    if nargin < 5 || isempty(f)
        f = 1;
    end
    
    if nargin < 6 || ~chlsky
        if nargin >= 4   %   Find "f" solutions.
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
    
    else

        % --- Cholesky factors (avoid inv) ---
        [Lx,pflag] = chol(C_xx,'lower');
        [Ly,qflag] = chol(C_yy,'lower');
        if pflag || qflag
            error('C_xx/C_yy not SPD. Increase gamma.');
        end

        % --- Objective for lambda3 uses consistent eigen mode and solves ---
        if nargin >= 4
            fun = @(l) foo3(Lx, Ly, C_xy, D_xy, l, f);
        else
            fun = @(l) foo3(Lx, Ly, C_xy, D_xy, l, 1);
        end
        lambda3 = fminsearch(fun, 0);

        % --- Build M with solves: M = K*K' where K = Cxx^{-1/2}(Cxy+lD)Cyy^{-T/2} ---
        K = (Lx \ (C_xy + lambda3 * D_xy)) / Ly';
        M = K * K';                % symmetric PSD

        % --- Largest real-part eigenvector (consistent with objective) ---
        opts.isreal = true;
        [W,D] = eigs(M, f, 'lr', opts);
        vx = W(:,end);

        % --- Map back, normalize in original metric ---
        w_x = Lx' \ vx;
        w_x = w_x ./ sqrt(w_x' * C_xx * w_x);

        lbd = sqrt(D(end,end));
        % w_y = -Cyy^{-1} (Cxy + l Dxy)' / lbd * w_x via solves
        rhs = (C_xy + lambda3 * D_xy)' * w_x;
        w_y = -(C_yy \ rhs) / lbd;


    end
    if w_x'*C_xy*w_y < 0
        w_y = -w_y; % The sign of "lbd" is not constrained. Make sure we maximize correlation
    end

end


function [tst] = foo2(C_xx, C_yy, C_xy, D_xy, lbd3, f)
    % calculate the f largest eigenvalues only!
    M = inv(C_xx)*(C_xy+lbd3*D_xy)*inv(C_yy)* ((C_xy+lbd3*D_xy)');
    [W,D] = eigs(M,f,'lr');
    w_x = W(:,f);
    w_x = w_x./sqrt(w_x'*C_xx*w_x);
    w_y = inv(C_yy)*(C_xy+lbd3*D_xy)'*w_x;
    tst = abs(w_x'*D_xy*w_y);
end



function [tst] = foo3(Lx, Ly, C_xy, D_xy, lbd3, f)
    % --- Consistent eigen criterion and solves (no inv) ---
    % M = K*K' with K = Cxx^{-1/2}(Cxy+lD)Cyy^{-T/2}
    K = (Lx \ (C_xy + lbd3 * D_xy)) / Ly';
    M = K * K';

    opts.isreal = true; 
    [W,~] = eigs(M, f, 'lr', opts);
    vx = W(:,end);

    % w_x in original coordinates (scale irrelevant for tst)
    w_x = Lx' \ vx;
    w_x = w_x ./ max(norm(w_x), 1e-12);

    % w_y via solves (unnormalized ok for confound objective)
    rhs = (C_xy + lbd3 * D_xy)' * w_x;
    w_y = Ly \ (Ly' \ rhs);

    % Confound term to minimize
    tst = abs(w_x' * D_xy * w_y);
end
