function [w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, D_xy, options)
%COMPUTE_WEIGHTS  Single-solution CRM weight vectors.
%
%   [w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, D_xy)
%   [w_x, w_y, lambda3] = compute_weights(..., f=2, gamma=0.01, chlsky=true)

    arguments
        C_xx double {mustBeSquareMatrix(C_xx)}
        C_yy double {mustBeSquareMatrix(C_yy)}
        C_xy double
        D_xy double
        options.f (1,1) double {mustBePositive} = 1
        options.gamma (1,1) double {mustBeNonnegative} = 0
        options.chlsky (1,1) logical = true
    end

    f      = options.f;
    gamma  = options.gamma;
    chlsky = options.chlsky;

    % --- Regularization ---
    if gamma > 0
        C_xx = C_xx + gamma * eye(size(C_xx,1));
        C_yy = C_yy + gamma * eye(size(C_yy,1));
    end

    if ~chlsky
        fun = @(l) foo2(C_xx, C_yy, C_xy, D_xy, l, f);
        lambda3 = fminsearch(fun, 0);

        M = inv(C_xx) * (C_xy + lambda3*D_xy) * inv(C_yy) * (C_xy + lambda3*D_xy)';
        [W, D] = eigs(M, f, 'lr');
        w_x = W(:,f);
        w_x = w_x ./ sqrt(w_x' * C_xx * w_x);
        lbd = sqrt(D(f,f));
        w_y = -inv(C_yy) * (C_xy + lambda3*D_xy)' / lbd * w_x;

    else
        % --- Cholesky factors (avoid inv) ---
        [Lx, pflag] = chol(C_xx, 'lower');
        [Ly, qflag] = chol(C_yy, 'lower');
        if pflag || qflag
            error('C_xx/C_yy not SPD. Increase gamma.');
        end

        fun = @(l) foo3(Lx, Ly, C_xy, D_xy, l, f);
        lambda3 = fminsearch(fun, 0);

        K = (Lx \ (C_xy + lambda3 * D_xy)) / Ly';
        M = K * K';

        opts.isreal = true;
        [W, D] = eigs(M, f, 'lr', opts);
        vx = W(:,end);

        w_x = Lx' \ vx;
        w_x = w_x ./ sqrt(w_x' * C_xx * w_x);

        lbd = sqrt(D(end,end));
        rhs = (C_xy + lambda3 * D_xy)' * w_x;
        w_y = -(C_yy \ rhs) / lbd;
    end

    if w_x' * C_xy * w_y < 0
        w_y = -w_y;
    end
end


function tst = foo2(C_xx, C_yy, C_xy, D_xy, lbd3, f)
    M = inv(C_xx)*(C_xy+lbd3*D_xy)*inv(C_yy)*((C_xy+lbd3*D_xy)');
    [W,D] = eigs(M,f,'lr');
    w_x = W(:,f);
    w_x = w_x./sqrt(w_x'*C_xx*w_x);
    w_y = inv(C_yy)*(C_xy+lbd3*D_xy)'*w_x;
    tst = abs(w_x'*D_xy*w_y);
end


function tst = foo3(Lx, Ly, C_xy, D_xy, lbd3, f)
    K = (Lx \ (C_xy + lbd3 * D_xy)) / Ly';
    M = K * K';

    opts.isreal = true;
    [W,~] = eigs(M, f, 'lr', opts);
    vx = W(:,end);

    w_x = Lx' \ vx;
    w_x = w_x ./ max(norm(w_x), 1e-12);

    rhs = (C_xy + lbd3 * D_xy)' * w_x;
    w_y = Ly \ (Ly' \ rhs);

    tst = abs(w_x' * D_xy * w_y);
end

function mustBeSquareMatrix(M)
    if ~ismatrix(M) || size(M,1) ~= size(M,2)
        error('Matrix must be square.');
    end
end

function mustBeNonnegative(x)
    if any(x < 0)
        error('Value must be nonnegative.');
    end
end
