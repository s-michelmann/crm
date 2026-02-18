function [w_x, w_y] = compute_weights_sparse(C_xx, C_yy, C_xy, D_xy, params)
%COMPUTE_WEIGHTS_SPARSE  Sparse CRM (legacy interface).
%
%   [w_x, w_y] = compute_weights_sparse(C_xx, C_yy, C_xy, D_xy)
%   [w_x, w_y] = compute_weights_sparse(..., f=2, gamma=0.01, mu=Inf, ...)
%
%   Prefer compute_weights_sparse_init_rand for new code (supports random
%   init, automatic sparsity selection, Cholesky path).
%
%   mu defaults to lambda3 from the dense init (Inf → auto).
%   step_size defaults to 1/sigma_max(C_xy + mu*D_xy) (0 → auto).

    arguments
        C_xx double {mustBeSquareMatrix(C_xx)}
        C_yy double {mustBeSquareMatrix(C_yy)}
        C_xy double
        D_xy double
        params.f (1,1) double {mustBePositive} = 1
        params.mu (1,1) double = Inf              % Inf → auto (use lambda3)
        params.step_size (1,1) double {mustBeNonnegative} = 0  % 0 → auto (1/L)
        params.theta_x (1,1) double {mustBePositive} = 100
        params.theta_y (1,1) double {mustBePositive} = 100
        params.gamma (1,1) double {mustBeNonnegative} = 0
        params.max_iter (1,1) double {mustBeInteger, mustBePositive} = 10000
        params.tol (1,1) double {mustBePositive} = 0.001
    end

    theta_x = params.theta_x;
    theta_y = params.theta_y;

    [w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, D_xy, ...
        f=params.f, gamma=params.gamma);

    % --- Confound penalty weight ---
    if isinf(params.mu)
        mu = lambda3;
    else
        mu = params.mu;
    end

    % --- Step size ---
    if params.step_size == 0
        L = svds(C_xy + mu * D_xy, 1);
        step = 1 / max(L, eps);
    else
        step = params.step_size;
    end

    % --- Combined matrix (matches dense solver convention) ---
    M_xy = C_xy + mu * D_xy;

    % --- Regularized covariance (same metric as dense init) ---
    gamma = params.gamma;
    p = size(C_xx, 1);
    q = size(C_yy, 1);
    C_xx_r = C_xx + gamma * eye(p);
    C_yy_r = C_yy + gamma * eye(q);

    % Iterative Solver
    for iter = 1:params.max_iter
        w_x_old = w_x;
        w_y_old = w_y;

        w_x = w_x + step * (M_xy * w_y);
        w_y = w_y + step * (M_xy' * w_x);

        w_x = apply_threshold(w_x, theta_x);
        w_y = apply_threshold(w_y, theta_y);

        nrm_x = w_x' * C_xx_r * w_x;
        if nrm_x < 1e-20
            w_x = w_x_old;
            break
        end
        w_x = w_x / sqrt(nrm_x);

        nrm_y = w_y' * C_yy_r * w_y;
        if nrm_y < 1e-20
            w_y = w_y_old;
            break
        end
        w_y = w_y / sqrt(nrm_y);

        if norm(w_x - w_x_old) + norm(w_y - w_y_old) < params.tol
            break;
        end
    end
end

function w_out = apply_threshold(w, theta)

    l1_norm = sum(abs(w));
    if l1_norm <= theta
        w_out = w;
        return;
    end

    delta_min = 0;
    delta_max = max(abs(w));

    for k = 1:50
        delta = (delta_min + delta_max) / 2;
        w_temp = sign(w) .* max(abs(w) - delta, 0);

        if sum(abs(w_temp)) > theta
            delta_min = delta;
        else
            delta_max = delta;
        end
    end
    w_out = sign(w) .* max(abs(w) - delta_max, 0);
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
