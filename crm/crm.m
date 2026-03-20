function [w_x, w_y, lambda3, Wxs, Wys, lambdas, corrs] = ...
    crm(C_xx, C_yy, C_xy, D_xy, options)
%CRM  Unified CRM solver — dispatches to the appropriate sub-function.
%
%   [w_x, w_y, lambda3] = crm(C_xx, C_yy, C_xy, D_xy)
%   [w_x, w_y, lambda3] = crm(..., gamma=0.05)
%   [w_x, w_y, lambda3, Wxs, Wys, lambdas, corrs] = crm(..., n_init=5)
%   [w_x, w_y] = crm(..., sparsity=0.7)
%   [w_x, w_y, ~, Wxs, Wys, ~, corrs] = crm(..., sparsity=0.7, n_init=5)
%
%   Dispatch logic:
%     sparse + multi-init  → compute_weights_sparse_multi_rand
%     sparse + single-init → compute_weights_sparse_init_rand
%     dense  + multi-init  → compute_weights_multi_rand
%     dense  + single-init → compute_weights_init_rand          (DEFAULT)
%
%   Sparse mode is triggered when sparsity, theta_x, or theta_y is set.
%
%   See also compute_weights_init_rand, compute_weights_multi_rand,
%            compute_weights_sparse_init_rand, compute_weights_sparse_multi_rand

    arguments
        C_xx double {mustBeSquareMatrix(C_xx)}
        C_yy double {mustBeSquareMatrix(C_yy)}
        C_xy double
        D_xy double
        % --- Common ---
        options.f (1,1) double {mustBePositive} = 1
        options.gamma (1,1) double {mustBeNonnegative} = 0
        options.chlsky (1,1) logical = true
        % --- Multi-init ---
        options.n_init (1,1) double {mustBeInteger, mustBePositive} = 1
        options.rank_ratio_thresh (1,1) double = NaN
        % --- Sparse (any non-default triggers sparse path) ---
        options.sparsity (1,1) double = NaN
        options.theta_x (1,1) double {mustBeNonnegative} = 0
        options.theta_y (1,1) double {mustBeNonnegative} = 0
        options.mu (1,1) double = Inf
        options.step_size (1,1) double {mustBeNonnegative} = 0
        options.max_iter (1,1) double {mustBeInteger, mustBePositive} = 10000
        options.tol (1,1) double {mustBePositive} = 1e-6
    end

    is_sparse = ~isnan(options.sparsity) || options.theta_x > 0 || options.theta_y > 0;

    % --- Initialize optional multi-init outputs ---
    Wxs = []; Wys = []; lambdas = []; corrs = [];

    if is_sparse && options.n_init > 1
        % ---- Sparse multi-init ----
        args = build_sparse_multi_args(options);
        [w_x, w_y, Wxs, Wys, corrs] = ...
            compute_weights_sparse_multi_rand(C_xx, C_yy, C_xy, D_xy, args{:});
        lambda3 = NaN;

    elseif is_sparse
        % ---- Sparse single-init ----
        args = build_sparse_args(options);
        [w_x, w_y] = ...
            compute_weights_sparse_init_rand(C_xx, C_yy, C_xy, D_xy, args{:});
        lambda3 = NaN;

    elseif options.n_init > 1
        % ---- Dense multi-init ----
        args = build_multi_args(options);
        [w_x, w_y, lambda3, Wxs, Wys, lambdas, corrs] = ...
            compute_weights_multi_rand(C_xx, C_yy, C_xy, D_xy, args{:});

    else
        % ---- Dense single-init (DEFAULT) ----
        [w_x, w_y, lambda3] = compute_weights_init_rand( ...
            C_xx, C_yy, C_xy, D_xy, ...
            f=options.f, gamma=options.gamma, chlsky=options.chlsky, k=0);
    end
end


% =========================================================================
%  Helper functions — strip irrelevant options for each sub-function
% =========================================================================

function args = build_multi_args(opts)
    s.f      = opts.f;
    s.gamma  = opts.gamma;
    s.chlsky = opts.chlsky;
    s.n_init = opts.n_init;
    s.rank_ratio_thresh = opts.rank_ratio_thresh;
    args = namedargs2cell(s);
end

function args = build_sparse_args(opts)
    s.f         = opts.f;
    s.gamma     = opts.gamma;
    s.chlsky    = opts.chlsky;
    s.mu        = opts.mu;
    s.step_size = opts.step_size;
    s.sparsity  = opts.sparsity;
    s.theta_x   = opts.theta_x;
    s.theta_y   = opts.theta_y;
    s.max_iter  = opts.max_iter;
    s.tol       = opts.tol;
    s.k         = 0;
    args = namedargs2cell(s);
end

function args = build_sparse_multi_args(opts)
    s.f         = opts.f;
    s.gamma     = opts.gamma;
    s.chlsky    = opts.chlsky;
    s.mu        = opts.mu;
    s.step_size = opts.step_size;
    s.sparsity  = opts.sparsity;
    s.theta_x   = opts.theta_x;
    s.theta_y   = opts.theta_y;
    s.max_iter  = opts.max_iter;
    s.tol       = opts.tol;
    s.n_init    = opts.n_init;
    s.rank_ratio_thresh = opts.rank_ratio_thresh;
    args = namedargs2cell(s);
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
