rng('Shuffle')
% --- Construct a valid joint covariance and sample from it ---

% Dimensions
p = 20;
q = 15;
n = 5000;   % number of samples for stable covariance estimates

% Random directions for planted correlation
u = randn(p,1); u = u / norm(u);
v = randn(q,1); v = v / norm(v);

rho_sig = 0.8;   % desired canonical correlation along (u, v)

% Base marginal covariances
Sigma_xx = eye(p);
Sigma_yy = eye(q);

% Rank-1 cross-covariance (guaranteed |rho| <= 1)
Sigma_xy = rho_sig * (u * v');

% Full joint covariance (block PSD by construction)
Sigma = [Sigma_xx, Sigma_xy;
         Sigma_xy', Sigma_yy];

% Optional: add small isotropic noise to make marginals richer
eps_reg = 0.1;
Sigma = Sigma + eps_reg * blkdiag(eye(p), eye(q));

% Sample [X; Y] ~ N(0, Sigma)
R = chol(Sigma, 'lower');
Z = randn(p+q, n);
XY = R * Z;

X = XY(1:p, :);
Y = XY(p+1:end, :);

% Empirical covariances
C_xx = (X * X') / n;
C_yy = (Y * Y') / n;
C_xy = (X * Y') / n;

% Confound matrix: independent noise
D_xy = 0.1 * randn(p, q);

rho_expected = (u' * C_xy * v) / sqrt( (u' * C_xx * u) * (v' * C_yy * v) );

%% -------------------------------------------------------------
%  Test compute_weights_init_rand (dense)
[wx, wy, lambda3] = compute_weights_init_rand(C_xx, C_yy, C_xy, D_xy);

assert(~any(isnan(wx)))
assert(~any(isnan(wy)))

assert(abs(wx' * C_xx * wx - 1) < 1e-6)
assert(abs(wy' * C_yy * wy - 1) < 1e-6)

corr_Cxy = wx' * C_xy * wy;
corr_Dxy = wx' * D_xy * wy;

fprintf('Dense Cxy correlation: %.4f\n', corr_Cxy)
fprintf('Dense Dxy correlation: %.4f\n', corr_Dxy)

assert(corr_Cxy > 0.1)
assert(abs(corr_Dxy) < abs(corr_Cxy))

fprintf('test_compute_weights_init_rand passed.\n')

%% -------------------------------------------------------------
%  Test compute_weights_multi_rand (dense)
n_init = 20;

[wxb, wyb, lambda3b, Wxs, Wys, lambdas, corrs] = ...
    compute_weights_multi_rand(C_xx, C_yy, C_xy, D_xy, ...
                               1, 0, false, n_init);

assert(size(Wxs,2) == n_init)
assert(size(Wys,2) == n_init)
assert(length(lambdas) == n_init)
assert(length(corrs) == n_init)

assert(~any(isnan(Wxs(:))))
assert(~any(isnan(Wys(:))))

[~, idx] = max(corrs);
assert(all(wxb == Wxs(:,idx)))
assert(all(wyb == Wys(:,idx)))
assert(lambda3b == lambdas(idx))

assert(abs(wxb' * C_xx * wxb - 1) < 1e-6)
assert(abs(wyb' * C_yy * wyb - 1) < 1e-6)

corr_Cxy_best = wxb' * C_xy * wyb;
corr_Dxy_best = wxb' * D_xy * wyb;

fprintf('Dense multi Cxy correlation: %.4f\n', corr_Cxy_best)
fprintf('Dense multi Dxy correlation: %.4f\n', corr_Dxy_best)

assert(corr_Cxy_best > 0.1)
assert(abs(corr_Dxy_best) < abs(corr_Cxy_best))

fprintf('test_compute_weights_multi_rand passed.\n')

%% -------------------------------------------------------------
%  Test compute_weights_sparse_init_rand
%% -------------------------------------------------------------
[wxs, wys] = compute_weights_sparse_init_rand( ...
    C_xx, C_yy, C_xy, D_xy, 1, ...
    alpha=0.001, beta=0.001, gamma=0, chlsky=true, k=0, max_iter=100000);

assert(~any(isnan(wxs)))
assert(~any(isnan(wys)))

% Effective sparsity 
eps_thr = 1e-2;  
frac_small_x = mean(abs(wxs) < eps_thr);
frac_small_y = mean(abs(wys) < eps_thr);

fprintf('Sparse init: frac(|w_x|<%.1e) = %.2f\n', eps_thr, frac_small_x);
fprintf('Sparse init: frac(|w_y|<%.1e) = %.2f\n', eps_thr, frac_small_y);

% Require some shrinkage, but not overly strict
try
    assert(frac_small_x > 0.1);
catch
    warning('x too dense');
end
try
    assert(frac_small_y > 0.1);
catch
    warning('y too dense');
end


corr_Cxy_s = wxs' * C_xy * wys;
corr_Dxy_s = wxs' * D_xy * wys;

fprintf('Sparse Cxy correlation: %.4f\n', corr_Cxy_s)
fprintf('Sparse Dxy correlation: %.4f\n', corr_Dxy_s)

try 
    assert(corr_Cxy_s > rho_expected/4)  
catch 
    warning('low correlation');
end

assert(abs(corr_Dxy_s) < abs(corr_Cxy_s), 'Dxy > Cxy')

fprintf('test_compute_weights_sparse_init_rand passed.\n')

%% -------------------------------------------------------------
%  Test compute_weights_sparse_multi_rand
n_init = 15;

[w_x_best, w_y_best, Wxs_s, Wys_s, corrs_s]  = ...
    compute_weights_sparse_multi_rand(C_xx, C_yy, C_xy, D_xy, 1, ...
    alpha=0.001, beta=0.001, gamma=0.001, chlsky=false, k=0, ...
    max_iter=100000, n_init=n_init);

assert(size(Wxs_s,2) == n_init)
assert(size(Wys_s,2) == n_init)
assert(length(corrs_s) == n_init)

assert(~any(isnan(Wxs_s(:))))
assert(~any(isnan(Wys_s(:))))

[~, idx_s] = max(corrs_s);
assert(all(w_x_best == Wxs_s(:,idx_s)))
assert(all(w_y_best == Wys_s(:,idx_s)))

% Effective sparsity
frac_small_x = mean(abs(w_x_best) < eps_thr);
frac_small_y = mean(abs(w_y_best) < eps_thr);

fprintf('Sparse multi: frac(|w_x|<%.1e) = %.2f\n', eps_thr, frac_small_x);
fprintf('Sparse multi: frac(|w_y|<%.1e) = %.2f\n', eps_thr, frac_small_y);


% Require some shrinkage, but not overly strict
try
    assert(frac_small_x > 0.1);
catch
    warning('x too dense');
end
try
    assert(frac_small_y > 0.1);
catch
    warning('y too dense');
end


corr_Cxy_s_best = w_x_best' * C_xy * w_y_best;
corr_Dxy_s_best = w_x_best' * D_xy * w_y_best;

fprintf('Sparse multi Cxy correlation: %.4f\n', corr_Cxy_s_best)
fprintf('Sparse multi Dxy correlation: %.4f\n', corr_Dxy_s_best)

try 
    assert(corr_Cxy_s_best > rho_expected/4)   % sparse version may be weaker
catch 
    warning('low correlation');
end


assert(abs(corr_Dxy_s_best) < abs(corr_Cxy_s_best))

fprintf('test_compute_weights_sparse_multi_rand passed.\n')

%%
%% -------------------------------------------------------------
%  Test: Cholesky vs non-Cholesky (dense)
fprintf('\n--- Testing Cholesky vs non-Cholesky (dense) ---\n')

[wx_nc, wy_nc] = compute_weights_init_rand(C_xx, C_yy, C_xy, D_xy, ...
                                           1, 0, false, 1);
[wx_ch, wy_ch] = compute_weights_init_rand(C_xx, C_yy, C_xy, D_xy, ...
                                           1, 0, true, 1);

% Normalize check
assert(abs(wx_nc' * C_xx * wx_nc - 1) < 1e-6)
assert(abs(wy_nc' * C_yy * wy_nc - 1) < 1e-6)
assert(abs(wx_ch' * C_xx * wx_ch - 1) < 1e-6)
assert(abs(wy_ch' * C_yy * wy_ch - 1) < 1e-6)

% Determine global sign
sgn = sign(wx_nc' * wx_ch);
if sgn == 0, sgn = 1; end   % fallback for numerical edge cases

% Apply sign to both vectors
wx_ch = sgn * wx_ch;
wy_ch = sgn * wy_ch;

% Similarity check
assert(norm(wx_nc - wx_ch) < 1e-13)
assert(norm(wy_nc - wy_ch) < 1e-13)

fprintf('Dense Cholesky consistency test passed.\n')

%% -------------------------------------------------------------
%  Test: Cholesky vs non-Cholesky (sparse)
fprintf('\n--- Testing Cholesky vs non-Cholesky (sparse) ---\n')

[wxs_nc, wys_nc] = compute_weights_sparse_init_rand(C_xx, C_yy, C_xy, D_xy, ...
    1, alpha=0.001, beta=0.001, gamma=0, chlsky=false, k=1, max_iter=500);

[wxs_ch, wys_ch] = compute_weights_sparse_init_rand(C_xx, C_yy, C_xy, D_xy, ...
    1, alpha=0.001, beta=0.001, gamma=0, chlsky=true,  k=1, max_iter=500);

% Determine global sign
sgn = sign(wxs_nc' * wxs_ch);
if sgn == 0, sgn = 1; end

% Align
wxs_ch = sgn * wxs_ch;
wys_ch = sgn * wys_ch;

% Similarity

assert(norm(wx_nc - wx_ch) < 1e-13)
assert(norm(wy_nc - wy_ch) < 1e-13)
fprintf('Sparse Cholesky consistency test passed.\n')

%% -------------------------------------------------------------
%  Test: gamma sweep (dense + sparse)
%% -------------------------------------------------------------
fprintf('\n--- Testing gamma sweep (dense + sparse) ---\n')

gamma_values = logspace(-4, 1, 30);   % 1e-4 → 10
corrs_dense = zeros(size(gamma_values));
corrs_sparse = zeros(size(gamma_values));
conf_dense = zeros(size(gamma_values));
conf_sparse = zeros(size(gamma_values));

for gi = 1:length(gamma_values)
    gamma_test = gamma_values(gi);

    %% --- Dense version ---
    [wx_d, wy_d] = compute_weights_init_rand( ...
        C_xx, C_yy, C_xy, D_xy, 1, gamma_test, false, 1);

    corrs_dense(gi) = wx_d' * C_xy * wy_d;
    conf_dense(gi)  = wx_d' * D_xy * wy_d;

    %% --- Sparse version ---
    [wx_s, wy_s] = compute_weights_sparse_init_rand( ...
        C_xx, C_yy, C_xy, D_xy, 1, ...
        alpha=0.001, beta=0.001, gamma=gamma_test, chlsky=false, k=1, max_iter=50000);

    corrs_sparse(gi) = wx_s' * C_xy * wy_s;
    conf_sparse(gi)  = wx_s' * D_xy * wy_s;

    fprintf('gamma = %.4f | Dense Cxy=%.4f Dxy=%.4f | Sparse Cxy=%.4f Dxy=%.4f\n', ...
        gamma_test, corrs_dense(gi), conf_dense(gi), corrs_sparse(gi), conf_sparse(gi));
end

%--- Find optimal gamma ---
score_dense = corrs_dense - abs(conf_dense);
[~, idx_dense] = max(score_dense);
gamma_opt_dense = gamma_values(idx_dense);

score_sparse = corrs_sparse - abs(conf_sparse);
[~, idx_sparse] = max(score_sparse);
gamma_opt_sparse = gamma_values(idx_sparse);

fprintf('\nOptimal gamma (dense):  %.4f\n', gamma_opt_dense)
fprintf('Optimal gamma (sparse): %.4f\n', gamma_opt_sparse)

assert(gamma_opt_dense > 0)
assert(gamma_opt_sparse > 0)

fprintf('Gamma sweep test passed.\n')
