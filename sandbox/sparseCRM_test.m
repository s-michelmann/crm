%% Test Script for Sparse CRM
close all; clear all;

n_samples = 10000;
p = 22; % dimensions for x
q = 40; % dimensions for y

% True weights
w_x_true = zeros(p, 1); w_x_true(1:5) = 1; % Only first 5 features relevant
w_y_true = zeros(q, 1); w_y_true(1:5) = 1; 

% Confound that we want to ignore
d_x_true = zeros(p, 1); d_x_true(10:15) = 1; % Overlaps with different features randi(q,1,5)
d_y_true = zeros(q, 1); d_y_true(10:15) = 1;

% Generate Data
S = randn(n_samples, 1);
C = randn(n_samples, 1);
X = S * w_x_true' + 2 * C * d_x_true' + 0.5 * randn(n_samples, p); % Data matrices: Signal + Confound + Noise
Y = S * w_y_true' + 2 * C * d_y_true' + 0.5 * randn(n_samples, q);

% Covarince matrices like in normal CRM
C_xx = (X' * X) / n_samples;
C_yy = (Y' * Y) / n_samples;
C_xy = (X' * Y) / n_samples;
D_xy = d_x_true * d_y_true'; 

% Sparse CRM Parameters
params.theta_x = 1;  % Sparsity constraint (lower = sparser)
params.theta_y = 5;
params.alpha = 0.05;    % Weight for the push away from D_xy
params.beta = 0.05;     % Weight for the pull toward C_xy
params.max_iter = 500;
params.tol = 1e-5;
params.gamma = 0;    % Seb's Ridge term. For good measure

% Running sparse CRM + Plot
[w_x, w_y] = compute_weights_sparse(C_xx, C_yy, C_xy, D_xy, params);

figure(1),clf;

subplot(1,2,1);
stem(w_x, 'filled'); hold on;
stem(w_x_true, 'r--', 'LineWidth', 1.5);
title('Weights for X');
legend('Estimated w_x', 'True Signal Support');
xlabel('Feature Index');

subplot(1,2,2);
stem(w_y, 'filled'); hold on;
stem(w_y_true, 'r--', 'LineWidth', 1.5);
title('Weights for Y');
legend('Estimated w_y', 'True Signal Support');
xlabel('Feature Index');

% Double check results
corr_result = (w_x' * C_xy * w_y);
confound_result = (w_x' * D_xy * w_y);
disp([corr_result, confound_result])