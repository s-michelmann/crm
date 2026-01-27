close all; clear all;

rng(42);

n = 1000;
p = 50;

true_signal = 0.2*randn(n, 1);   % A weak signal to recover
confound_signal = randn(n, 1);   % The noise signal to orthogonalize against
mix = randn(1, p);

X = true_signal                           + 0.5 * randn(n, 1);
Y = (true_signal * mix) + confound_signal + 0.5 * randn(n, p);

T = confound_signal; 

X = X - mean(X);
Y = Y - mean(Y);
T = T - mean(T);

Cyy = (Y' * Y);
Cxy = (X' * Y); 
Dxy = (T' * Y);
Cyy_inv = inv(Cyy);

% do univariateCRM
lambda3 = - Dxy * Cyy_inv * Cxy' ./ Dxy * Cyy_inv * Dxy';
w_y_raw = Cyy_inv * (Cxy + lambda3 * Dxy)';
w_y = w_y_raw ./ sqrt(w_y_raw' * Cyy * w_y_raw);

% For comparison, do standard regression.
w_standard = Cyy_inv * Cxy';
w_standard = w_standard / sqrt(w_standard' * Cyy * w_standard);

% Correlation with True Signal
corr_crm = corr( Y * w_y, true_signal);
corr_standard = corr( Y * w_standard, true_signal);

fprintf('\n')
fprintf('Constraint wDw (CRM): %.4f (Should be ~0)\n', Dxy * w_y);
fprintf('Correlation with True Signa (CRM):   %.4f\n', corr_crm);
fprintf('\n')
fprintf('Constraint wDw (Standard): %.4f (Likely non-zero)\n', Dxy * w_standard);
fprintf('Correlation with True Signal (Standard):   %.4f\n', corr_standard);