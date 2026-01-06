close all;
clear all;

Nsubjects = 20;
Nobservations = 2000;
noise = 0.01;

relevant_signal  = randn(Nsubjects, Nobservations);

hiddensignal = randn(1, Nobservations); % A low rank irrelevant signal
irrelevant_signal = randn(Nsubjects, 1)*hiddensignal;

Mr = randn(Nsubjects,Nsubjects);
Mi = randn(Nsubjects,Nsubjects);

X =    relevant_signal +    irrelevant_signal + noise*randn(Nsubjects, Nobservations); 
Y = Mr*relevant_signal + Mi*irrelevant_signal + noise*randn(Nsubjects, Nobservations); 

S = X;
T = Mi*irrelevant_signal + noise*randn(Nsubjects, Nobservations);

C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';   

% Method 1: Regressing out the irrelevant signal, and CCA on residuals.
Z = T;
C_xz = X*Z';
C_zz = Z*Z';
C_zy = Z*Y';
D_xy_parCCA = -C_xz*inv(C_zz)*C_zy;
[w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, D_xy_parCCA);
disp("partialCCA")
corrcoef(w_x'*X, w_y'*irrelevant_signal) %should be small
corrcoef(w_x'*X, w_y'*Y) %should be large


% Method 2: CRM  
D_xy = S*T';
[w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, D_xy);
disp("CRM")
corrcoef(w_x'*X, w_y'*irrelevant_signal) %should be small
corrcoef(w_x'*X, w_y'*Y) %should be large

