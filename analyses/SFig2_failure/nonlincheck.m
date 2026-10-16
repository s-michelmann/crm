% CRM Nonlinear Failure Mode Simulation (Updated)
% This script demonstrates how linear CRM fails when a shared signal 
% is driven by a nonlinear transformation of a confound (z^2), while 
% the linear orthogonality constraint passes due to symmetry (Cov(z, z^2) = 0).

close all;
clear all;
tic;

rng(1);
Nobservations = 10000;
Nsubjects = 10;
Nrepeats = 20;
Ncrossval = 5;

% Parameter ranges for the sweep
nonlinear_range = 0.2:0.2:3.0; % Strength of the nonlinear signal (z^2)
noise_range = 0.2:0.2:3.0;     % Amplitude of added Gaussian noise

all_signal_corr = zeros(length(noise_range), length(nonlinear_range));
all_constraint  = zeros(length(noise_range), length(nonlinear_range));
all_best_corr   = zeros(length(noise_range), length(nonlinear_range));

for inoise = 1:length(noise_range)
    noise_amp = noise_range(inoise);
    
    parfor ik = 1:length(nonlinear_range)
        k_nl = nonlinear_range(ik);
        
        fold_cor  = [];
        fold_con  = [];
        fold_best = [];
        
        for repeat = 1:Nrepeats
            % 1. Generate symmetric confound z
            z = randn(1, Nobservations);
            
            % 2. Linear confound components tested by CRM (S and T proportional to z)
            S_linear = randn(Nsubjects, 1) * z;
            T_linear = randn(Nsubjects, 1) * z;
            
            % 3. Nonlinear shared signal driven entirely by z^2
            z_nonlinear = k_nl * (z.^2 - mean(z.^2)); % zero-meaned nonlinear term
            X_shared = randn(Nsubjects, 1) * z_nonlinear;
            Y_shared = randn(Nsubjects, 1) * z_nonlinear;
            
            % 4. Construct observed data X and Y
            X = X_shared + S_linear + noise_amp * randn(Nsubjects, Nobservations);
            Y = Y_shared + T_linear + noise_amp * randn(Nsubjects, Nobservations);
            
            % Cross-validation loop
            crm_cor_fold  = [];
            crm_con_fold  = [];
            crm_best_fold = [];
            
            for ifold = 1:Ncrossval
                sz = floor(Nobservations / Ncrossval);
                testflag = false(1, Nobservations);
                testflag((ifold-1)*sz + 1 : ifold*sz) = true;
                
                Xtrain = X(:, ~testflag);
                Xtest  = X(:, testflag);
                Ytest  = Y(:, testflag);
                Z_test_signal = k_nl * (z(testflag).^2 - mean(z(testflag).^2));
                
                Ytrain = Y(:, ~testflag);
                Strain = S_linear(:, ~testflag);
                Ttrain = T_linear(:, ~testflag);
                
                % CRM matrices
                C_xy = Xtrain * Ytrain';
                C_xx = Xtrain * Xtrain';
                C_yy = Ytrain * Ytrain';
                D_xy = Strain * Ttrain';
                D_xytest = S_linear(:, testflag) * T_linear(:, testflag)';
                
                % Run CRM
                [r_wx, r_wy, ~, ~, ~, ~, ~] = crm(C_xx, C_yy, C_xy, D_xy);
                
                if ~isempty(r_wx)
                    w_x = r_wx(:,1);
                    w_y = r_wy(:,1);
                    
                    % 1. Out-of-sample correlation with nonlinear signal (z^2)
                    cc = corrcoef(w_x' * Xtest, Z_test_signal);
                    if size(cc, 2) > 1
                        crm_cor_fold(end+1) = cc(2,1)^2;
                    else
                        crm_cor_fold(end+1) = 0;
                    end
                    
                    % 2. Constraint satisfaction on testing data
                    crm_con_fold(end+1) = (w_x' * D_xytest * w_y)^2;
                    
                    % 3. CRM best X-Y correlation found out-of-sample
                    cc_best = corrcoef(w_x' * Xtest, w_y' * Ytest);
                    if size(cc_best, 2) > 1
                        crm_best_fold(end+1) = cc_best(2,1)^2;
                    else
                        crm_best_fold(end+1) = 0;
                    end
                end
            end
            
            if ~isempty(crm_cor_fold)
                fold_cor(end+1)  = mean(crm_cor_fold);
                fold_con(end+1)  = mean(crm_con_fold);
                fold_best(end+1) = mean(crm_best_fold);
            end
        end
        
        all_signal_corr(inoise, ik) = mean(fold_cor);
        all_constraint(inoise, ik)  = mean(fold_con);
        all_best_corr(inoise, ik)   = mean(fold_best);
    end
end

toc;

%% Plotting Results
figure(1), clf;
set(gcf, 'Position', [100, 100, 1300, 400]);

% Panel a: Perceived Signal Correlation
subplot(1,3,2), hold on;
imagesc(nonlinear_range, noise_range, all_signal_corr);
set(gca, 'YDir', 'normal');
colorbar;
xlabel("Nonlinear Signal Strength");
ylabel("Noise Amplitude");
title("a) CRM Signal R^2 (Found)");
clim([0, 1]);
set(gca, 'tickdir', 'out');
xlim([0.1,3])
ylim([0.1,3])
text(-0.3, 3.1, "b", 'FontSize', 15)

% Panel b: Constraint Residual
subplot(1,3,3), hold on;
imagesc(nonlinear_range, noise_range, all_constraint);
set(gca, 'YDir', 'normal');
colorbar;
xlabel("Nonlinear Signal Strength");
ylabel("Noise Amplitude");
title("Constraint Residual");
clim([0, 1e-4]);
set(gca, 'tickdir', 'out');
xlim([0.1,3])
ylim([0.1,3])
text(-0.3, 3.1, "c", 'FontSize', 15)

% Panel c: CRM X-Y Correlation
subplot(1,3,1), hold on;
imagesc(nonlinear_range, noise_range, all_best_corr);
set(gca, 'YDir', 'normal');
colorbar;
xlabel("Nonlinear Signal Strength");
ylabel("Noise Amplitude");
title("CRM X-Y Correlation (R^2)");
clim([0, 1]);
set(gca, 'tickdir', 'out');
xlim([0.1,3])
ylim([0.1,3])
text(-0.3, 3.1, "a", 'FontSize', 15)

% save("SupplFig2.mat");
% exportgraphics(figure(1), 'SupplFig2.pdf');