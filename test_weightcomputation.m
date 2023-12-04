figure(1), clf;


Nobservations = 10000;
Nsubjects = 20;
Nrepeats = 100; % Averaging For the plot

rng(1)

for noise = 0.05:0.1:2
    data = zeros(3,Nrepeats);
    for repeat = 1:Nrepeats
        X = randn(Nsubjects, Nobservations); 

        M = randn(Nsubjects,Nsubjects);

        Y = M*X + noise*randn(Nsubjects, Nobservations); % Datasets so that X and Y are correlated + corrupted by noise

        C_xy = X*Y';
        C_xx = X*X';
        C_yy = Y*Y';
        
        S = randn(Nsubjects, Nobservations);  % Same brain areas in different experiment. Just noise
        T = randn(Nsubjects,Nsubjects)*S + 0.5*randn(Nsubjects, Nobservations);
        D_xy = S*T';
        
        f = 2;
        [w_x, w_y, lbd3i] = compute_weights(C_xx,C_yy,C_xy, D_xy,f);

        data(1, repeat) = (w_x'*C_xy*w_y).^2; %should be as close to "1" as possible
        data(2, repeat) = (w_x'*D_xy*w_y).^2; %should be as close to "0" as possible;
        cc=corrcoef(w_x, w_y'*M);
        data(3, repeat) = cc(2,1).^2;

        [w_x, w_y, lbd3i] = compute_weights(C_xx,C_yy,C_xy, 0*D_xy,f);
        data(4, repeat) = (w_x'*C_xy*w_y).^2;
        data(5, repeat) = (w_x'*D_xy*w_y).^2;
        cc=corrcoef(w_x, w_y'*M);
        data(6, repeat) = cc(2,1).^2;
    end

    figure(1)
    
    subplot(1,3,1), hold on;
    mu = mean(data(1,:));
    err = std(data(1,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'o', 'MarkerFaceColor', "#0072BD", 'Color',"#0072BD") % CRM result for best corr (max)
    mu = mean(data(4,:));
    err = std(data(4,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'o', 'MarkerFaceColor', "#77AC30", 'Color',"#77AC30") % CCA zero for best corr

    subplot(1,3,2), hold on;
    mu = mean(data(2,:));
    err = std(data(2,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'o', 'MarkerFaceColor', "#0072BD", 'Color',"#0072BD") % CRM result for D (=0)
    mu = mean(data(5,:));
    err = std(data(5,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'o', 'MarkerFaceColor', "#77AC30", 'Color',"#77AC30") % CCA result for D (not constrained)

    subplot(1,3,3), hold on;
    mu = mean(data(3,:));
    err = std(data(3,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'o', 'MarkerFaceColor', "#0072BD", 'Color',"#0072BD") % CRM result for corr(wTrue, w)
    mu = mean(data(6,:));
    err = std(data(6,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'o', 'MarkerFaceColor', "#77AC30", 'Color',"#77AC30") % CCA result for corr(wTrue, w)
end

subplot(1,3,1)
ylim([-0.1,1.1])
xlim([-0.1,2.1])
% ylabel("CRM (blue); CCA (green)")
xlabel("Noise \alpha")
ylabel("w_x'*C_{xy}*w_y")
set(gca, 'tickdir','out');
legend('RSM', 'CCA', 'Location','southwest')

subplot(1,3,2)
ylim([-0.1,5.1])
xlim([-0.1,2.1])
xlabel("Noise")
ylabel("w_x'*D_xy*w_y")
set(gca, 'tickdir','out');

subplot(1,3,3)
ylim([-0.1,1.1])
xlim([-0.1,2.1])
xlabel("Noise")
ylabel("corr(w_x, w_y'*M)^2")
set(gca, 'tickdir','out');

text(-0.5,1.15, "C", 'FontSize', 16)
text(-3.3,1.15, "B", 'FontSize', 16)
text(-6.3,1.15, "A", 'FontSize', 16)

%% Making a matrix with one eigenvalue of zero
% Mtmp = randn(Nsubjects,Nsubjects);
% defu = @(u) det(Mtmp + u*eye(size(Mtmp)));
% usingular = fzero(defu,0)
% M = Mtmp + usingular*eye(size(Mtmp));
