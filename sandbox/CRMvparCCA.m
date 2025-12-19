close all;
clear all;

tic %8h runtime

rng(1)

Nobservations = 10000;
Nsubjects = 10;
Nrepeats = 48;
Nrank = 8;

all_results = zeros(Nrank, 8);
noisecounter = 1;
M = zeros(30,8);

for noise = 0.1:0.1:3

    parfor rank = 1:Nrank
    
        data = zeros(4, Nrepeats);
        for repeat = 1:Nrepeats
            hiddensignal = randn(1, Nobservations); % A rank 1 hidden signal that we are interested in.
            A = randn(Nsubjects, 1)*hiddensignal;
    
            confound = noise*randn(rank, Nobservations);
            S = randn(Nsubjects, rank)*confound;
            T = randn(Nsubjects, rank)*confound;
    
            X = A + S + randn(Nsubjects, Nobservations); 
            Y = A + T + randn(Nsubjects, Nobservations); 
            
            % CRM
            C_xy = X*Y';
            C_xx = X*X';
            C_yy = Y*Y';
            D_xy = S*T';
            [r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, D_xy);
            if length(r_wx)>0
                w_x = r_wx(:,1);
                w_y = r_wy(:,1);
                cc = corrcoef(w_x'*X, hiddensignal);
                data(1, repeat, rank) = cc(2,1).^2;
                data(2, repeat, rank) = (w_x'*D_xy*w_y).^2;
            end

            % parCCA
            Z = (S + T)/2;
            C_xz = X*Z';
            C_zz = Z*Z';
            C_zy = Z*Y';
            D_xy_parCCA = -C_xz*inv(C_zz)*C_zy;
            [r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, D_xy_parCCA);
            if length(r_wx)>0
                w_x = r_wx(:,1);
                w_y = r_wy(:,1);
                cc = corrcoef(w_x'*X, hiddensignal);
                data(3, repeat, rank) = cc(2,1).^2;
                data(4, repeat, rank) = (w_x'*D_xy*w_y).^2;
            end
        end
    
        a = mean(data(1,:));                 %1
        b = std(data(1,:)) / sqrt(Nrepeats); %2
    
        c = mean(data(2,:));                 %3
        d = std(data(2,:)) / sqrt(Nrepeats); %4
    
        e = mean(data(3,:));                 %5
        f = std(data(3,:)) / sqrt(Nrepeats); %6
    
        g = mean(data(4,:));                 %7
        h = std(data(4,:)) / sqrt(Nrepeats); %8
    
        all_results(rank,:) = [a,b,c,d,e,f,g,h];
    end

    M(noisecounter,:) = (all_results(:,1)-all_results(:,5))./all_results(:,1);
    noisecounter = noisecounter + 1;
end

%%

figure(1),clf;

subplot(1,2,1), hold on;
errorbar(1:Nrank, all_results(:,1), all_results(:,2), 'o', 'MarkerFaceColor', "#0072BD", 'Color',"#0072BD") % CRM result for best corr (max)
errorbar(1:Nrank, all_results(:,5), all_results(:,6), 'o', 'MarkerFaceColor', "#77AC30", 'Color',"#77AC30") % parCCA zero for best corr
ylim([-0.1,1.1])
xlim([-0.1, 8.5])
xlabel("Rank of nuisance variable")
ylabel("R^2 with hidden signal")
set(gca, 'tickdir','out');
legend('CRM', 'parCCA', 'Location','northeast')
text(-2.7, 1.1, "a", 'FontSize', 21)

subplot(1,2,2), hold on;
imagesc(1:8, 0.25:0.25:3, M)
set(gca, 'YDir', 'normal')
colorbar;
xlabel("Rank of nuisance variable")
xlim([1,8])
ylim([0.25,3.1])
ylabel("Noise")
set(gca, 'tickdir','out');
title("Improvement of CRM over par CCA in %")
rectangle('Position',[1.1 2.9 6.8 0.15],'Curvature',0.2, 'EdgeColor','red')
text(0, 3.1, "b", 'FontSize', 21)


%exportgraphics(figure(1), 'suppl_improvement.pdf');

toc