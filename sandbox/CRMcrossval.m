close all;
clear all;

tic % 26h

rng(1)

Nobservations = 10000;
Nsubjects = 10;
Nrepeats = 36; % Something divisible by 12
Nrank = 8;
Ncrossval = 5;
noiserange = 0.1:0.1:3;

all_results = zeros(Nrank, Nrank);
noisecounter = 1;
M = zeros(length(noiserange),Nrank);
N = zeros(length(noiserange),Nrank);

for noise = noiserange
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
            
            CRMfoldcor = []; CRMfoldcon = []; CCAfoldcor = []; CCAfoldcon = [];
            for ifold = 1:Ncrossval
                size = floor(Nobservations/Ncrossval);
                testflag = zeros(1,Nobservations);
                testflag( (ifold-1)*size+1 : (ifold)*size ) = 1;

                Xtrain = X(:,~testflag);
                Xtest = X(:,testflag==1);
                hiddensignaltest = hiddensignal(testflag==1);
                Ytrain = Y(:,~testflag);
                Strain = S(:,~testflag);
                Ttrain = T(:,~testflag);

                % CRM
                C_xy = Xtrain*Ytrain';
                C_xx = Xtrain*Xtrain';
                C_yy = Ytrain*Ytrain';
                D_xy = Strain*Ttrain';
                D_xytest = S(:,testflag==1)*T(:,testflag==1)';
                [r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, D_xy);
                if length(r_wx)>0
                    w_x = r_wx(:,1);
                    w_y = r_wy(:,1);
                    cc = corrcoef(w_x'*Xtest, hiddensignaltest);
                   CRMfoldcor = [CRMfoldcor; cc(2,1).^2];
                   CRMfoldcon = [CRMfoldcon; (w_x'*D_xytest*w_y).^2];
                end
    
                % parCCA
                Z = (Strain + Ttrain)/2;
                C_xz = Xtrain*Z';
                C_zz = Z*Z';
                C_zy = Z*Ytrain';
                D_xy_parCCA = -C_xz*inv(C_zz)*C_zy;
                [r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, D_xy_parCCA);
                if length(r_wx)>0
                    w_x = r_wx(:,1);
                    w_y = r_wy(:,1);
                    cc = corrcoef(w_x'*Xtest, hiddensignaltest);
                    CCAfoldcor = [CCAfoldcor; cc(2,1).^2];
                    CCAfoldcon = [CCAfoldcon; (w_x'*D_xytest*w_y).^2];
                end

            end
            data(1, repeat) = mean(CRMfoldcor);
            data(2, repeat) = mean(CRMfoldcon); % CRM constraint
            data(3, repeat) = mean(CCAfoldcor);
            data(4, repeat) = mean(CCAfoldcon); % CCA constraint

        end
    
        a = mean(data(1,:));                 %1
        b = std(data(1,:)) / sqrt(Nrepeats); %2
    
        c = mean(data(2,:));                 %3
        d = std(data(2,:)) / sqrt(Nrepeats); %4
    
        e = mean(data(3,:));                 %5
        f = std(data(3,:)) / sqrt(Nrepeats); %6
    
        g = mean(data(4,:));                 %7
        h = std(data(4,:)) / sqrt(Nrepeats); %8
    
        all_results(rank,:) = [a,b,c,d,e,f,g,h]; %Order: crmcor | err | crmcon | err |ccacorr | err | ccacon | err
    end

    M(noisecounter,:) = all_results(:,7); %CCA constrain
    N(noisecounter,:) = all_results(:,3); %CRM constrain

    noisecounter = noisecounter + 1;
end

%%

figure(1),clf;

subplot(1,2,1), hold on;
imagesc(1:8, 0.25:0.25:3, M)
title("CCA con")
xlim([1,8])
ylim([0.25,3.1])
xlabel("Rank of nuisance variable")
set(gca, 'YDir', 'normal')
colorbar;
ylabel("Noise")
clim([0,0.05])
title("Overfitting of CCA")
set(gca, 'tickdir','out');
text(0, 3.1, "a", 'FontSize', 21)

subplot(1,2,2), hold on;
imagesc(1:8, 0.25:0.25:3, N)
title("CRM con")
set(gca, 'YDir', 'normal')
colorbar;
xlabel("Rank of nuisance variable")
title("Overfitting of CCA")
xlim([1,8])
ylim([0.25,3.1])
clim([0,0.00001])
ylabel("Noise")
set(gca, 'tickdir','out');
text(0, 3.1, "b", 'FontSize', 21)

save("CRMcrossval.mat")
%exportgraphics(figure(1), 'suppl_overfitting.pdf');

toc