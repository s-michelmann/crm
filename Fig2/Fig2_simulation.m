close all;
clear all;

tic % around 3h use

rng(1)

figure(1), clf;

Nobservations = 10000;
Nsubjects = 10;
Nrepeats = 10; % Averaging For the plot

for noise = 0.05:0.5:10
    data = zeros(4,Nrepeats);
    for repeat = 1:Nrepeats
        X = randn(Nsubjects, Nobservations); 

        M = randn(Nsubjects,Nsubjects);

        Y = M*X + noise*randn(Nsubjects, Nobservations); % Datasets so that X and Y are correlated + corrupted by noise

        C_xy = X*Y';
        C_xx = X*X';
        C_yy = Y*Y';
        
        S = randn(Nsubjects, Nobservations);  % Same brain areas in different experiment. Just noise
        T = (0.25.*randn(Nsubjects,Nsubjects) + 0.75.*M) * S;

        D_xy = S*T';
        
        [r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, D_xy);
        
        w_x = r_wx(:,1); % This is the best weight vector
        w_y = r_wy(:,1);

        data(1, repeat) = (w_x'*C_xy*w_y).^2;
        data(2, repeat) = (w_x'*D_xy*w_y).^2;

        [r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, 0*D_xy);
        w_x = r_wx(:,1);
        w_y = r_wy(:,1);
        data(3, repeat) = (w_x'*C_xy*w_y).^2;
        data(4, repeat) = (w_x'*D_xy*w_y).^2;
    end

    figure(1)

    subplot(3,3,1), hold on;
    mu = mean(data(1,:));
    err = std(data(1,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'o', 'MarkerFaceColor', "#0072BD", 'Color',"#0072BD") % CRM result for best corr (max)
    mu = mean(data(3,:));
    err = std(data(3,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'o', 'MarkerFaceColor', "#77AC30", 'Color',"#77AC30") % CCA zero for best corr

    subplot(3,3,2), hold on;
    mu = mean(data(2,:));
    err = std(data(2,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'o', 'MarkerFaceColor', "#0072BD", 'Color',"#0072BD") % CRM result for D (=0)
    mu = mean(data(4,:));
    err = std(data(4,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'o', 'MarkerFaceColor', "#77AC30", 'Color',"#77AC30") % CCA result for D (not constrained)
end

%%
subplot(3,3,1)
ylim([-0.1,1.1])
xlim([-0.1, 10.1])
xlabel("Noise")
ylabel("Correlation w_xC_{xy}w_y")
set(gca, 'tickdir','out');
legend('CRM', 'CCA', 'Location','northeast')
text(-2.5, 1.1, "a", 'FontSize', 21)

subplot(3,3,2)
ylim([-0.1,0.7])
xlim([-0.1, 10.1])
xlabel("Noise")
ylabel("Constraint w_xD_{xy}w_y")
set(gca, 'tickdir','out');
legend('CRM', 'CCA', 'Location','northeast')
text(-2.8, 0.7, "b", 'FontSize', 21)


[r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, D_xy);

%%

% Add plot of the roots
for lbd3 = (min(r_lam)-1): 0.01: (max(r_lam)+1)
    M = inv(C_xx)*(C_xy+lbd3*D_xy)*inv(C_yy)* ((C_xy+lbd3*D_xy)');
    [W,eigvalues] = eig(M, 'vector');

    constraint = zeros(Nsubjects,1);    
    correlation = zeros(Nsubjects,1);

    for i = 1:Nsubjects
        w_x = W(:,i);
        w_x = w_x./sqrt(w_x'*C_xx*w_x);
        w_y = inv(C_yy)*(C_xy+lbd3*D_xy)'*w_x ./ sqrt(eigvalues(i));
        constraint(i) = w_x'*D_xy*w_y;
        correlation(i) = w_x'*C_xy*w_y;
    end
    
    subplot(3,3,3)
    scatter(lbd3*ones(Nsubjects,1), constraint, 10, correlation, "filled")
    hold on;
end

subplot(3,3,3)
plot(r_lam, 0, 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor','r')
plot([min(r_lam)-1,max(r_lam)+1], [0,0], 'k--')
xlabel("\lambda_3")
ylabel("Constraint w_xD_{xy}w_y")
a=colorbar;
a.Label.String = 'Correlation w_xC_{xy}w_y';
caxis([-1,1])
xlim([ min(r_lam)-1, max(r_lam)+1])
set(gca, 'tickdir','out');
text(-4.5,0.4, "c", 'FontSize', 21)

%% Lower three plots
T = 2000; % Total number of observations.
N = 100;  % Total variate number, eg. recorded EEG channels
t = 1:T;

X = randn(N, T);
Y = randn(N, T);

hidden_signal = 2*(sin(t/100)>0) - 1; % Representation of interest.
for i = 1:10:N
    X(i, :) = X(i, :) + hidden_signal;
    Y(i, :) = Y(i, :) + hidden_signal;
end

A0 = 2; % Dominating signal over the representation of interest.
for i = 1:N
    X(i,:) = X(i,:) + A0*sin(t/10);
    Y(i,:) = Y(i,:) + A0*sin(t/10);
end

fs = 1;
Xfilt = zeros(N, T);
Yfilt = zeros(N, T);
for i = 1:N
    Xfilt(i,:) = bandpass(X(i,:), [0.015 0.020],fs);
    Yfilt(i,:) = bandpass(Y(i,:), [0.015 0.020],fs);
end

%% Perform the analysis
C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';
D_xy = Xfilt*Yfilt';

[r_wxCCA, r_wyCCA, r_lamCCA, wxcxywyCCA, wxdxywyCCA, wxcxxwxCCA, wycyywyCCA] = compute_weights_full(C_xx, C_yy, C_xy, 0*C_xy);

[r_wxCRM, r_wyCRM, r_lamCRM, wxcxywyCRM, wxdxywyCRM, wxcxxwxCRM, wycyywyCRM] = compute_weights_full(C_xx, C_yy, C_xy, D_xy);

%%
w_xCCA = r_wxCCA(:,1);
w_xCRM = r_wxCRM(:,1);

w_yCCA = r_wyCCA(:,1);
w_yCRM = r_wyCRM(:,1);

subplot(3,3,4)
imagesc(X)
title("Data A")
ylabel("Variate")
a=colorbar;
caxis([-6,6])
xlim([0,1000])
a.Label.String = 'Signal [a.u.]';
xlabel("Obseration")
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')
text(-300,-10, "d", 'FontSize', 21)

subplot(3,3,5)
imagesc(Y)
title("Data  B")
ylabel("Variate number")
b=colorbar;
caxis([-6,6])
xlim([0,1000])
b.Label.String = 'Signal [a.u.]';
xlabel("Observation")
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')
text(-350,-10, "e", 'FontSize', 21)

subplot(3,3,6)
plot(w_xCCA*1e4,'-o')
hold on
plot(w_yCCA*1e4,'-o')
plot([0,N],[0,0],'k--')
xlabel("Variate number")
ylabel("CCA weight")
set(gca, 'tickdir','out');
ylim([-3, 3])
legend('Data A', 'Data B', 'Location','northeast')
ax = gca
box(ax,'off')
text(-20,3.6, "f", 'FontSize', 21)

subplot(3,3,7)
plot(X'*w_xCCA*10)
hold on
plot(Y'*w_yCCA*10)
ylabel("CCA component")
xlabel("Observation number")
set(gca, 'tickdir','out');
plot(-hidden_signal*std(X'*w_xCRM)*10,'k-.', 'LineWidth', 2)
ylim([-0.5, 0.8])
xlim([0,1000])
legend('Data A', 'Data B', 'Representation of interest', 'Location','northwest')
ax = gca
box(ax,'off')
text(-250,0.82, "g", 'FontSize', 21)

subplot(3,3,8)
plot(w_xCRM*1e3, '-o')
hold on
plot(w_yCRM*1e3, '-o')
plot([0,N],[0,0],'k--')
xlabel("Variate number")
ylabel("CRM weight")
set(gca, 'tickdir','out');
legend('Data A', 'Data B', 'Location','northeast')
ylim([-4, 4])
ax = gca
box(ax,'off')
text(-27,4.3, "h", 'FontSize', 21)

subplot(3,3,9)
plot(X'*w_xCRM*10)
hold on
plot(Y'*w_yCRM*10)
ylabel("CRM Component")
xlabel("Observation number")
set(gca, 'tickdir','out');
plot(-hidden_signal*std(X'*w_xCRM)*10,'k-.', 'LineWidth', 2)
ylim([-0.5, 0.8])
xlim([0,1000])
ax = gca
box(ax,'off')
legend('Data A', 'Data B', 'Representation of interest', 'Location','northwest')
text(-200,0.8, "i", 'FontSize', 21)


%exportgraphics(figure(1), 'fig2.pdf');

toc