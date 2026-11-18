close all; clear all;

rng(1)

figure(1), clf;

T = 2000; % Total number of observations.
N = 100;  % Total variate number, eg. recorded EEG channels
t = 1:T;

fnoise = 0.1;      % noise signal
fenvelope = 0.002; % noise envelope

X = randn(N, T);
Y = randn(N, T);

hidden_signal = 2*(sin(t/100)>0) - 1; % Representation of interest.
for i = 1:10:N
    X(i, :) = X(i, :) + hidden_signal;
    Y(i, :) = Y(i, :) + hidden_signal;
end

A0 = 2; % Dominating signal over the representation of interest.
for i = 1:N
    X(i,:) = X(i,:) + A0*sin(fnoise*t);
    Y(i,:) = Y(i,:) + A0*sin(fnoise*t);
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
[w_xCRM, w_yCRM, ~] = crm(C_xx, C_yy, C_xy, D_xy);


figure(1),clf;

subplot(2,3,1)
plot(w_xCRM, '-')
hold on
plot(w_yCRM, '-')
plot([0,N],[0,0],'k--')
xlabel("Variate number")
ylabel("CRM Weight")
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')

subplot(2,3,4)
plot(X'*w_xCRM)
hold on
plot(Y'*w_yCRM)
ylabel("CRM Component")
xlabel("Observation number")
set(gca, 'tickdir','out');
plot(-hidden_signal*std(X'*w_xCRM),'k-.', 'LineWidth', 2)
xlim([0,1000])
legend('Data A', 'Data B', 'Representation of interest', 'Location','northwest')
ax = gca
box(ax,'off')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,3,2) % Ridge Regularized CCA
[w_xCRMRidge, w_yCRMRidge, ~] = crm(C_xx, C_yy, C_xy, D_xy, gamma=20000);
plot(w_xCRMRidge,'-')
hold on
plot(w_yCRMRidge,'-')
plot([0,N],[0,0],'k--')
ylabel("RidgeCRM Weight")
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')

subplot(2,3,5)
plot(X'*w_xCRMRidge)
hold on
plot(Y'*w_yCRMRidge)
ylabel("RidgeCRM Component")
set(gca, 'tickdir','out');
plot(-hidden_signal*std(X'*w_xCRMRidge),'k-.', 'LineWidth', 2)
xlim([0,1000])
ax = gca;
box(ax,'off')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,3,3) % Sparse CCA
[w_xCRMSparse, w_yCRMSparse, ~] = crm(C_xx, C_yy, C_xy, D_xy,sparsity=0.35);
plot(w_xCRMSparse,'-')
hold on
plot(w_yCRMSparse,'-')
plot([0,N],[0,0],'k--')
xlabel("Variate number")
ylabel("Sparse CCA Weight")
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')

subplot(2,3,6)
plot(X'*w_xCRMSparse)
hold on
plot(Y'*w_yCRMSparse)
ylabel("Sparse CCA Component")
xlabel("Observation number")
set(gca, 'tickdir','out');
plot(-hidden_signal*std(X'*w_xCRMSparse),'k-.', 'LineWidth', 2)
xlim([0,1000])
ax = gca;
box(ax,'off')


figure(2)
imagesc(X)