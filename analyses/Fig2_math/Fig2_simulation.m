close all; clear all;

tic

rng(1)

figure(1), clf;

T = 2000; % Total number of observations.
N = 100;  % Total variate number, eg. recorded EEG channels
t = 1:T;

fnoise = 0.1;      % noise signal

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

[w_xCCA, w_yCCA, ~] = crm(C_xx, C_yy, C_xy, 0*D_xy);
[w_xCRM, w_yCRM, ~] = crm(C_xx, C_yy, C_xy, D_xy);

%%

figure(1),clf;
subplot(2,3,1)
imagesc(X)
title("Data A")
ylabel("Variate")
a=colorbar;
caxis([-6,6])
xlim([0,1000])
a.Label.String = 'Signal [a.u.]';
xlabel("Observation")
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')

subplot(2,3,2)
plot(w_xCCA*1e4,'-')
hold on
plot(w_yCCA*1e4,'-')
plot([0,N],[0,0],'k--')
xlabel("Variate number")
ylabel("CCA weight")
set(gca, 'tickdir','out');
ylim([-5, 5])
legend('Data A', 'Data B', 'Location','northeast')
ax = gca
box(ax,'off')

subplot(2,3,3)
plot(-w_xCRM*1e3, '-')
hold on
plot(-w_yCRM*1e3, '-')
plot([0,N],[0,0],'k--')
xlabel("Variate number")
ylabel("CRM weight")
set(gca, 'tickdir','out');
legend('Data A', 'Data B', 'Location','northeast')
ylim([-4, 4])
ax = gca
box(ax,'off')

subplot(2,3,4)
imagesc(Y)
title("Data  B")
ylabel("Variate")
b=colorbar;
caxis([-6,6])
xlim([0,1000])
b.Label.String = 'Signal [a.u.]';
xlabel("Observation")
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')

subplot(2,3,5)
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

subplot(2,3,6)
plot(X'*w_xCRM*10)
hold on
plot(Y'*w_yCRM*10)
ylabel("CRM Component")
xlabel("Observation number")
set(gca, 'tickdir','out');
plot(-hidden_signal*std(X'*w_xCRM)*10,'k-.', 'LineWidth', 2)
ylim([-0.5, 0.8])
xlim([0,1000])
ax = gca;
box(ax,'off')
legend('Data A', 'Data B', 'Representation of interest', 'Location','northwest')

%exportgraphics(figure(1), 'fig2.pdf');

toc