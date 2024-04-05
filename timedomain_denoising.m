
rng(1)

T = 2000; % duration in time bins.
N = 100;  % number of recorded EEG channels
t = 1:T;

X = randn(N, T);
Y = randn(N, T);

hidden_signal = 2*(sin(t/100)>0) - 1;
for i = 1:10:N
    X(i, :) = X(i, :) + hidden_signal;
    Y(i, :) = Y(i, :) + hidden_signal;
end

A0 = 2; % amplitude of shared contaminating noise
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

[w_xCCA, w_yCCA, ~] = compute_weights(C_xx,C_yy,C_xy, 0*C_xy,1);
[w_xCRM, w_yCRM, ~] = compute_weights(C_xx,C_yy,C_xy, D_xy,1);

figure(2),clf;

subplot(2,3,1)
imagesc(X)
title("Data from area A")
ylabel("EEG electrode #")
xlabel("Time")
set(gca, 'tickdir','out');
text(-100,-10, "A", 'FontSize', 16)

subplot(2,3,4)
imagesc(Y)
title("Data from area B")
ylabel("EEG electrode #")
xlabel("Time")
set(gca, 'tickdir','out');
text(-100,-10, "B", 'FontSize', 16)

subplot(2,3,2)
plot(w_xCCA,'-o')
hold on
plot(w_yCCA,'-o')
plot([0,N],[0,0],'k--')
title("Canonical Vectors")
xlabel("EEG electrode #")
ylabel("Weight")
set(gca, 'tickdir','out');
ylim([-3e-4, 3e-4])
text(-10,3.6e-4, "C", 'FontSize', 16)

subplot(2,3,3)
plot(X'*w_xCCA)
hold on
plot(Y'*w_yCCA)
title("CCA Components")
xlabel("Time")
set(gca, 'tickdir','out');
plot(hidden_signal*std(X'*w_xCRM),'k--', 'LineWidth', 2)
ylim([-0.05, 0.05])
text(-200,0.061, "D", 'FontSize', 16)

subplot(2,3,5)
plot(w_xCRM, '-o')
hold on
plot(w_yCRM, '-o')
plot([0,N],[0,0],'k--')
title("CRM Vectors")
xlabel("EEG electrode #")
ylabel("Weight")
set(gca, 'tickdir','out');
ylim([-4e-3, 4e-3])
text(-10,5e-3, "E", 'FontSize', 16)

subplot(2,3,6)
plot(X'*w_xCRM)
hold on
plot(Y'*w_yCRM)
title("CRM Components")
xlabel("Time")
set(gca, 'tickdir','out');
plot(hidden_signal*std(X'*w_xCRM),'k--', 'LineWidth', 2)
ylim([-0.05, 0.05])
text(-200,0.061, "F", 'FontSize', 16)

exportgraphics(figure(2), 'simulation3.pdf');