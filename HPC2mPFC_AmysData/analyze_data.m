% Data citation: https://www.jneurosci.org/content/45/10/e1241242025 Rosenblum et al. 2025


%close all;
clear all;

%% Load data + Make spectra

nff = 1024;
fs = 2000;
nsc = 2000; % 1s chunks for spectrum
nov = floor(nsc/2);

load('/Users/ms81/Desktop/PFC_black.mat')
y = Samples(:);
y = (y-mean(y))./std(y);
[sy,f,t] = spectrogram(y,hamming(nsc, 'periodic'),nov,nff,fs);
Y = log(abs(sy).^2);

load('/Users/ms81/Desktop/HPC_red.mat')
x = Samples(:);
x = (x-mean(x))./std(x);
[sx,f,t] = spectrogram(x,hamming(nsc, 'periodic'),nov,nff,fs);
X = log(abs(sx).^2);

X = X(f<100,:);
Y = Y(f<100,:);
f = f(f<100);

%%
C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';

[w_xCCA, w_yCCA, ~] = compute_weights(C_xx,C_yy,C_xy, 0*C_xy,1);

%% use CRM to denoise
sfilt = 2;
ffilt = 59.6; % Center noise

filter = exp( - (f-ffilt).^2 ./ (2*sfilt*sfilt))/(sfilt*sqrt(2*pi));
T = filter.*Y; % Only Y has 60 Hz hum.
S = X;  

D_xy = S*T';
[w_xCRM, w_yCRM, ~] = compute_weights(C_xx,C_yy,C_xy, D_xy, 1);

%%
figure(1),clf;
subplot(2,3,1)
imagesc(t, f, X)
title("Spectrogram of HPC")
ylabel("Frequency")
xlabel("Time")
set(gca, 'tickdir','out');
text(-100,-10, "A", 'FontSize', 16)

subplot(2,3,4)
imagesc(t, f, Y)
title("Spectrogram of mPFC")
xlabel("Time")
ylabel("Frequency")
set(gca, 'tickdir','out');
text(-100,-10, "B", 'FontSize', 16)

subplot(2,3,2)
plot(f, w_xCCA, 'o-')
hold on
plot(f, w_yCCA, 'o-')
plot([0,110], [0,0],'k--')
title("Canonical Vectors")
xlabel("Frequency")
ylabel("Weight")
set(gca, 'tickdir','out');
%text(-10,6e-3, "C", 'FontSize', 16)

subplot(2,3,3)
plot(X'*w_xCCA)
hold on
plot(Y'*w_yCCA)
title("CCA Components")
xlabel("Time")
set(gca, 'tickdir','out');
%text(-200,0.12, "D", 'FontSize', 16)

subplot(2,3,5)
plot(f,w_xCRM, 'o-')
hold on
plot(f,w_yCRM, 'o-')
plot([0,110],[0,0],'k--')
title("CRM Vectors")
xlabel("Frequency")
ylabel("Weight")
set(gca, 'tickdir','out');
%text(-10,6e-3, "E", 'FontSize', 16)

subplot(2,3,6)
plot(X'*w_xCRM)
hold on
plot(Y'*w_yCRM)
title("CRM Components")
xlabel("Time")
set(gca, 'tickdir','out');
%text(-200, 0.12, "F", 'FontSize', 16)
%exportgraphics(figure(1), 'simulation2.pdf');

disp('- CCA/CRM -')
corrcoef(X'*w_xCCA, Y'*w_yCCA)
corrcoef(X'*w_xCRM, Y'*w_yCRM)
disp('------')

%% Explicit control for theta coupling

xTheta = bandfilt(x, 9, 7, fs);

yTheta = bandfilt(y, 9, 7, fs);

HPCsig = log(abs(hilbert(xTheta)).^2);
PFCsig = log(abs(hilbert(yTheta)).^2);
corrcoef(HPCsig, PFCsig)

function [output] = bandfilt(x, flp, fhp, fs)

    [b,a] = butter(5,flp/(fs/2)); % flp Hz lowpass
    x_filt = filter(b,a,x);
    
    [b,a] = butter(5,fhp/(fs/2), 'high'); % fhp Hz highpass
    output = filter(b,a,x_filt);

end
