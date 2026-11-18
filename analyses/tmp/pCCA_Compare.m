close all; clear all;
rng(1)

T = 2000; % Total number of observations.
N = 100;  % Total variate number, eg. recorded EEG channels
t = 1:T;

% Taken from Fig. 2 for comparison with pCCA
fnoise = 0.1;      % noise signal
X = randn(N, T);
Y = randn(N, T);
hidden_signal = 2*(sin(t/100)>0) - 1; % Representation of interest.
for i = 1:10:N
    X(i, :) = X(i, :) + hidden_signal; Y(i, :) = Y(i, :) + hidden_signal;
end
A0 = 2; % Dominating signal over the representation of interest.
for i = 1:N
    X(i,:) = X(i,:) + A0*sin(fnoise*t); Y(i,:) = Y(i,:) + A0*sin(fnoise*t);
end

figure(1),clf;

subplot(4,3,1) % Raw data
imagesc(X)

subplot(4,3,4) % partial CCA via fit
Xcleaned = zeros(size(X));
Xfit = zeros(size(X));
for i = 1:N
    b0 = sin(fnoise*t') \ X(i,:)';
    b1 = cos(fnoise*t') \ X(i,:)';
    Xfit(i,:) =   b0*sin(fnoise*t) + b1*cos(fnoise*t);
    Xcleaned(i,:) = X(i,:) -  b0*sin(fnoise*t) - b1*cos(fnoise*t);
end
Ycleaned = zeros(size(Y));
Yfit = zeros(size(Y));
for i = 1:N
    b0 = sin(fnoise*t') \ Y(i,:)';
    b1 = cos(fnoise*t') \ Y(i,:)';
    Yfit(i,:) =   b0*sin(fnoise*t) + b1*cos(fnoise*t);
    Ycleaned(i,:) = Y(i,:) -  b0*sin(fnoise*t) - b1*cos(fnoise*t);
end
C_xy = Xcleaned*Ycleaned';
C_xx = Xcleaned*Xcleaned';
C_yy = Ycleaned*Ycleaned';
[w_xCCAred, w_yCCAred, ~] = crm(C_xx, C_yy, C_xy, 0*C_xy);
plot(Xcleaned'*w_xCCAred)
hold on
plot(Ycleaned'*w_yCCAred)
ylabel("pCCA data-noisemodel")

subplot(4,3,7) %partial cca via cca on notch-filtered data
fs = 1;
Xfilt = zeros(N, T); Yfilt = zeros(N, T);
for i = 1:N
    Xfilt(i,:) = bandpass(X(i,:), [0.015 0.020],fs);
    Yfilt(i,:) = bandpass(Y(i,:), [0.015 0.020],fs);
end
Xbandfil = X - Xfilt;
Ybandfil = Y - Yfilt;
C_xy = Xbandfil*Ybandfil';
C_xx = Xbandfil*Xbandfil';
C_yy = Ybandfil*Ybandfil';
[w_xCCAband, w_yCCAband, ~] = crm(C_xx, C_yy, C_xy, 0*C_xy);
plot(Xbandfil'*w_xCCAband)
hold on
plot(Ybandfil'*w_yCCAband)
ylabel("pCCA notchfiltered(data)")


subplot(4,3,10) % CRM control
C_xy = X*Y'; C_xx = X*X';
C_yy = Y*Y'; D_xy = Xfilt*Yfilt';
[w_xCRM, w_yCRM, ~] = crm(C_xx, C_yy, C_xy, D_xy);
plot(X'*w_xCRM, '-')
hold on
plot(Y'*w_yCRM, '-')
plot([0,N],[0,0],'k--')
xlabel("Variate number")
ylabel("CRM")
set(gca, 'tickdir','out');
legend('Data A', 'Data B', 'Location','northeast')
ax = gca
box(ax,'off')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation with non-stationary noise
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
    X(i,:) = X(i,:) + A0*sin(fnoise*t).*(cos(t*fenvelope).^2);
    Y(i,:) = Y(i,:) + A0*sin(fnoise*t).*(cos(t*fenvelope).^2);
end

subplot(4,3,2) % Raw data
imagesc(X)

subplot(4,3,5) % partial CCA via fit
Xcleaned = zeros(size(X));
Xfit = zeros(size(X));
for i = 1:N
    b0 = sin(fnoise*t') \ X(i,:)';
    b1 = cos(fnoise*t') \ X(i,:)';
    Xfit(i,:) =   b0*sin(fnoise*t) + b1*cos(fnoise*t);
    Xcleaned(i,:) = X(i,:) -  b0*sin(fnoise*t) - b1*cos(fnoise*t);
end
Ycleaned = zeros(size(Y));
Yfit = zeros(size(Y));
for i = 1:N
    b0 = sin(fnoise*t') \ Y(i,:)';
    b1 = cos(fnoise*t') \ Y(i,:)';
    Yfit(i,:) =   b0*sin(fnoise*t) + b1*cos(fnoise*t);
    Ycleaned(i,:) = Y(i,:) -  b0*sin(fnoise*t) - b1*cos(fnoise*t);
end
C_xy = Xcleaned*Ycleaned';
C_xx = Xcleaned*Xcleaned';
C_yy = Ycleaned*Ycleaned';
[w_xCCAred, w_yCCAred, ~] = crm(C_xx, C_yy, C_xy, 0*D_xy);
plot(Xcleaned'*w_xCCAred)
hold on
plot(Ycleaned'*w_yCCAred)

subplot(4,3,8) %partial cca via cca on notch-filtered data
Xfilt = zeros(N, T);
Yfilt = zeros(N, T);
for i = 1:N
    Xfilt(i,:) = bandpass(X(i,:), [0.015 0.020],fs);
    Yfilt(i,:) = bandpass(Y(i,:), [0.015 0.020],fs);
end
Xbandfil = X - Xfilt;
Ybandfil = Y - Yfilt;
C_xy = Xbandfil*Ybandfil';
C_xx = Xbandfil*Xbandfil';
C_yy = Ybandfil*Ybandfil';
[w_xCCAband, w_yCCAband, ~] = crm(C_xx, C_yy, C_xy, 0*D_xy);
plot(Xbandfil'*w_xCCAband*10)
hold on
plot(Ybandfil'*w_yCCAband*10)

subplot(4,3,11) % CRM control
C_xy = X*Y'; C_xx = X*X';
C_yy = Y*Y'; D_xy = Xfilt*Yfilt';
[w_xCRM, w_yCRM, ~] = crm(C_xx, C_yy, C_xy, D_xy);
plot(X'*w_xCRM*1e3, '-')
hold on
plot(Y'*w_yCRM*1e3, '-')
plot([0,N],[0,0],'k--')
xlabel("Variate number")
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Non-sinusoidal noise

X = randn(N, T);
Y = randn(N, T);
hidden_signal = 2*(sin(t/100)>0) - 1; 
for i = 1:10:N
    X(i, :) = X(i, :) + hidden_signal;
    Y(i, :) = Y(i, :) + hidden_signal;
end
A0 = 2;
confound_nonsin = sign(sin(fnoise * t));
for i = 1:N
    X(i,:) = X(i,:) + A0 * confound_nonsin;
    Y(i,:) = Y(i,:) + A0 * confound_nonsin;
end

subplot(4,3,3) % Raw data
imagesc(X)

subplot(4,3,6) % partial CCA via fit
Xcleaned = zeros(size(X));
Xfit = zeros(size(X));
for i = 1:N
    b0 = sin(fnoise*t') \ X(i,:)';
    b1 = cos(fnoise*t') \ X(i,:)';
    Xfit(i,:) =   b0*sin(fnoise*t) + b1*cos(fnoise*t);
    Xcleaned(i,:) = X(i,:) -  b0*sin(fnoise*t) - b1*cos(fnoise*t);
end
Ycleaned = zeros(size(Y));
Yfit = zeros(size(Y));
for i = 1:N
    b0 = sin(fnoise*t') \ Y(i,:)';
    b1 = cos(fnoise*t') \ Y(i,:)';
    Yfit(i,:) =   b0*sin(fnoise*t) + b1*cos(fnoise*t);
    Ycleaned(i,:) = Y(i,:) -  b0*sin(fnoise*t) - b1*cos(fnoise*t);
end
C_xy = Xcleaned*Ycleaned';
C_xx = Xcleaned*Xcleaned';
C_yy = Ycleaned*Ycleaned';
[w_xCCAred, w_yCCAred, ~] = crm(C_xx, C_yy, C_xy, 0*D_xy);
plot(Xcleaned'*w_xCCAred)
hold on
plot(Ycleaned'*w_yCCAred)

subplot(4,3,9) %partial cca via cca on notch-filtered data
Xfilt = zeros(N, T);
Yfilt = zeros(N, T);
for i = 1:N
    Xfilt(i,:) = bandpass(X(i,:), [0.015 0.020],fs);
    Yfilt(i,:) = bandpass(Y(i,:), [0.015 0.020],fs);
end
Xbandfil = X - Xfilt;
Ybandfil = Y - Yfilt;
C_xy = Xbandfil*Ybandfil';
C_xx = Xbandfil*Xbandfil';
C_yy = Ybandfil*Ybandfil';
[w_xCCAband, w_yCCAband, ~] = crm(C_xx, C_yy, C_xy, 0*D_xy);
plot(Xbandfil'*w_xCCAband*10)
hold on
plot(Ybandfil'*w_yCCAband*10)

subplot(4,3,12) % CRM control
C_xy = X*Y'; C_xx = X*X';
C_yy = Y*Y'; D_xy = Xfilt*Yfilt';
[w_xCRM, w_yCRM, ~] = crm(C_xx, C_yy, C_xy, D_xy);
plot(X'*w_xCRM*1e3, '-')
hold on
plot(Y'*w_yCRM*1e3, '-')
plot([0,N],[0,0],'k--')
xlabel("Variate number")
set(gca, 'tickdir','out');
ax = gca
box(ax,'off')