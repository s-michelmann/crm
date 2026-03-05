clear all; close all;

%% Simulate data with a
% - shared non-stationary 60Hz hum
% - cross-frequency amplitude coupled signals at 10 Hz and 70 Hz.
% - noise background.

fs = 500;              % Sampling frequency (Hz)
T = 300;                % Duration (seconds)
t = (0:1/fs:T-1/fs)';   % Time vector
f_noise = 60;           % The "contaminant" frequency
n_channels = 2;         % Number of signals

raw_noise = randn(length(t), n_channels);
[b, a] = butter(2, 100/(fs/2)); % soft 100 Hz low-pass for backgrpund.
background = filter(b, a, raw_noise);
background = background ./ std(background);

modulation = randn(length(t), 1);
[b, a] = butter(4, 1/(fs/2)); % 1Hz low-pass filter for cross-frequency signal
shared_signal_amplitude = abs(filter(b, a, modulation)).^2;

hum = sin(2*pi*f_noise*t); % Hum
hum = hum ./ std(hum);

for i = 1:n_channels
    f_signal = 10+70*(i-1);
    signal = shared_signal_amplitude.*sin(2*pi*f_signal*t);
    signal = signal ./ std(signal);
    simulated_signals(:,i) = 3*background(:,i) + signal + hum.*(t < T/2);
end


% Clean data using Global Regression
% X = [sin(2*pi*f_noise*t), cos(2*pi*f_noise*t), ones(size(t))];
% beta = X \ simulated_signals;
% fit = X * beta;
% cleaned_signals = simulated_signals - fit;

% Plot to make sure that it looks good
nff = 512;
nsc = 100; % chunks for spectrum 
nov = floor(nsc/2);  

x = simulated_signals(:,1);
x = (x-mean(x))./std(x);
[sx,f,t] = spectrogram(x,hamming(nsc, 'periodic'),nov,nff,fs);
X = abs(sx).^2;
X = X(f<100,:);

y = simulated_signals(:,2);
y = (y-mean(y))./std(y);
[sy,f,t] = spectrogram(y,hamming(nsc, 'periodic'),nov,nff,fs);
Y = abs(sy).^2;
Y = Y(f<100,:);
f = f(f<100);

figure(1)
subplot(2,1,1)
title("Spectrogram X")
imagesc(t,f,X)

subplot(2,1,2)
title("Spectrogram Y")
imagesc(t,f,Y)

corrcoef(X(8,:), Y(80,:)) % double check of aplitude coupling -> two bands are correlated. All good.

%% perform CRM

C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';

[w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, 0*C_xy, f=1, gamma=0, chlsky=false);

w_x'*(C_xy*w_y)


figure(2),clf;
plot(f, w_x, 'r')
hold on;
plot(f, w_y, 'b')
title("Expected: peaks at 10, 60 and 70")

