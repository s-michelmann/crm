
rng(1)

T = 1000; % duration in time bins.
fsx = 20;
fsy = 80;
sa = 6;   % spectral width
As = 4;

fn = 60;
sn = 2;
An = 10;

A0 = 1; % background noise

% Make slowly varying signal time seris
f = As*make_timeseries(T); % signal time series.
g = An*make_timeseries(T); % noise time series at 60 Hz


X = zeros(100,T);
Y = zeros(100,T);

for fr = 1:100        % Frequency bins
    for t = 1:T      % recordings
        X(fr,t) = A0*randn(1);
        X(fr,t) = X(fr,t) + f(t) * exp( - (fr-fsx).^2 ./ (2*sa*sa));
        X(fr,t) = X(fr,t) + g(t) * exp( - (fr-fn).^2 ./ (2*sn*sn));

        Y(fr,t) = A0*randn(1);
        Y(fr,t) = Y(fr,t) + f(t) * exp( - (fr-fsy).^2 ./ (2*sa*sa));
        Y(fr,t) = Y(fr,t) + g(t) * exp( - (fr-fn).^2 ./ (2*sn*sn));
    end
end


%% Perform the analysis
C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';

[w_xCCA, w_yCCA, ~] = compute_weights(C_xx,C_yy,C_xy, 0*C_xy,1);

figure(1),clf;
subplot(2,2,1)
imagesc(X)
title("Spectrogram of area X")
ylabel("Frequency")
xlabel("Time")
set(gca, 'tickdir','out');
text(-100,-10, "A", 'FontSize', 16)

subplot(2,2,2)
imagesc(Y)
title("Spectrogram of area Y")
xlabel("Time")
set(gca, 'tickdir','out');
text(-100,-10, "B", 'FontSize', 16)

subplot(2,2,3)
plot(w_xCCA)
hold on
plot(w_yCCA)
title("CCA result for w_x/w_y")
xlabel("Frequency")
ylabel("Weight on frequency bin")
set(gca, 'tickdir','out');
ylim([-5e-3, 5e-3])
text(-10,6e-3, "C", 'FontSize', 16)

% use CRM to denoise
filt=zeros(100,1);
sfilt=2;
ffilt=60;
for fr = 1:100
    filt(fr) = exp( - (fr-ffilt).^2 ./ (2*sfilt*ffilt));
end
S = filt.*X;  % Same brain areas in different experiment. Just noise
T = filt.*Y;
D_xy = S*T';
[w_xCRM, w_yCRM, ~] = compute_weights(C_xx,C_yy,C_xy, D_xy,1);

subplot(2,2,4)
plot(w_xCRM)
hold on
plot(w_yCRM)
title("CRM result for w_x/w_y")
xlabel("frequency")
set(gca, 'tickdir','out');
ylim([-5e-3, 5e-3])
text(-10,6e-3, "D", 'FontSize', 16)

exportgraphics(figure(1), 'simulation2.pdf');

function y = make_timeseries(T)
    data = randn(T, 1);
    y = smoothdata(data, "gaussian", 30);
end