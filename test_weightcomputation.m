figure(1), clf;


Nobservations = 1000;
Nsubjects = 20;
Nrepeats = 20; % Averaging For the plot

for noise = 0:0.1:2
    data = zeros(3,Nrepeats);
    for repeat = 1:Nrepeats
        time_series = randn(Nsubjects, Nobservations); 
        M = randn(Nsubjects,Nsubjects);
        X = time_series + noise*randn(Nsubjects, Nobservations);
        Y = M*time_series + noise*randn(Nsubjects, Nobservations); % Datasets so that X and Y are correlated + corrupted by noise
        C_xy = X*Y';
        C_xx = X*X';
        C_yy = Y*Y';
        
        S = randn(Nsubjects, Nobservations);  % Same 10 brain areas in different experiment. Just noise
        T = randn(Nsubjects, Nobservations);
        D_xy = S*T';
        
        f = 3;
        [w_x, w_y, lbd3i] = compute_weights(C_xx,C_yy,C_xy, D_xy,f);
        
        bestcc = corrcoef(inv(M)*Y, X);
        data(1, repeat) = bestcc(2); % best case scenario
        data(2, repeat) = (w_x'*C_xy*w_y).^2; %should be as close to "1" as possible
        data(3, repeat) = (w_x'*D_xy*w_y).^2; %should be as close to "0" as possible;
    end

    figure(1), hold on;
    mu = mean(data(1,:));
    err = std(data(1,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'bo')
    mu = mean(data(2,:));
    err = std(data(2,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'ro')
    mu = mean(data(3,:));
    err = std(data(3,:)) / sqrt(Nrepeats);
    errorbar(noise, mu, err, 'go')
end

xlabel("Noise")
ylabel("True corr (blue); CRM (red); CRM (green)")
ylim([-0.1,1])
xlim([0,2])
