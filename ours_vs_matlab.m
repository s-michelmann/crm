
Nobservations = 10000;
Nsubjects = 20;
noise = 0.1;
X = randn(Nsubjects, Nobservations); 
M = randn(Nsubjects,Nsubjects);
Y = M*X + noise*randn(Nsubjects, Nobservations); % Datasets so that X and Y are correlated + corrupted by noise

C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';
        
S = randn(Nsubjects, Nobservations);  % Same brain areas in different experiment. Just noise
T = randn(Nsubjects,Nsubjects)*S + 0.5*randn(Nsubjects, Nobservations);
D_xy = S*T';
[w_x, w_y, lbd3i] = compute_weights(C_xx, C_yy, C_xy, 0, 1);


[A,B] = canoncorr(X',Y'); % Matlab

figure(1)
subplot(1,2,1)
scatter(A(:,Nsubjects), w_x);
xlabel("Matlab, weights on x")
ylabel("Our results, weights on x")

subplot(1,2,2)
scatter(B(:,Nsubjects), w_y);
xlabel("Matlab, weights on y")
ylabel("Our results, weights on y")
