
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

%% Test of analytical solution.

%w must be an eivenvector to Dxy inv(Cyy)(Cxy+l3 Dxy)' w

[w_x, w_y, lbd3i] = compute_weights(C_xx, C_yy, C_xy, D_xy, 20);

disp([w_x'*C_xy*w_y, w_x'*C_xx*w_x, w_y'*C_yy*w_y, w_x'*D_xy*w_y])

%%
%w_x'*(D_xy*inv(C_yy)*(C_xy + lbd3i*D_xy)'*w_x)
% Note that w_x is normed such that w_x'*C_xx*w_x =1

M = inv(C_xx)*C_xy*inv(C_yy)* ((C_xy-D_xy*inv(C_yy)*(C_xy'))');
[W,D] = eig(M);
w_xA = W(:,1);
w_xA = w_xA./sqrt(w_xA'*C_xx*w_xA);

foo = @(lbd3i) w_xA'*(D_xy*inv(C_yy)*(C_xy + lbd3i*D_xy)'*w_xA);
lam = fzero(foo, 1);
w_yA = inv(C_yy)*(C_xy + lam*D_xy)'*w_xA;
w_yA = w_yA./sqrt(w_yA'*C_yy*w_yA);

disp([w_xA'*C_xy*w_yA, w_xA'*C_xx*w_xA, w_yA'*C_yy*w_yA, w_xA'*D_xy*w_yA])
cv

%% Testplot

lbd3i = 1;
data  = [];
data2 = [];
for lbd3i = -1:0.01:1
    fun = @(a) det( inv(C_xx)*(C_xy+lbd3i*D_xy)*inv(C_yy)* ((C_xy+lbd3i*D_xy)') - a*diag(19,19));
    best_a = fzero(fun,-1);
    data = [data; best_a];
    disp(fun(best_a))
end


plot(-1:0.01:1, data, '-o')