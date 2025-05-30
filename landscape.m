
Nsubjects = 3;
Nobservations = 50;
noise = 0.5;

X = randn(Nsubjects, Nobservations); 

M = randn(Nsubjects,Nsubjects);

Y = M*X + noise*randn(Nsubjects, Nobservations); % Datasets so that X and Y are correlated + corrupted by noise

C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';

S = randn(Nsubjects, Nobservations);  % Same brain areas in different experiment. Just noise
T = randn(Nsubjects,Nsubjects)*S + 0.5*randn(Nsubjects, Nobservations);
D_xy = S*T';

figure(1), clf;
for lbd3 = -1:0.01:1
    M = inv(C_xx)*(C_xy+lbd3*D_xy)*inv(C_yy)* ((C_xy+lbd3*D_xy)');
    [W,D] = eig(M);

    eigenvalues = diag(D);

    constraint = zeros(Nsubjects,1);
    for i = 1:Nsubjects
        w_x = W(:,i);
        w_x = w_x./sqrt(w_x'*C_xx*w_x);
        w_y = inv(C_yy)*(C_xy+lbd3*D_xy)'*w_x;
        constraint(i) = w_x'*D_xy*w_y;
    end

    subplot(1,2,1)
    scatter(lbd3*ones(Nsubjects,1), eigenvalues, 'k.')
    xlabel("lambda_3")
    ylabel("Eigenvalues)")
    hold on;

    subplot(1,2,2)
    scatter(lbd3*ones(Nsubjects,1), constraint, 'k.')
    ylabel("Constraint")
    plot([-1,1], [0,0], 'k--')
    hold on;
end


[w_x, w_y, lbd3i] = compute_weights(C_xx,C_yy,C_xy, D_xy,1);
plot(lbd3i, w_x'*D_xy*w_y, 'o', 'MarkerFaceColor','r')

[w_x, w_y, lbd3i] = compute_weights(C_xx,C_yy,C_xy, D_xy,2);
plot(lbd3i, w_x'*D_xy*w_y, 'o', 'MarkerFaceColor','g')

[w_x, w_y, lbd3i] = compute_weights(C_xx,C_yy,C_xy, D_xy,3);
plot(lbd3i, w_x'*D_xy*w_y, 'o', 'MarkerFaceColor','b')