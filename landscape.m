rng(42)

Nsubjects = 5;
Nobservations = 100;
noise = 0.8;

X = randn(Nsubjects, Nobservations); 

M = randn(Nsubjects,Nsubjects);

Y = M*X + noise*randn(Nsubjects, Nobservations); % Datasets so that X and Y are correlated + corrupted by noise

C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';

S = randn(Nsubjects, Nobservations);  % Same brain areas in different experiment. Just noise
T = randn(Nsubjects,Nsubjects)*S + 0.5*randn(Nsubjects, Nobservations);
D_xy = S*T';

[r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, D_xy);

figure(1), clf;

for lbd3 = (min(r_lam)-1): 0.01: (max(r_lam)+1)
    M = inv(C_xx)*(C_xy+lbd3*D_xy)*inv(C_yy)* ((C_xy+lbd3*D_xy)');
    [W,eigvalues] = eig(M, 'vector');

    constraint = zeros(Nsubjects,1);    
    correlation = zeros(Nsubjects,1);

    for i = 1:Nsubjects
        w_x = W(:,i);
        w_x = w_x./sqrt(w_x'*C_xx*w_x);
        w_y = inv(C_yy)*(C_xy+lbd3*D_xy)'*w_x ./ sqrt(eigvalues(i));
        constraint(i) = w_x'*D_xy*w_y;
        correlation(i) = w_x'*C_xy*w_y;
    end
    
    figure(1)
    scatter(lbd3*ones(Nsubjects,1), constraint, [], correlation)
    hold on;
end

%Check the roots

plot(r_lam, 0, 'o', 'MarkerFaceColor', 'r')
plot([min(r_lam),max(r_lam)], [0,0], 'k-')
title(length(r_lam))
xlabel("\lambda_3")
ylabel("Constraint = wx*Dxy*wy")
a=colorbar;
a.Label.String = 'Correlation = wx*Cxy*wy';
caxis([-1,1])