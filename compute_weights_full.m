function [r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, D_xy)
% Returns all the lbd3 that it can find.

% Scan through range of lambda3 to get candidate solutions
lbd3_range = -10:0.001:10;
constraint = zeros(length(C_xx),length(lbd3_range));

list_lbd3s = [];
list_wx = [];
list_wy = [];
list_constraint1 = [];
list_constraint2 = [];
list_constraint3 = [];

list_correlation = [];

f = waitbar(0, 'Finding roots');

for lbd3i = 1:length(lbd3_range)
    waitbar(lbd3i/length(lbd3_range), f, 'Finding roots');

    lbd3 = lbd3_range(lbd3i);

    M = inv(C_xx)*(C_xy+lbd3*D_xy)*inv(C_yy)*((C_xy+lbd3*D_xy)');
    [W, eigvalues] = eig(M, 'vector');

    for i = 1:length(C_xx)
        w_x = W(:,i);
        w_x = w_x./sqrt(w_x'*C_xx*w_x);
        w_y = inv(C_yy)*(C_xy+lbd3*D_xy)'*w_x ./ sqrt(eigvalues(i));
        constraint(i, lbd3i) = w_x'*D_xy*w_y;
        if lbd3i>1
            flip = (sign(constraint(i, lbd3i)) * sign(constraint(i, lbd3i-1)) <0);
            dif = abs(abs(constraint(i, lbd3i)) - constraint(i, lbd3i-1)) / mean(diff(lbd3_range));
            if flip & (dif<10)
                list_lbd3s = [list_lbd3s, lbd3];
                list_wx = [list_wx, w_x];
                list_wy = [list_wy, w_y];
                list_constraint1 = [list_constraint1, w_x'*D_xy*w_y ];  % 0
                list_constraint2 = [list_constraint2, w_x'*C_xx*w_x ];  % 1
                list_constraint3 = [list_constraint3, w_y'*C_yy*w_y ];  % 1
                list_correlation = [list_correlation, w_x'*C_xy*w_y ];  % max
            end
        end
    end
end

% Sort them by correlation
[B,I] = sort(list_correlation, 'descend');

r_lam  = list_lbd3s(I);
r_wx   = list_wx(:,I);
r_wy   = list_wy(:,I);
wxcxywy = list_correlation(I);
wxdxywy = list_constraint1(I);
wxcxxwx = list_constraint2(I);
wycyywy = list_constraint2(I);

end