

% X is a matrix that is features x observations x samples
% crmXXtrain computes the vectors Vx and Vy such that the correlation of
% (Vx times X) with (X times Vy) is maximal across observations and samples
% of the same content AND the correlation of (Vx times X) with (X times
% Vy)is minimal aross observations and samples of different labels
[Vx, Vy, lbd3] = crmXXtrain(X, labels, s1, s2, 20);

% X is a matrix that is features x observations x samples
% crmXYtrain computes the vectors Vx and Vy such that the correlation of
% (Vx times X) with (Y times Vy) is maximal across observations and samples
% of the same content AND the correlation of (Vx times X) with (Y times
% Vy)is minimal aross observations and samples of different labels
[Vx, Vy, lbd3] = crmXYtrain(X, Y, labelsx, labelsy,s1, s2(4:end));


% e.g. samples could be the first 200ms of an EEG recording. 

% this applies the vectors to some held out data and computes the
% correlation across observations and samples for instances of same content
% and instances of different content
[corrsame, corrdiff] = crmXXtest(X, labels, s1, s2, 20,Vx, Vy);

[corrsame, corrdiff] = crmXYtest(X, Y, ...
    labelsx, labelsy,s1, s2, 20, Vx, Vy);