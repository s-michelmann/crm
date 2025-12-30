
close all;
clear all;

load('./E65.mat') % a struct with data & behavior

trials_use = [9,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,53,55,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,117,118,119,120,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210];

behavioralVariables = nic_output.behavioralVariables;
trialn = behavioralVariables.Trial;
 
ROIactivities = nic_output.ROIactivities;
[T,N] = size(ROIactivities);
neural_data = double(ROIactivities( ismember(trialn, trials_use), :));

%% Set rng() and load data
rng(1);

Datarange    = nansum(neural_data,2)>0;
Neurons      = nansum(neural_data,1)>5;
neural_data  = neural_data(Datarange,Neurons)';

for i = 1:sum(Neurons)
    neural_data(i,:) = (neural_data(i,:) - mean(neural_data(i,:))) ./ std(neural_data(i,:));
end

behavioral_data = [behavioralVariables.Position(Datarange), ...
    behavioralVariables.Position_X(Datarange), ...
    behavioralVariables.Velocity(Datarange), ...
    behavioralVariables.Yvelocity(Datarange), ...
    behavioralVariables.Xvelocity(Datarange), ...
    behavioralVariables.Evidence(Datarange), ...
    behavioralVariables.Time(Datarange), ...
    behavioralVariables.ViewAngle(Datarange),...
    behavioralVariables.Choice(Datarange)]';

for i = 1:9
    behavioral_data(i,:) = (behavioral_data(i,:) - mean(behavioral_data(i,:)))./ std(behavioral_data(i,:));
end

%% Run CRM
% Idea1: task1 behavior vs neurons VS task2 behavior v neurons
% Idea2: correct behavior vs neurons VS error task v neurons

flag = behavioralVariables.ChoiceCorrect(Datarange) == 1;

X = neural_data(:,flag); % good trials
Y = behavioral_data(:, flag);

S = neural_data(:, ~flag); % error trials
T = behavioral_data(:, ~flag);
 
C_xy = X*Y';
C_xx = X*X';
C_yy = Y*Y';
D_xy = S*T';

%% warning('off','MATLAB:singularMatrix')

tic
[w_x, w_y, lambda3] = compute_weights(C_xx, C_yy,C_xy, D_xy);
toc

%% analysis

posy = behavioral_data(1,:);
posx = behavioral_data(2,:);
evi  = behavioral_data(6,:);
tri = behavioralVariables.Trial(Datarange);
corrcoef(w_x'*X, w_y'*Y)
corrcoef(w_x'*S, w_y'*T)

figure(1)
rx1 = mean(behavioralVariables.Position_X(Datarange));
rx2 = std(behavioralVariables.Position_X(Datarange));
ry1 = mean(behavioralVariables.Position(Datarange));
ry2 = std(behavioralVariables.Position(Datarange));

scatter(posy*ry2 + ry1, posx*rx2 + rx1, [], w_x'*neural_data)
clim([-0.01,0.01])
xlabel("Position of the mouse in cm")
ylabel("Position of the mouse in cm")
% subplot(2,1,2)
% for t = unique(tri)'
%     acc = w_x'*neural_data;
%     plot(posy(tri==t), acc(tri==t))
%     hold on;
% end

% last: Retina P vs movie, controlling for P v luminance