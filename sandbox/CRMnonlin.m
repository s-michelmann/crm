
close all;
clear all;

load('./E65.mat') % a struct with data & behavior

trials_use = [9,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,53,55,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,117,118,119,120,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210];

behavioralVariables = nic_output.behavioralVariables;
trialn = behavioralVariables.Trial;
 
ROIactivities = nic_output.ROIactivities;
[T,N] = size(ROIactivities);
neural_data = double(ROIactivities( ismember(trialn, trials_use), :));

%% z scoring
for i = 1:375
    neural_data(:,i) = (neural_data(:,i)) ./ std(neural_data(:,i));
end

%% Set rng() and load data
rng(1);

Datarange    = nansum(neural_data,2)>0;
Neurons      = nansum(neural_data,1)>0;
neural_data        = neural_data(Datarange,Neurons)';

behavioral_data = [behavioralVariables.Position(Datarange), behavioralVariables.Position_X(Datarange), behavioralVariables.Velocity(Datarange), behavioralVariables.Yvelocity(Datarange), behavioralVariables.Xvelocity(Datarange), behavioralVariables.Evidence(Datarange), behavioralVariables.Choice(Datarange), behavioralVariables.ViewAngle(Datarange)]';


for i = 1:8
    behavioral_data(:,i) = (behavioral_data(:,i)) ./ std(behavioral_data(:,i));
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

% warning('off','MATLAB:singularMatrix')

tic
[r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, D_xy);
toc

%% analysis

w_x = r_wx(:,1);
w_y = r_wy(:,1);

corrcoef(w_x'*X, w_y'*Y)

% evi = behavioralVariables.Evidence(Datarange);
% behavioralVariables.PriorChoice(Datarange)


% last: Retina P vs movie, controlling for P v luminance