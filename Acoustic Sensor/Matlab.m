clc;
clear all;
close all;

opt = nlgreyestOptions;
opt.Display = 'on';
opt.SearchOptions.MaxIterations = 500;
% opt.SearchMethod = 'grad';

%% Model configuration
FileName     = 'Acoustic_sensor_model';               
Order        = [2 1 4];                     % Dimension of input, output and state [ny nu nx].
% Parameters   = [0, 0, 0, 0, 0, 0]';                    % Initial guess
Parameters   = [-6, -6, 1.8, 6, -0.8, 1.2]';                    % Para with 20% error
% Parameters   = [-5, -5, 2, 5, -1, 1]';                    % real Para
InitialStates = zeros(4, 1);                % Initial state
Ts            = 0;                          % Time-continuous system.

nlgr = idnlgrey(FileName, Order, Parameters, InitialStates, Ts); 
nlgr = setpar(nlgr,'Name',{'k1';'k2';'k3';'k4';'k5';'k6'});

% nlgr.SimulationOptions.Solver = 'ode45';  % Solver selected
% nlgr.SimulationOptions.FixedStep = 0.1; % Fixed step


%% Import data
dataset = load('data\ChirpStimulation.mat').dataset;
length = 500;
u = dataset(1:length, 1);
y = dataset(1:length, 2:3);

% y = dataset(1:length-1, 2:3);
% y = [0, 0; y];

z = iddata(y, u, 0.1, 'Name', 'Actual Data'); 
z.Tstart = 0;
z.TimeUnit = 's'; % Unit of time

% present(z);


%% Grey box identification
compare(z,nlgr); % Comparation before training
nlgr = nlgreyest(z,nlgr,nlgreyestOptions('Display','on'),opt);
figure;
compare(z,nlgr); % Comparation after training


%% save trajectory and calculate RMSE/VAF
z = iddata([], dataset(1:1000, 1), 0.1);
y_sim = sim(nlgr, z, InitialStates);
out = y_sim.OutputData;
% save('Microphone_matlab_pred.mat', 'out'); 
figure;
plot(1:1000, [y_sim.OutputData(:,1), dataset(1:1000, 2)]);
figure;
plot(1:1000, [y_sim.OutputData(:,2), dataset(1:1000, 3)]);

% train
w_pre = out(1:500,1);
pitch_pre = out(1:500,2);
w = dataset(1:500, 2);
pitch = dataset(1:500, 3);
trian_y_RMSE = norm(w-w_pre)/sqrt(size(w, 1));
trian_c_RMSE = norm(pitch-pitch_pre)/sqrt(size(pitch, 1));
trian_y_VAF = 1 - norm(w-w_pre)^2/norm(w)^2;
trian_c_VAF = 1 - norm(pitch-pitch_pre)^2/norm(pitch)^2;

% test
w_pre = out(501:1000,1);
pitch_pre = out(501:1000,2);
w = dataset(501:1000, 2);
pitch = dataset(501:1000, 3);
Test_y_RMSE = norm(w-w_pre)/sqrt(size(w, 1));
Test_c_RMSE = norm(pitch-pitch_pre)/sqrt(size(pitch, 1));
Test_y_VAF = 1 - norm(w-w_pre)^2/norm(w)^2;
Test_c_VAF = 1 - norm(pitch-pitch_pre)^2/norm(pitch)^2;


%% Estimated parameter
L = 1/nlgr.Parameters(6).Value;
m = 1/nlgr.Parameters(4).Value;
B = -m*nlgr.Parameters(1).Value;
m = m;
K = -m*nlgr.Parameters(2).Value;
e_A = 1/nlgr.Parameters(3).Value;
R = -L*nlgr.Parameters(5).Value;
L = 1/nlgr.Parameters(6).Value;

