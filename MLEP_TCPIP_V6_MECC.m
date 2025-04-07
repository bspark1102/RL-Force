% New reward 

%% Clear memory
close all; clear all; clc;
fclose('all');

%% Create MLEP instance and select Energy Plus model and weather data

ep = mlep;

% Select Energy Plus Model in .idf
ep.idfFile =('031721_Single_NoSunspace_Rooftop_MLEP_V920_MAY_NatVent_IdealLoads_AL_AR7');

% Select Weather file in .epw
ep.epwFile=('USA_NY_Albany.Intl.AP.725180_TMYx.2004-2018.epw');


%% Display Input/output table and determine run period

% % Initialize MLEP co-simulation 
ep.initialize;

% display outputTable 
outputTable = ep.outputTable;

% display inputTable, which is the objects in the "External Interface: Schedule" in .idf
inputTable = ep.inputTable;

runPeriod_Days   = 31;
runperiod_Second = 24*60*60*runPeriod_Days;
                
%% Set parameters and define variables

n_iter = 1; %number of months to run, becomes n_iter*31 in python 
t_num = 4463; %number of timesteps
windowsize = 10; %size of the window

% get forecast information
load('outdoor_temp_save_may.mat');
forecast = out_history_temperature;


a_hist = zeros(n_iter,t_num);
t_hist = zeros(n_iter,t_num);
o_hist = zeros(n_iter,t_num);
s_hist = zeros(n_iter,t_num);
r_hist = zeros(n_iter,t_num);
r1_hist = zeros(n_iter,t_num);
r2_hist = zeros(n_iter,t_num);
e_hist = zeros(n_iter,t_num);
f_hist = zeros(n_iter,t_num);
rad_hist = zeros(n_iter,t_num);

ce_hist = zeros(n_iter,t_num);
he_hist = zeros(n_iter,t_num);
et_hist = zeros(n_iter,t_num);


% episodal stuff
epe_hist = []; % energy per episode
dev_hist = [];
cumr_hist = [];
rpe_hist = [];
% flux_history = [];

isr_hist = zeros(n_iter,t_num);

% define a window for each indoor/outdoor temperature
ind_time_window = zeros(1,windowsize);
out_time_window = zeros(1,windowsize);

comfort_zone = [20 25];
comfort_counter = [];

%% Setting

%Define learning parameters
alpha = 0.65; %learning Rate
gamma = 0.98; %Discount Factor
epsilon = 0.9; %Chance for random action

%% Create State

k1 = [0;1;2]; % Temperature Trends for indoor air temperature
k3 = [0;1;2]; % Temperature Trends for outdoor air temperature
k2 = [0;1;2;3;4;5;6;7;8;9]; % Indoor Air Temperature Bins
k4 = [0;1;2;3;4;5;6;7]; % Outdoor Air Temperature Bins
k5 = [0;1]; % radiation bins
k6 = [0;1];
k7 = [0;1];

state_space = setprod(k1, k2, k3, k4, k5,k6, k7); % Create state space with the number of possible combinations of states

actions = [0 1 2 3 4 5 6 7]; % eight actions, vent and ins
n_actions = length(actions); % the length of the array of possible actions
n_states = size(state_space,1); % the length of the array of possible states
Q = rand(n_states, n_actions)-0.5; % create a Q-table dependent on the length of the array of the possible actions and the size of the array of the possible states
Q_history = zeros(n_states, n_actions, 1);


done = 0;
%%
t = tcpip('127.0.0.1', 50000);


%%


tic
fopen(t) %------------------------------0000
schedule = zeros(1,6);


% For the loop below, the idea is:
% 1) get measurements -> convert into states
% 2) determine which actions to take based on the states and execute
% 3) repeat step 1, get new states and calculate the rewards
% 4) update the Q-function(matrix)
% 5) repeat steps 1 through 4 (outer loop)
% ---you will find these numbers/steps below in the corresponding parts of
% the code


for iter=1:n_iter % outer loop
    cr_temp = 0;
    % Start MLEP co-simulation and communication

    ep.start;

    iter % print current iteration

    % Initialize the below "while" loop in MATLAB
    t_ep = 0;  % initialize timestep during energy plus simulation, in seconds, multiples of 'timestep'   


    init_flag = 1; % a flag used to determine if we are in the first trial or not

    Q_history(:,:,iter) = Q; % save Q-tables for records

    %----beginning of step 1)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ts = 1;
    % Read Output data from Energyplus at current timestep
    [y, t_ep] = ep.read; 

    br_temp = y(1,3); % get bedroom temperature, first column of output values from IDF
    ot_temp = y(1,1); % get outdoor temperature, Secon column of output values from IDF
    hg = y(1,12);
    hl = y(1,16);

    hf = hg - hl;
    rad = y(1,2);
    
    nf = normalizer(1,hf);
    nrad = normalizer(2,rad);
    
    fc = temp2foreV3(forecast, ts);

    h_enrg = y(1,6);
    c_enrg = y(1,9);
    
    isr = y(1,34);
    
    t_enrg = h_enrg+c_enrg;

    ind_time_window(1,:) = br_temp;
    out_time_window(1,:) = ot_temp;

    tot_eps_e = 0;
    
    estat = 0;
    
    if t_enrg >0
        estat = 400;
    end
    
    tsdat = [ind_time_window,out_time_window]*10;
%     sdat = 
    sdat = [tsdat, fc*10 , nf*400, estat, 0];
    
    %AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    fwrite(t, sdat, 'int16');
    %AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    
    % -----------function for getting the state goes here -------
%     state = RL_MLEP_stateconv_hvac1(ind_time_window,out_time_window,t_enrg,hf,fc, state_space);
    state = 100;
    %----end of step 1)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    aa = 0;
        
    while (t_ep) < runperiod_Second-600   % a single episode (Inner loop)
        % runs until the last timestep-1 because we are doing ep.read twice
        % per loop
        
%         state_history = [state_history, state]; % append state history
        s_hist(iter,ts) = state;
        
        % Append every last measurement to a history array
        % (Just for checking purposes)
        
        t_hist(iter,ts) = br_temp;
        o_hist(iter,ts) = ot_temp;
        f_hist(iter,ts) = hf;
        
        ce_hist(iter,ts) = h_enrg;
        he_hist(iter,ts) = c_enrg;
        et_hist(iter,ts) = t_enrg;
        
        isr_hist(iter,ts) = isr;
        
        %----beginning of step 2)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % -----------determine action (via epsilon greedy) and send to EP
        % need to write a function
        % greedy action yeilds the highest Q-value 
        % Actions will be sent here
 
%         [a,isgreedy] = greedy_action(Q, state, epsilon); % gives 1 or 2
%         e_hist(iter,ts) = isgreedy;
        
        
        prev_a = aa; % Save previous action (used in reward function)

        
        aa = fread(t, [1, 1]);  %------------------------------0000
        
%         if iter == 1
%            aa = 0;
%         elseif iter == 2
%            aa = naive_control(ot_temp,isr); 
%         end
        
        a_hist(iter,ts) = aa;

        
        schedule = act7_select6_sim(aa);

        ep.write(schedule,t_ep); % send action to EP
        
        %----end of step 2)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
        
        %----beginning of step 3)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % This is the same as step 1, but we also get the rewards here
        
        % -----------observe the new space after taking the actions -------
        % do the same thing (the part at the beginning)
        
        % Read Output data from Energyplus at current timestep
        [y, t_ep] = ep.read; 

        br_temp = y(1,3); % get bedroom temperature, first column of output values from IDF
        ot_temp = y(1,1); % get outdoor temperature, Secon column of output values from IDF
        hg = y(1,12);
        hl = y(1,16);
        
        hf = hg - hl;
        rad = y(1,2);

        nf = normalizer(1,hf);
        nrad = normalizer(2,rad);
        
        fc = temp2foreV3(forecast, ts);
        
        h_enrg = y(1,6);
        c_enrg = y(1,9);
        
        isr = y(1,34);

        
        t_enrg = h_enrg+c_enrg;

        ind_time_window = pushBack(ind_time_window, br_temp);
        out_time_window = pushBack(out_time_window, ot_temp);
        

        
        % -----------function for getting the state goes here -------
%         new_state = RL_MLEP_stateconv_hvac1(ind_time_window,out_time_window,t_enrg,hf,fc, state_space);
        new_state = 100;
        % -----------function for getting the rewards goes here -------
    
        if init_flag == 1 % Use current value as previous temperature
            prevt = br_temp; % for the first iteration
        else % else use the previous value
            prevt = t_hist(iter,ts-1);
        end
        

        if mod(ts,144)==0
            done = 1;
        elseif ts == 4463
            done = 1;
        else
            done = 0;
        end

%         [r,r1,r2,r3] = new_reward_hvac3_dual(t_enrg,fc,comfort_zone,br_temp);
        [r,r1,r2,r3] = new_reward_hvac4_action(t_enrg, aa, prev_a);
        
        r_hist(iter,ts) = r;
        r1_hist(iter,ts) = r1;
        r2_hist(iter,ts) = r2;
        
        estat = 0;

        if t_enrg >0
            estat = 400;
        end

        tsdat = [ind_time_window,out_time_window]*10;
    %     sdat = 
        sdat = [tsdat, fc*10 , nf*400, estat, r];
        
        %AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        fwrite(t, sdat, 'int16');
        %AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        
        %----beginning of step 4)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % -----------update the Q-function here-------
        % Q(s,a) <- update rule
%         Q(state, a) =  Q(state, a) + alpha*(r + gamma*(max(Q(state,:)) - Q(state,a))); 
        %----end of step 4)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
%         qval_history(1,ts,iter) = Q(state,1);
%         qval_history(2,ts,iter) = Q(state,2);
        
        state = new_state; % save new state into state
        
        init_flag = 0; % Flag to determine first iteration
        ts = ts+1;
        cr_temp = cr_temp + r;
        tot_eps_e = tot_eps_e + t_enrg;
    end
    
%     tslice = ind_history_temperature(1,end-4462:end);
%     tdev = std(tslice)
%     dev_hist = [dev_hist, tdev];
    total_rewards = cr_temp
%     epsilon = epsilon*exp(-0.07*iter) % Exponential Decay of epsilon

    % Stop MLEP co-simulation process, ep needs to be stopped after every
    % iteration
    ep.stop;

    tot_eps_e
    
    epe_hist = [epe_hist, tot_eps_e];
    cumr_hist = [cumr_hist, cr_temp];
    
    % Need to pause for a certain period; elsewise the program crashes
    % (we need to wait for ep to completely close)
    pause(5);

end % ----- end of outer loop
fclose(t);
toc

%% Visualization and analysis

figure('Renderer', 'painters', 'Position', [10 10 500 300])
fig1 = plot(epe_hist, 'LineWidth',1.5);
grid on;
% title('Performance History (Energy vs. Iteration)');
xlabel('Iteration (episode)','FontSize',12);
ylabel('Total energy consumption (Heat + Cool)','FontSize',12);
% fy1 = fig1.YAxis;
% fy1.FontSize = 12;
% fx1 = fig1.XAxis;
% fx1.FontSize = 12;


figure('Renderer', 'painters', 'Position', [10 10 500 300])
fig2 = plot(cumr_hist, 'LineWidth',1.5);
grid on;
% title('Performance History (Rewards vs. Iteration)');
xlabel('Iteration (episode)','FontSize',12);
ylabel('Cumulative reward per episode','FontSize',12);
% fy2 = fig2.YAxis;
% fy2.FontSize = 12;
% fx2 = fig2.XAxis;
% fx2.FontSize = 12;
ylim([-25000,10000])

%% Econ comp

figure
plot(ql_et(2:end),'r')
hold on
plot(et_hist(1700,2:end),'b')

legend('Q-learning','REINFORCE')
% legend('outdoor','Indoor Temperature')



grid on


xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})
title('Temporal History of Energy Consumption')
xlabel('Day')
ylabel('Mechanical energy consumption (J)')


%% Visualization for different episodes

figure
plot(o_hist(1,:),'k--')
hold on
plot(t_hist(1,:),'m')
hold on
plot(t_hist(2,:),'g')
hold on
plot(t_hist(3,:),'b')
hold on
% plot(t_hist(1300,:),'r')
% hold on
% plot(t_hist(1259,:),'y')
% hold on

line([0,4463],[comfort_zone(2),comfort_zone(2)],'Color','red')
hold on
line([0,4463],[comfort_zone(1),comfort_zone(1)],'Color','red')
hold on
% line([0

for i=1:30
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end


legend('outdoor','Uncontrolled', 'Rudimentary','PG','Upper limit','Lower limit')
% legend('outdoor','Indoor Temperature')






xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})
title('Temperature History in May')
xlabel('Day')
ylabel('Temperature (Celsius)')
ylim([0,30]);

%% Visualization for different episodes

figure
plot(o_hist(1,:),'k--')
hold on
% plot(t_hist(1,:),'m')
% hold on
% % 
% % 
% plot(ql_temp,'g')
hold on
% plot(t_hist(1700,:),'b')
hold on
% plot(t_hist(1300,:),'r')
% hold on
% plot(t_hist(1259,:),'y')
% hold on

line([0,4463],[comfort_zone(2),comfort_zone(2)],'Color','red')
hold on
line([0,4463],[comfort_zone(1),comfort_zone(1)],'Color','red')
hold on
% line([0

for i=1:30
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end


legend('outdoor','Q-learning', 'REINFORCE','Upper limit','Lower limit')
% legend('outdoor','Indoor Temperature')






xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})
title('Temperature History in May')
xlabel('Day')
ylabel('Temperature (Celsius)')
ylim([0,30]);



%% Visualization for different episodes

figure
plot(o_hist(1,:),'k--')
hold on
plot(t_hist(1,:),'m')
hold on
% 
% 
plot(t_hist(500,:),'g')
hold on
plot(t_hist(1700,:),'b')
hold on
% plot(t_hist(1300,:),'r')
% hold on
% plot(t_hist(1259,:),'y')
% hold on

line([0,4463],[comfort_zone(2),comfort_zone(2)],'Color','red')
hold on
line([0,4463],[comfort_zone(1),comfort_zone(1)],'Color','red')
hold on
% line([0

for i=1:30
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end


legend('outdoor','Uncontrolled','REINFORCE','Upper limit','Lower limit')
% legend('outdoor','Indoor Temperature')






xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})
title('Temperature History for Selected Episodes (Iterations)')
xlabel('Day')
ylabel('Temperature (Celsius)')
ylim([0,30]);



%% HVAC actions plot

episode = 3286;

figure
plot(o_hist(1,:),'k--')
hold on
plot(t_hist(episode,:),'b')
hold on

hvac_stat = zeros(1, size(et_hist,2));

for i=1:size(et_hist,2)
    if et_hist(episode,i) > 0
        hvac_stat(1,i) = 40;
    end
end

bins = linspace(1,4463,4463);
bins = bins-0.5;




line([0,4463],[comfort_zone(2),comfort_zone(2)],'Color','red')
hold on
line([0,4463],[comfort_zone(1),comfort_zone(1)],'Color','red')
hold on

bar(bins,hvac_stat,'FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)
hold on

for i=1:30
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end


legend('outdoor','ep30 (after training)','Upper limit','Lower limit','HVAC status (ON)')
% legend('outdoor','Indoor Temperature')






xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})
title('Temperature History for Selected Episodes (Iterations)')
xlabel('Day')
ylabel('Temperature (Celsius)')
ylim([0,35]);

%% Action plot
epsd = 1700;
[td_vent,td_ins] = act2bin8(a_hist(epsd,:));

startr = 2;
endr = 29;

sd = (startr-1)*144;
ed = (endr-1)*144;


h1 = subplot(4,1,1); % ---------------------------------------------------
plot(o_hist(1,:),'k--')
hold on
plot(t_hist(epsd,:),'b')
hold on
line([0,t_num+10],[comfort_zone(2),comfort_zone(2)],'Color','red')
hold on
line([0,t_num+10],[comfort_zone(1),comfort_zone(1)],'Color','red')
hold on
bar(rad_hist(epsd,:))
% line([0

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Temperature History for Selected Episodes (Iterations)')
xlabel('Day')
ylabel('Temperature (Celsius)')
ylim([10,27]);
xlim([sd,ed]);

p1 = get(h1,'position');

h2 = subplot(4,1,2); % ---------------------------------------------------
% bar(td_vent,'FaceAlpha',0,'EdgeColor','green','BarWidth',1)
bar(td_vent,'FaceColor','red','FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)
hold on

p2 = p1;
p2(2) = p1(2) - p1(4);

% set(h2, 'position', p2);

ylim([0,1]);
xlim([sd,ed]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Ventilation')
xlabel('Day')
ylabel('Status (Fraction)')



h3 = subplot(4,1,3); % ---------------------------------------------------
% bar(td_ins,'FaceAlpha',0,'EdgeColor','red','BarWidth',1)
bar(td_ins,'FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)

p3 = p2;
p3(2) = p2(2) - p2(4);

hold on

% set(h3, 'position', p3);

ylim([0,1]);
xlim([sd,ed]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Insulation')
xlabel('Day')
ylabel('Status (On/Off)')

h4 = subplot(4,1,4); % ---------------------------------------------------
% bar(td_ins,'FaceAlpha',0,'EdgeColor','red','BarWidth',1)
% bar(td_ins,'FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)
plot(et_hist(epsd,:)*0.00027778)
% 
% p3 = p2;
% p3(2) = p2(2) - p2(4);

hold on

% set(h3, 'position', p3);

ylim([0,100]);
xlim([sd,ed]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,500000],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Mechanical Energy Consumption (Heating + Cooling)')
xlabel('Day')
ylabel('Consumed Energy (Wh)')

% axis tight;



%% Action plot
epsd = 1700;
[td_vent,td_ins] = act2bin8(a_hist(epsd,:));

startr = 2;
endr = 29;

sd = (startr-1)*144;
ed = (endr-1)*144;


h1 = subplot(4,1,1); % ---------------------------------------------------
plot(o_hist(1,:),'k--')
hold on
plot(t_hist(epsd,:),'b')
hold on
line([0,t_num+10],[comfort_zone(2),comfort_zone(2)],'Color','red')
hold on
line([0,t_num+10],[comfort_zone(1),comfort_zone(1)],'Color','red')
hold on
bar(rad_hist(epsd,:))
% line([0

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Temperature History for Selected Episodes (Iterations)')
xlabel('Day')
ylabel('Temperature (Celsius)')
ylim([10,27]);
xlim([sd,ed]);

p1 = get(h1,'position');

h2 = subplot(4,1,2); % ---------------------------------------------------
% bar(td_vent,'FaceAlpha',0,'EdgeColor','green','BarWidth',1)
bar(td_vent,'FaceColor','red','FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)
hold on

p2 = p1;
p2(2) = p1(2) - p1(4);

% set(h2, 'position', p2);

ylim([0,1]);
xlim([sd,ed]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Ventilation')
xlabel('Day')
ylabel('Status (Fraction)')



h3 = subplot(4,1,3); % ---------------------------------------------------
% bar(td_ins,'FaceAlpha',0,'EdgeColor','red','BarWidth',1)
bar(td_ins,'FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)

p3 = p2;
p3(2) = p2(2) - p2(4);

hold on

% set(h3, 'position', p3);

ylim([0,1]);
xlim([sd,ed]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Insulation')
xlabel('Day')
ylabel('Status (On/Off)')

h4 = subplot(4,1,4); % ---------------------------------------------------
% bar(td_ins,'FaceAlpha',0,'EdgeColor','red','BarWidth',1)
% bar(td_ins,'FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)
plot(et_hist(epsd,:)*0.00027778)
% 
% p3 = p2;
% p3(2) = p2(2) - p2(4);

hold on

% set(h3, 'position', p3);

ylim([0,100]);
xlim([sd,ed]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,500000],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Mechanical Energy Consumption (Heating + Cooling)')
xlabel('Day')
ylabel('Consumed Energy (Wh)')

% axis tight;



%% Action plot for paper
epsd = 1700;
[td_vent,td_ins] = act2bin8(a_hist(epsd,:));

startr = 8;
endr = 11;

sd = (startr-1)*144;
ed = (endr-1)*144;

figure('Renderer', 'painters', 'Position', [10 10 500 550])

h1 = subplot(9,1,1:5); % ---------------------------------------------------
plot(o_hist(1,:),'k--','LineWidth',1.5)
hold on
plot(t_hist(epsd,:),'b','LineWidth',1.5)
hold on
line([0,t_num+10],[comfort_zone(2),comfort_zone(2)],'Color','red')
hold on
line([0,t_num+10],[comfort_zone(1),comfort_zone(1)],'Color','red')
hold on


for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

lgd1 = legend('Outdoor Temperature','Indoor Temperature (REINFORCE)',...
    'Location','southeast');
lgd1.FontSize = 10;
legend boxoff

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
set(gca,'xticklabel',{[]})


ylabel('Temperature (Celsius)','FontSize',14)
ylim([10,27]);
xlim([sd,ed]);
grid on

hy1 = h1.YAxis;
hy1.FontSize = 12;

p1 = get(h1,'position');

h2 = subplot(9,1,6); % ---------------------------------------------------
% bar(td_vent,'FaceAlpha',0,'EdgeColor','green','BarWidth',1)
bar(td_vent*10,'FaceColor','red','FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)
hold on


p2 = get(h2,'position');
p2(2) = 0.4174;

% set(h2, 'position', p2);

ylim([0,1]);
xlim([sd,ed]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end
set(gca,'yticklabel',{[]})

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
set(gca,'xticklabel',{[]})

lgd2 = legend('Ventilation (10% open)','Location','northwest');
lgd2.FontSize = 10;
% ylabel('Status (Fraction)')
% legend boxoff



h3 = subplot(9,1,7); % ---------------------------------------------------
% bar(td_ins,'FaceAlpha',0,'EdgeColor','red','BarWidth',1)
bar(td_ins,'FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)
hold on

p3 = get(h3,'position');
p3(2) = 0.2974;
% set(h3, 'position', p3);

ylim([0,1]);
xlim([sd,ed]);

set(gca,'yticklabel',{[]})

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

lgd3 = legend('Insulation (On/Off)','Location','northwest');
lgd3.FontSize = 10;
% legend boxoff

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
set(gca,'xticklabel',{[]})

% ylabel('Status (On/Off)')

h4 = subplot(9,1,8:9); % ---------------------------------------------------
% bar(td_ins,'FaceAlpha',0,'EdgeColor','red','BarWidth',1)
% bar(td_ins,'FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)
plot(et_hist(epsd,:)*0.00027778,'b','LineWidth',1.5)

p4 = get(h4,'position');
p4(2) = 0.1774;
% set(h4, 'position', p4);

hold on

% set(h3, 'position', p3);

ylim([0,25]);
xlim([sd,ed]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,500000],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'});

xlabel('Day','FontSize',14)
ylabel('Load (Wh)','FontSize',14)
grid on
xtl = get(h4,'XTickLabel');
% set(xtl,'fontsize',18)
hy4 = h4.YAxis;
hy4.FontSize = 12;
hx4 = h4.XAxis;
hx4.FontSize = 12;
% axis tight;
% lgd4 = legend('Load','Location','northeast');
% lgd4.FontSize = 12;
% legend boxoff



%% QL/PG comp for paper
epsd = 1700;
startr = 22;
endr = 29;

sd = (startr-1)*144;
ed = (endr-1)*144;

figure('Renderer', 'painters', 'Position', [10 10 500 550])

h1 = subplot(3,1,1:2); % ---------------------------------------------------
plot(o_hist(1,:),'k--','LineWidth',1.5)
hold on

plot(ql_temp,'g','LineWidth',1.5)
hold on

plot(t_hist(epsd,:),'b','LineWidth',1.5)
hold on
line([0,t_num+10],[comfort_zone(2),comfort_zone(2)],'Color','red')
hold on
line([0,t_num+10],[comfort_zone(1),comfort_zone(1)],'Color','red')
hold on

hy1 = h1.YAxis;
hy1.FontSize = 12;


for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

% legend('Outdoor Temperature','Q-Learning','REINFORCE','Upper limit','Lower limit')

lgd1 = legend('Outdoor Temperature','Indoor Temperature (Q-Learning)',...
    'Indoor Temperature (REINFORCE)',...
    'Location','southwest');
lgd1.FontSize = 10;
% legend boxoff

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
% set(h1,'YTick',yTick(2:end));

% xlabel(h1.axes1,'')
% xlabel('')
% yticklabels1 = get(h1, 'YTickLabel');
% yticklabels1{1} = '';   %needs to exist but make it empty
% set(h1, 'YTickLabel', yticklabels1);

set(gca,'xticklabel',{[]})
% title('Temperature History for Selected Episodes (Iterations)')
% xlabel('Day')
ylabel('Temperature (Celsius)','FontSize',14)
ylim([10,27]);
xlim([sd,ed]);
grid on
p1 = get(h1,'position');

h2 = subplot(3,1,3); % ---------------------------------------------------
% bar(td_ins,'FaceAlpha',0,'EdgeColor','red','BarWidth',1)
% bar(td_ins,'FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)
plot(ql_ec*0.00027778,'g','LineWidth',1.5)
hold on
plot(et_hist(epsd,:)*0.00027778,'b','LineWidth',1.5)

% 
p2 = get(h2,'position');
% p2(2) = p2(2) - p1(4);
% p2(4) = p1(3)/2;
p2(2) = 0.1796;

hold on

% set(h2, 'position', p2);

ylim([0,600]);
xlim([sd,ed]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,500000],'Color','black','LineStyle',':');
    hold on
end

lgd2 = legend('Q-Learning','REINFORCE',...
    'Location','northwest');
lgd2.FontSize = 10;
legend boxoff

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

% title('Mechanical Energy Consumption (Heating + Cooling)')
xlabel('Day','FontSize',14)
ylabel('Load (Wh)','FontSize',14)

grid on
% yticklabels2 = get(h2, 'YTickLabel');
% % yticklabels2{3} = yticklabels2{2};
% % yticklabels2{5} = yticklabels2{4};
% yticklabels2{2} = '100';
% yticklabels2{3} = '200';
% yticklabels2{4} = '300';
% yticklabels2{5} = '400';
% yticklabels2{6} = '500';

hy2 = h2.YAxis;
hy2.FontSize = 12;

% yticklabels2{end} = '';   %needs to exist but make it empty
% set(h2, 'YTickLabel', yticklabels2);
% axis tight;



%% Action plot
epsd = 30;
[td_vent,td_ins] = act2bin8(a_hist(epsd,:));


h1 = subplot(3,1,1); % ---------------------------------------------------
plot(o_hist(1,:),'k--')
hold on
plot(t_hist(epsd,:),'b')
hold on
line([0,t_num+10],[comfort_zone(2),comfort_zone(2)],'Color','red')
hold on
line([0,t_num+10],[comfort_zone(1),comfort_zone(1)],'Color','red')
hold on
bar(rad_hist(epsd,:))
% line([0

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Temperature History for Selected Episodes (Iterations)')
xlabel('Day')
ylabel('Temperature (Celsius)')
ylim([15,30]);
xlim([0,t_num]);

p1 = get(h1,'position');

h2 = subplot(3,1,2); % ---------------------------------------------------
% bar(td_vent,'FaceAlpha',0,'EdgeColor','green','BarWidth',1)
bar(td_vent,'FaceColor','red','FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)
hold on

p2 = p1;
p2(2) = p1(2) - p1(4);

set(h2, 'position', p2);

ylim([0,1]);
xlim([0,t_num]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Ventilation')
xlabel('Day')
ylabel('Status (Fraction)')



h3 = subplot(3,1,3); % ---------------------------------------------------
% bar(td_ins,'FaceAlpha',0,'EdgeColor','red','BarWidth',1)
bar(td_ins,'FaceAlpha',0.5,'EdgeColor','none','BarWidth',1)

p3 = p2;
p3(2) = p2(2) - p2(4);

hold on

set(h3, 'position', p3);

ylim([0,1]);
xlim([0,t_num]);

for i=1:runPeriod_Days
    line([(i)*144,(i)*144],[0,40],'Color','black','LineStyle',':');
    hold on
end

xticks([0,144,144*2,144*3,144*4,144*5,144*6,144*7,144*8,144*9,144*10, ...
    144*11,144*12,144*13,144*14,144*15,144*16,144*17,144*18,144*19, ...
    144*20,144*21,144*22,144*23,144*24,144*25,144*26,144*27,144*28, ...
    144*29,144*30])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13', ...
    '14','15','16','17','18','19','20','21','22','23','24','25','26', ...
    '27','28','29','30','31'})

title('Insulation')
xlabel('Day')
ylabel('Status (On/Off)')
% axis tight;



%%
a01_cool = sum(ce_hist(1,:))
a01_heat = sum(he_hist(1,:))


%%
stackData = zeros(1,3,2); 

% Albany
stackData(1,1,1) = 11.258;
stackData(1,1,2) = 0.21188;
stackData(1,2,1) = 5.1929;
stackData(1,2,2) = 0.21213;
stackData(1,3,1) = 1.1436;
stackData(1,3,2) = 2.1489;

% SLC
stackData(2,1,1) = 14.365;
stackData(2,1,2) = 0.3756;
stackData(2,2,1) = 7.1793;
stackData(2,2,2) = 0.38793;
stackData(2,3,1) = 2.6861;
stackData(2,3,2) = 1.1774;

% LA
stackData(3,1,1) = 18.224;
stackData(3,1,2) = 0.14287;
stackData(3,2,1) = 12.112;
stackData(3,2,2) = 0.14287;
stackData(3,3,1) = 3.8601;
stackData(3,3,2) = 0.54827;

% Pittsburgh
stackData(4,1,1) = 15.328;
stackData(4,1,2) = 0.25736;
stackData(4,2,1) = 7.1060;
stackData(4,2,2) = 0.25736;
stackData(4,3,1) = 4.9659;
stackData(4,3,2) = 1.4043;

% Portland
stackData(5,1,1) = 13.554;
stackData(5,1,2) = 0.4946;
stackData(5,2,1) = 7.6332;
stackData(5,2,2) = 0.4964;
stackData(5,3,1) = 4.6035;
stackData(5,3,2) = 2.1869;

% Detroit
stackData(6,1,1) = 14.554;
stackData(6,1,2) = 0.54233;
stackData(6,2,1) = 7.0347;
stackData(6,2,2) = 0.54256;
stackData(6,3,1) = 5.1176;
stackData(6,3,2) = 2.9001;

% Kansas City
stackData(7,1,1) = 17.903;
stackData(7,1,2) = 0.19459;
stackData(7,2,1) = 9.9649;
stackData(7,2,2) = 0.19459;
stackData(7,3,1) = 7.1522;
stackData(7,3,2) = 1.2896;



% stackData(1,1,1) = 1;

groupLabels = {'Albany, NY','Salt Lake City, UT','Los Angeles, CA','Pittsburgh, PA',...
     'Portland, OR','Detroit, MI', 'Kansas City, MO'}; 
% h = plotBarStackGroups(stackData, groupLabels)
h = plotBarStackGroups(stackData, groupLabels);
% Chance the colors of each bar segment
% colors = jet(size(stackData,2)); %or define your own color order; 1 for each m segments
% colors = repelem(colors,size(stackData,1),1); 
% colors = mat2cell(colors,ones(size(colors,1),1),3);
colors = cell(6,1);

colors{1,1} = [1 0.6 0.6];
colors{4,1} = [1 0 0];

colors{2,1} = [0.6 1 0.6];
colors{5,1} = [0 1 0];

colors{3,1} = [0.6 0.6 1];
colors{6,1} = [0 0 1];
% 
% 
% 
set(h,{'FaceColor'},colors)

lgd = legend('No passive elements (Cooling Energy)','No passive elements (Heating Energy)', ...
'Baseline controller (Cooling Energy)','Baseline controller (Heating Energy)', ...
'REINFORCE (Cooling Energy)','REINFORCE (Heating Energy)','Location','northwest');
grid on
yl = ylabel('Mechanical Energy Consumption (E+07 J)');
lgd.FontSize = 14;
set(yl,'FontSize',14);


%% w/out Aug

stackData = zeros(1,3,2); 

% Jan
stackData(1,1,1) = 0;
stackData(1,1,2) = 23.754;
stackData(1,2,1) = 0;
stackData(1,2,2) = 27.579;
stackData(1,3,1) = 0.12701;
stackData(1,3,2) = 21.1144;

% Mar
stackData(2,1,1) = 0;
stackData(2,1,2) = 7.6087;
stackData(2,2,1) = 0;
stackData(2,2,2) = 9.2576;
stackData(2,3,1) = 0.48512;
stackData(2,3,2) = 5.9302;

% Albany May
stackData(3,1,1) = 11.258;
stackData(3,1,2) = 0.21188;
stackData(3,2,1) = 5.1929;
stackData(3,2,2) = 0.21213;
stackData(3,3,1) = 1.1436;
stackData(3,3,2) = 2.1489;

% Oct
stackData(4,1,1) = 0.25629;
stackData(4,1,2) = 7.0692;
stackData(4,2,1) = 0.25629;
stackData(4,2,2) = 4.0591;
stackData(4,3,1) = 0.95299;
stackData(4,3,2) = 5.0841;

% Dec
stackData(5,1,1) = 0;
stackData(5,1,2) = 13.757;
stackData(5,2,1) = 0;
stackData(5,2,2) = 17.431;
stackData(5,3,1) = 0;
stackData(5,3,2) = 10.801;

%%
stackData = zeros(1,3,2); 

% Jan
stackData(1,1,1) = 0;
stackData(1,1,2) = 23.754;
stackData(1,2,1) = 0;
stackData(1,2,2) = 27.579;
stackData(1,3,1) = 0.12701;
stackData(1,3,2) = 21.1144;

% Mar
stackData(2,1,1) = 0;
stackData(2,1,2) = 7.6087;
stackData(2,2,1) = 0;
stackData(2,2,2) = 9.2576;
stackData(2,3,1) = 0.48512;
stackData(2,3,2) = 5.9302;

% Albany May
stackData(3,1,1) = 11.258;
stackData(3,1,2) = 0.21188;
stackData(3,2,1) = 5.1929;
stackData(3,2,2) = 0.21213;
stackData(3,3,1) = 1.1436;
stackData(3,3,2) = 2.1489;

% Aug
stackData(4,1,1) = 30.746;
stackData(4,1,2) = 0;
stackData(4,2,1) = 15.037;
stackData(4,2,2) = 0;
stackData(4,3,1) = 30.398;
stackData(4,3,2) = 0.01;

% Oct
stackData(5,1,1) = 0.25629;
stackData(5,1,2) = 7.0692;
stackData(5,2,1) = 0.25629;
stackData(5,2,2) = 4.0591;
stackData(5,3,1) = 0.95299;
stackData(5,3,2) = 5.0841;

% Dec
stackData(6,1,1) = 0;
stackData(6,1,2) = 13.757;
stackData(6,2,1) = 0;
stackData(6,2,2) = 17.431;
stackData(6,3,1) = 0;
stackData(6,3,2) = 10.801;



% stackData(1,1,1) = 1;

groupLabels = {'January','March','May (Training environment)','August',...
     'October','December'}; 
% h = plotBarStackGroups(stackData, groupLabels)
h = plotBarStackGroups(stackData, groupLabels);
% Chance the colors of each bar segment
% colors = jet(size(stackData,2)); %or define your own color order; 1 for each m segments
% colors = repelem(colors,size(stackData,1),1); 
% colors = mat2cell(colors,ones(size(colors,1),1),3);
colors = cell(6,1);

colors{1,1} = [1 0.6 0.6];
colors{4,1} = [1 0 0];

colors{2,1} = [0.6 1 0.6];
colors{5,1} = [0 1 0];

colors{3,1} = [0.6 0.6 1];
colors{6,1} = [0 0 1];
% 
% 
% 
set(h,{'FaceColor'},colors)

lgd = legend('No passive elements (Cooling Energy)','No passive elements (Heating Energy)', ...
'Baseline controller (Cooling Energy)','Baseline controller (Heating Energy)', ...
'REINFORCE (Cooling Energy)','REINFORCE (Heating Energy)','Location','northwest');
grid on
yl = ylabel('Mechanical Energy Consumption (E+07 J)');
lgd.FontSize = 14;
set(yl,'FontSize',14);

%%

stackData = randi(5,3,5,7); 
groupLabels = {'a' 'b' 'c'}; 
% h = plotBarStackGroups(stackData, groupLabels)
h = plotBarStackGroups(stackData, groupLabels);
% Chance the colors of each bar segment
colors = jet(size(h,2)); %or define your own color order; 1 for each m segments
colors = repelem(colors,size(h,1),1); 
colors = mat2cell(colors,ones(size(colors,1),1),3);
set(h,{'FaceColor'},colors)


%%

citcom = [11.3 5.2 2.87; 16.3 10.7 5.73; 10.9 5.2 5.48; 15.1 9.4 1.12; 10.4 5.2 1.76];
figure
bar(citcom)


title('Performance of Trained Model in Different Environments (May)')
xlabel('City')
ylabel('Mechanical Energy Consumption (E+07 J)')
grid on
legend('Uncontrolled Model','Rudimentary Control','REINFORCE')

xticklabels({'Albany','Philadelphia','Denver','Chicago','Seattle'})


%%

moncom = [11.3 5.2 2.87; 7.1 4.1 6.8; 13.3 17.0 9.4];
figure

bar(moncom)


title('Performance of Trained Model in Different Months')
xlabel('City')
ylabel('Mechanical Energy Consumption (E+07 J)')
grid on
legend('Uncontrolled Model','Rudimentary Control','REINFORCE')

xticklabels({'May (trained)','October','December'})

%%

moncom = [-12000; -3800; 6630];
figure

bar(moncom)


% title('Performance of Trained Model in Different Months')
% xlabel('City')
ylabel('Cumulative Rewards')
grid on
legend('Uncontrolled Model','Rudimentary Control','REINFORCE')

xticklabels({'May (trained)','October','December'})
