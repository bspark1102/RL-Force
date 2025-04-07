%% Initialize
close all; clear all; clc;
fclose('all');

ep = mlep;

% Select Energy Plus Model in .idf
ep.idfFile =('070121j_Single_NoSunspace_Rooftop_IdealLoadsAir_South_ONEZONE_MLEP');

% Select Weather file in .epw
ep.epwFile=('USA_NY_Albany.Intl.AP.725180_TMYx.2004-2018');

% Initialize MLEP co-simulation 
ep.initialize;

% display outputTable 
outputTable = ep.outputTable;

% display inputTable, which is the objects in the "External Interface: Schedule" in .idf
inputTable = ep.inputTable;

runPeriod_Days   = 31;
runperiod_Second = 24*60*60*runPeriod_Days;
                
% Set parameters and define variables
n_iter = 1000; %number of months to run, becomes n_iter*31 in python 
t_num = 2881; %number of timesteps
windowsize = 5; %size of the window

% Initialize variables
a_hist = zeros(n_iter,t_num); % action history
t_hist = zeros(n_iter,t_num); % temperature history (indoor)
o_hist = zeros(n_iter,t_num); % temp history (outdoor)
f_hist = zeros(n_iter,t_num); % heat flux history
rrbs_hist = zeros(t_num,20); % PG state history

ce_hist = zeros(n_iter,t_num); % cooling energy history
he_hist = zeros(n_iter,t_num); % heating energy history
et_hist = zeros(n_iter,t_num); % total energy history (heat+cooling)

% define a window for each indoor/outdoor temperature
hf_time_window = zeros(1,windowsize);
hg_time_window = zeros(1,windowsize);
hl_time_window = zeros(1,windowsize);
et_time_window = zeros(1,windowsize);
etc_time_window = zeros(1,windowsize);
eth_time_window = zeros(1,windowsize);

% episodal stuff
epe_hist = []; % energy per episode
cumr_hist = [];

% define a window for each indoor/outdoor temperature
ind_time_window = zeros(1,windowsize);
out_time_window = zeros(1,windowsize);

comfort_zone = [18 25];
tot_eps_e =0;

t = tcpip('127.0.0.1', 50000);

%%

tic
fopen(t) %------------------------------0000
schedule = zeros(1,8);

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

    %----beginning of step 1)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ts = 1;
    % Read Output data from Energyplus at current timestep
    [y, t_ep] = ep.read; 
    
    y = y';
    
    br_temp = y(3,1); % get bedroom temperature, first column of output values from IDF
    ot_temp = y(1,1); % get outdoor temperature, Secon column of output values from IDF
 %----                      
    hg = y(6,1);
    hl = y(10,1);

    hf = hg - hl;
    rad = y(2,1);
    
    nf = normalizer(hf);
    nhg = normalizer(hg);
    nhl = normalizer(hl);
    
    h_enrg = y(5,1);
    c_enrg = y(4,1);
        
    t_enrg = h_enrg+c_enrg;
    
    estat = 0;
    
    if t_enrg >0
        estat = 400;
    end
    
    if ts == 1 % do only for first iteration
        ind_time_window(1,:) = y(3,1);
        out_time_window(1,:) = y(1,1);
        hf_time_window(1,:) = hf;
        hg_time_window(1,:) = nhg;
        hl_time_window(1,:) = nhl;
        et_time_window(1,:) = t_enrg;
        etc_time_window(1,:) = c_enrg;
        eth_time_window(1,:) = h_enrg;
    end
    
    ind_time_window = pushBack(ind_time_window, y(3,1));
    out_time_window = pushBack(out_time_window, y(1,1));
    hf_time_window = pushBack(hf_time_window, hf);
    hg_time_window = pushBack(hg_time_window, nhg);
    hl_time_window = pushBack(hl_time_window, nhl);
    et_time_window = pushBack(et_time_window, t_enrg);
    etc_time_window = pushBack(etc_time_window, c_enrg);
    eth_time_window = pushBack(eth_time_window, h_enrg);
    
    rrbs_hist(ts,1:5) = out_time_window*10;
    rrbs_hist(ts,6:10) = hg_time_window*400;
    rrbs_hist(ts,11:15) = hl_time_window*400;
    rrbs_hist(ts,16:end) = ind_time_window*10;
    
    sdat = [rrbs_hist(ts,:), 0];
    
    %AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    fwrite(t, sdat, 'int16');
    %AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    
    % -----------function for getting the state goes here -------
    aa = 0;
    ps = schedule; % save previous schedule for initial loop
    tot_eps_e = 0;

    while ts < 2882   % a single episode (Inner loop)
        % runs until the last timestep-1 because we are doing ep.read twice
        % per loop

        % Append every last measurement to a history array
        % (Just for checking purposes)
        t_hist(iter,ts) = br_temp;
        o_hist(iter,ts) = ot_temp;
        f_hist(iter,ts) = hf;
        
        ce_hist(iter,ts) = h_enrg;
        he_hist(iter,ts) = c_enrg;
        et_hist(iter,ts) = t_enrg;
        
        prev_a = aa; % Save previous action (used in reward function)

        aa = fread(t, [1, 1]);  %------------------------------0000
        
        a_hist(iter,ts) = aa;
                
        schedule = select_action(aa,comfort_zone,ot_temp,ps);

        ep.write(schedule,t_ep); % send action to EP
        ps = schedule;
        %----end of step 2)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        %----beginning of step 3)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % This is the same as step 1, but we also get the rewards here
        % -----------observe the new space after taking the actions -------
        % do the same thing (the part at the beginning)
        
        % Read Output data from Energyplus at current timestep
        [y, t_ep] = ep.read; 
        y=y';
        br_temp = y(3,1); % get bedroom temperature, first column of output values from IDF
        ot_temp = y(1,1); % get outdoor temperature, Secon column of output values from IDF
        %----                      
        hg = y(6,1);
        hl = y(10,1);

        hf = hg - hl;
        rad = y(2,1);

        nf = normalizer(hf);
        nhg = normalizer(hg);
        nhl = normalizer(hl);

        h_enrg = y(5,1);
        c_enrg = y(4,1);

        t_enrg = h_enrg+c_enrg;

        estat = 0;

        if t_enrg >0
            estat = 400;
        end

        if ts == 1 % do only for first iteration
            ind_time_window(1,:) = y(3,1);
            out_time_window(1,:) = y(1,1);
            hf_time_window(1,:) = hf;
            hg_time_window(1,:) = nhg;
            hl_time_window(1,:) = nhl;
            et_time_window(1,:) = t_enrg;
            etc_time_window(1,:) = c_enrg;
            eth_time_window(1,:) = h_enrg;
        end

        ind_time_window = pushBack(ind_time_window, y(3,1));
        out_time_window = pushBack(out_time_window, y(1,1));
        hf_time_window = pushBack(hf_time_window, hf);
        hg_time_window = pushBack(hg_time_window, nhg);
        hl_time_window = pushBack(hl_time_window, nhl);
        et_time_window = pushBack(et_time_window, t_enrg);
        etc_time_window = pushBack(etc_time_window, c_enrg);
        eth_time_window = pushBack(eth_time_window, h_enrg);

        rrbs_hist(ts,1:5) = out_time_window*10;
        rrbs_hist(ts,6:10) = hg_time_window*400;
        rrbs_hist(ts,11:15) = hl_time_window*400;
        rrbs_hist(ts,16:end) = ind_time_window*10;

            
        if init_flag == 1 % Use current value as previous temperature
            prevt = br_temp; % for the first iteration
        else % else use the previous value
            prevt = t_hist(iter,ts-1);
        end
        
        [r,r1,r2] = get_reward(t_enrg, aa, prev_a);
        
        r_hist(iter,ts) = r;
        r1_hist(iter,ts) = r1;
        r2_hist(iter,ts) = r2;
        
        sdat = [rrbs_hist(ts,:), r];
        %AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        fwrite(t, sdat, 'int16');
        %AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        
        init_flag = 0; % Flag to determine first iteration
        ts = ts+1;
        cr_temp = cr_temp + r;
        tot_eps_e = tot_eps_e + t_enrg;
    end
    
    total_rewards = cr_temp

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

