  % SUMMER RULES
% note that all available values are those logged at the completion of the last timestep; there is no "now"
% need to lock out night insulation completely

UNIT_AT_array = ind_time_window;
OS_AT_array = out_time_window;
OS_AT = ot_temp;

LR_WHG_array = hg_time_window;
LR_WHL_array = hl_time_window ;

UNIT_AT_nv = UNIT_AT_array >= 18.5;       % Logical array of timesteps in which indoor air temp is above NV bound
UNIT_AT_vwarm = UNIT_AT_array >= 18.5;    % Logical array of indoor air VERY WARM timesteps
UNIT_AT_cool = UNIT_AT_array < 18.5;      % Logical array of indoor air COOL timesteps

OS_AT_cold = OS_AT_array <= 16;         % Logical array of outdoor air COLD timesteps
OS_AT_cool = (OS_AT_array > 16 & OS_AT_array <= 18);    % Logical array of outdoor air COOL timesteps 
OS_AT_buffer = UNIT_AT_array - OS_AT_array;             % Numeric buffer btw OSAT and ISAT at each timestep in past hour
OS_AT_nv = OS_AT_buffer >= 0;           % Logical array of timesteps in which OSAT is cooler than ISAT

LR_WHG_pos = LR_WHG_array > 0;          % Logical array of timesteps in which LR Op window heat gain was positive (>0)
LR_WHG_neg = LR_WHG_array <= 0;         % Logical array of timesteps in which LR Op window heat gain was negative (<=0)
LR_WHL_def = LR_WHL_array >= 2;         % Logical array of timesteps in which LR Op window heat loss rate was 5W or more

% SAVE VALUES
% WHGlogic = zeros(endTime,5);
% WHGlogic(iLog,:) = LR_WHG_array;

% action_index = 0

% SHADING QUESTIONS
if any(UNIT_AT_vwarm) % If the indoor air was very warm at any timestep in the previous hour
    
    if any(LR_WHG_pos)      % If window was gaining heat at any time in the past hour
%         a = 1;
%         b = 1;
%         c = 1;
%         d = 1;                  % Turn all shades ON to limit solar heat gain
        
        ait = 0; % temporary action index
    elseif all(LR_WHL_def) % Else if window was losing heat definitively at all times in the past hour
%         a = 0;
%         b = 0;
%         c = 0;
%         d = 0;                  % Turn all shades OFF to allow radiative cooling
        
        ait = 1;
    else  % Else window was losing heat, but not substantially, over all of the past hour
%         a = a;
%         b = b;
%         c = c;
%         d = d;                  % Keep shade status of last timestep
        
        ait = 2;
    end
    
else                 % Otherwise, the indoor air has not been overheating consistently 
%         a = a;
%         b = b;
%         c = c;
%         d = d;                  % Leave all shades as they were in previous timestep
        
        ait = 3;
end

% NV QUESTIONS
% If the indoor air temp has been above the NV bound for all of the past hour
if all(UNIT_AT_nv)
    
    % OSAT ramp approach
    if OS_AT <= 11                        % If OSAT is COLD
%         e = 0.01;                           % Turn on NV very low

        if ait == 0
            ai = 0;
        elseif ait == 1
            ai = 1;
        elseif ait == 2
            ai = 2;
        elseif ait == 3
            ai = 3;
        end
        
    elseif (OS_AT > 11) && (OS_AT <= 14)  % Else if OSAT is VERY COOL
%         e = 0.01 * OS_AT;                   % Turn on NV at 1% of OSAT
        
        if ait == 0
            ai = 4;
        elseif ait == 1
            ai = 5;
        elseif ait == 2
            ai = 6;
        elseif ait == 3
            ai = 7;
        end
        
    elseif (OS_AT > 14) && (OS_AT <= 16)  % Else if OSAT is COOL
%         e = 0.02 * OS_AT;                   % Turn on NV at 2% of OSAT
        
        if ait == 0
            ai = 8;
        elseif ait == 1
            ai = 9;
        elseif ait == 2
            ai = 10;
        elseif ait == 3
            ai = 11;
        end
        
    elseif (OS_AT > 16) && (OS_AT < 18.5)   % Else if OSAT is FRESH
%         e = 0.05 * OS_AT;                   % Turn on NV at 5% of OSAT
        
        if ait == 0
            ai = 12;
        elseif ait == 1
            ai = 13;
        elseif ait == 2
            ai = 14;
        elseif ait == 3
            ai = 15;
        end
        
    % Else if OSAT is in the tstat range and consistently cooler than ISAT    
    elseif (OS_AT >= 18.5) && (OS_AT <= 25) && all(OS_AT_nv)     
%         e = 1;     
        
        if ait == 0
            ai = 16;
        elseif ait == 1
            ai = 17;
        elseif ait == 2
            ai = 18;
        elseif ait == 3
            ai = 19;
        end
        
        % Turn on NV fully
    else % Else OSAT is too hot
%         e = 0;                              % Leave NV off
        
        if ait == 0
            ai = 20;
        elseif ait == 1
            ai = 21;
        elseif ait == 2
            ai = 22;
        elseif ait == 3
            ai = 23;
        end
        
    end
    
else
%     e = 0; % Otherwise, indoor air is cool and NV is not needed; leave NV OFF
    
        if ait == 0
            ai = 24;
        elseif ait == 1
            ai = 25;
        elseif ait == 2
            ai = 26;
        elseif ait == 3
            ai = 27;
            
        end
    
end