function schedule = select_action(action,comfortzone,ot_temp,ps)
    g = comfortzone(1);
    h = comfortzone(2);
    
    av = floor(action/4);
    ai = mod(action,4);
    
    OS_AT = ot_temp;
    f=0;
    if ai==0     % If window was gaining heat at any time in the past hour
            a = 1;
            b = 1;
            c = 1;
            d = 1;                  % Turn all shades ON to limit solar heat gain

    elseif ai==1 % Else if window was losing heat definitively at all times in the past hour
            a = 0;
            b = 0;
            c = 0;
            d = 0;                  % Turn all shades OFF to allow radiative cooling

    elseif ai==2 % Else window was losing heat, but not substantially, over all of the past hour
            a = ps(1);
            b = ps(2);
            c = ps(3);
            d = ps(4);                  % Keep shade status of last timestep

    elseif ai==3                 % Otherwise, the indoor air has not been overheating consistently 
            a = ps(1);
            b = ps(2);
            c = ps(3);
            d = ps(4);                  % Leave all shades as they were in previous timestep
    end

    % NV QUESTIONS
    if av==0
        e = 0.01;                           % Turn on NV very low
    elseif av==1  % Else if OSAT is VERY COOL
        e = 0.01 * OS_AT;                   % Turn on NV at 1% of OSAT
    elseif av==2  % Else if OSAT is COOL
        e = 0.02 * OS_AT;                   % Turn on NV at 2% of OSAT
    elseif av==3   % Else if OSAT is FRESH
        e = 0.05 * OS_AT;                   % Turn on NV at 5% of OSAT
    elseif av==4     
        e = 1;                              % Turn on NV fully
    elseif av==5% Else OSAT is too hot
        e = 0;                              % Leave NV off
    elseif av==6
        e = 0; % Otherwise, indoor air is cool and NV is not needed; leave NV OFF
    end
        
    schedule = [a; b; c; d; e; f; g; h]';
end