function schedule = act7_select6_sim(action)
    schedule = zeros(1,6);
    % 1 - large BR insulation
    % 2 - small BR insulation
    % 3 - Sliding door insulation (fixed)
    % 4 - Sliding door insulation (operable)
    % 5 - Ventillation BR
    % 6 - Ventilation Living
    
    
    % important note : ventilation on is 1, insulation on is also 1 but
    % their effects are the opposite so the numbers need to be inverted
    if action == 0 % windows closed and insulation on / minimum heat exchange
        schedule(1,1) = 1;
        schedule(1,2) = 1;
        schedule(1,5) = 0;
        schedule(1,6) = 0;
    elseif action == 1 % vent on insul on / one action
        schedule(1,1) = 1;
        schedule(1,2) = 1;
        schedule(1,5) = 1;
        schedule(1,6) = 1;
    elseif action == 2 % vent off insul off / the other action
        schedule(1,1) = 0;
        schedule(1,2) = 0;
        schedule(1,5) = 0;
        schedule(1,6) = 0;
    elseif action == 3 % vent half insul off
        schedule(1,1) = 0;
        schedule(1,2) = 0;
        schedule(1,5) = 0.5;
        schedule(1,6) = 0.5;
    elseif action == 4 % vent half insul on
        schedule(1,1) = 1;
        schedule(1,2) = 1;
        schedule(1,5) = 0.5;
        schedule(1,6) = 0.5;
    elseif action == 5 % vent 0.1 insul off
        schedule(1,1) = 0;
        schedule(1,2) = 0;
        schedule(1,5) = 0.1;
        schedule(1,6) = 0.1;
    elseif action == 6 % vent 0.1 insul on
        schedule(1,1) = 1;
        schedule(1,2) = 1;
        schedule(1,5) = 0.1;
        schedule(1,6) = 0.1;
    else % windows open and insulation off / maximum heat exchange
        schedule(1,1) = 0;
        schedule(1,2) = 0;
        schedule(1,5) = 1;
        schedule(1,6) = 1;
    end
        
end