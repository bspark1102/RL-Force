function [vent, ins] = act2bin8(actionhistory)

    len = size(actionhistory,2);
    vent = zeros(len,1);
    ins = zeros(len,1);
    
    
    for i=1:len
        if actionhistory(i) == 0
            vent(i) = 0;
            ins(i) = 1;
        elseif actionhistory(i) == 1
            vent(i) = 1;
            ins(i) = 1;
        elseif actionhistory(i) == 2
            vent(i) = 0;
            ins(i) = 0;
        elseif actionhistory(i) == 3
            vent(i) = 0.5;
            ins(i) = 0;
        elseif actionhistory(i) == 4
            vent(i) = 0.5;
            ins(i) = 1;
        elseif actionhistory(i) == 5
            vent(i) = 0.1;
            ins(i) = 0;
        elseif actionhistory(i) == 6
            vent(i) = 0.1;
            ins(i) = 1;
        else
            vent(i) = 1;
            ins(i) = 0;
        end
    
    end

end