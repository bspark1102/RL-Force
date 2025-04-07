function [ai,av] = action_decouple(single_action_array)
    
    tlen = size(single_action_array,2);
    
    ai = zeros(1,tlen);
    av = zeros(1,tlen);
    
    for i=1:tlen
        temp_action = single_action_array(1,i);
        
        av(1,i) = floor(temp_action/4);
        ai(1,i) = mod(temp_action,4);
    end

end