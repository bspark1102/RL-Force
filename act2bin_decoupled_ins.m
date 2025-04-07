function ai = act2bin_decoupled_ins(raw_action_index)
    
    timestep = size(raw_action_index,2);
    
    ai = zeros(1,timestep);
    
%     prev_a = 0;
    for i=1:timestep
       if raw_action_index(1,i)==0
           ai(1,i) = 1;
       elseif raw_action_index(1,i)==1
           ai(1,i) = 0;
       elseif raw_action_index(1,i)==2 && i>1
           ai(1,i) = ai(1,i-1);
       elseif raw_action_index(1,i)==3 && i>1
           ai(1,i) = ai(1,i-1);
       end
       
       
       
    end
    
end