function av = act2bin_decoupled_ven(raw_action_index,ot_hist)
    
    timestep = size(raw_action_index,2);
    
    av = zeros(1,timestep);
    
%     prev_a = 0;
    for i=1:timestep
       
       ot_temp = ot_hist(i);
       
       if raw_action_index(1,i)==0
           av(1,i) = 0.01;
       elseif raw_action_index(1,i)==1
           av(1,i) = 0.01*ot_temp;
       elseif raw_action_index(1,i)==2 && i>1
           av(1,i) = 0.02*ot_temp;
       elseif raw_action_index(1,i)==3 && i>1
           av(1,i) = 0.05*ot_temp;
       elseif raw_action_index(1,i)==4 && i>1
           av(1,i) = 1;
       elseif raw_action_index(1,i)==5 && i>1
           av(1,i) = 0;
       elseif raw_action_index(1,i)==6 && i>1
           av(1,i) = 0;
       else
           av(1,i) = 99;
       end
       
       
       
    end
    
end