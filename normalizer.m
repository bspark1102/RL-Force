function normv = normalizer(type, value)


    if type == 1 % flux
%         tf = 1;
        if value >0
            normv = (1-exp(-0.05*value))/(1+exp(-0.05*value));
        else
            normv = (1-exp(-0.18*value))/(1+exp(-0.18*value));
        end
        
    elseif type == 2 % rad
%         tf = 2;

        normv = (1-exp(-0.009*value))/(1+exp(-0.009*value));
%         if value > 50
%             normv = (1-exp(-0.02*value))/(1+exp(-0.02*value));
%         else
%             normv = (1-exp(-0.002*value))/(1+exp(-0.002*value)); 
%         end

    elseif type == 3 % econ
        
    end
    
%     normv = 

end