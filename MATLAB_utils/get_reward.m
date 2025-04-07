function [r,r1,r2] = get_reward(energy, a, pa)
    w1 = 1;
    w2 = 0.3;
   
    r1 = 0;
    r2 = 0;
    
    if energy == 0
        r1 = +5;
    else
        ne = energy*5/(10e5);
        r1 = -1*(5+ne);
    end
    
    if a ~= pa
        r2 = -1;
    else
        r2 = +1;
    end
    
    r = w1*r1 + w2*r2;
end