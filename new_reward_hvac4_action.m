% function [r,r1,r2,r3] = new_reward_hvac1(prevT,currT,a,pa,comfZone,prevR)
function [r,r1,r2,r3] = new_reward_hvac4_action(energy, a, pa)

    % prevT and currT are the previous and curent indoor Temperatures
    % Both are scalar values
    
    % comfZone is the comfort zone in a 1x2 vector, where the first element
    % is the lower bound and second element is the upper bound of the
    % comfort zome
        
    w1 = 1;
    w2 = 0.3;
    w3 = 0.5;
    w4 = 0;
    w5 = 0;
    
    r1 = 0;
    r2 = 0;
    r3 = 0;
    r4 = 0;
    r5 = 0;
    
%     if energy == 0
%         r1 = +10;
%     elseif (energy <= 0.01*10e6) || (energy > 0)
%         r1 = -10;
%     elseif (energy <= 0.3*10e6) || (energy > 0.01*10e6)
%         r1 = -30;
%     elseif (energy <= 1*10e6) || (energy > 0.3*10e6)
%         r1 = -50;
%     else
%         r1 = -100;
%     end
    
    if energy == 0
        r1 = +5;
    else
        ne = energy*5/(10e5);
        r1 = -1*(5+ne);
    end
    
    
%     cent = mean(comfZone);
%     
%     diff = currT - cent;
%     pdiff = prevT - cent;
%     
%     ub = comfZone(2)-cent;
%     lb = -ub;
%     ubd = ub-1;
%     lbd = lb+1;
%     
%     s_flag = 0; % indicates status for staying within/escaping or outside
%                0 1 2 3
%     
%     abs_diff = abs(diff) - abs(pdiff);
%                
%     if currT < comfZone(2) && currT >= comfZone(1) % currently within
%         if prevT < comfZone(2) && prevT >= comfZone(1) % previously was
%             s_flag = 0;
%             r4 = 0;
%         else % previously wasnt, entered comfzone (good case)
%             s_flag = 1;
%             r4 = abs_diff;
%         end
%     else % currently outside
%         if prevT < comfZone(2) && prevT >= comfZone(1) % exit
%            s_flag = 2;
%            r4 = -2*abs_diff^3;
%         else % stayed outside
%            s_flag = 3;
%            r4= -2*abs_diff*2;
%         end
%     end
%                
%                
%     First get the direction----------------
%     if pdiff > 0 && diff >0 % both signs positive, on upper half
%         if pdiff > diff % if previous is larger; got smaller current
%             dir = 1; % moving towards
%         else
%             dir = -1; % moving away
%         end
%     elseif pdiff <= 0 && diff <= 0 % both signs negative, on lower half
%         if pdiff < diff
%             dir = 1; % moving towards
%         else
%             dir = -1; % moving away
%         end
%     else % cross center; signs opposite
%         dir = 0;
%     end
%     -----------------------------
%     
%     
%     Next assign r1 and r2 -----------
%     
%     if (diff > lbd && diff <= ubd) % within center (plateau)
%         r1 = 10 ;
%         
%         if dir == 1 
%             r2 = 0.1;
%         elseif dir == -1
%             r2 = -0.1;
%         else
%             r2 = 0;
%         end
%         
%         bound_stat = 1;
%         
%         
%     elseif (diff <= lbd && diff > lb) % lower deadband
%         r1 = 22 - 8*(diff);
%         r1 = 10 ;
% 
%         if dir == 1 
%             r2 = 0.1;
%         elseif dir == -1
%             r2 = -0.1;
%         else
%             r2 = 0;
%         end
%         
%         bound_stat = 1;
%         
%     elseif (diff > ubd && diff <= ub) % upper deadband
%         r1 = 22 - 8*(diff);
%         
%         if dir == 1 
%             r2 = 0.1;
%         elseif dir == -1
%             r2 = -0.1;
%         else
%             r2 = 0;
%         end
%         
%         bound_stat = 1;
%         
%     elseif (diff > ub) % outside upper bound (hotter)
%         r1 = -8-(diff-4)*(diff+4)*0.8;
%         
%         if dir == 1 
%             r2 = -1*(-8-(diff-4)*(diff+4)*0.8) + 15;
%         elseif dir == -1
%             r2 = -10;
%         else
%             r2 = 0;
%         end
%         
%         bound_stat = 0;
%         
%     elseif (diff <= lb) % outside lower bound (colder)
%         r1 = 4+(25/12)+(13/8)*diff;
%         
%         if dir == 1 
%             r2 = -1*(4+(25/12)+(13/8)*diff)+10;
%         elseif dir == -1
%             r2 = -10;
%         else
%             r2 = 0;
%         end
%         
%         bound_stat = 0;
% 
%         
%     else
%         r1 = 0;
%         aaa = 999
%     end
% 
%     
    if a ~= pa

        r2 = -1;

    else

        r2 = +1;

    end
%     
%     crt = w1*r1 + w2*r2 + w3*r3 + w4*r4;
% 
%     
%     if crt>prevR % good case
%         if crt <= 0
%             r5 = abs(crt)+1;
%         else
%             r5 = 1;
%         end
%     else % bad case
%         if crt <= 0
%             r5 = -5;
%         else
%             r5 = -1;
%         end
%     end
    
    
    r = w1*r1 + w2*r2 + w3*r3 + w4*r4 + w5*r5;

end