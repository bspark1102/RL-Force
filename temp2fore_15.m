function farr = temp2fore_15(hist, cInd)
    flen = 252;
    tlen = 2881;
    
    farr = zeros(1,10);
    
    if tlen-(cInd+flen) >0
        for i=0:28:flen
            temp = hist(cInd+(i-1)+1:cInd+i+1);
            tmax = max(temp);
            farr(1,(i/28)+1) = tmax;
        end
    else
        temp = hist(cInd:end);
        tmax = max(temp);
        farr(1,:) = tmax;
    end
    
% %     tmin = min(temp);
% 
%     
%     if tmax >= 26
%         farr = 1;
%     else
%         farr = 0;
%     end
    
end