function farr = temp2foreV3(hist, cInd)
    flen = 378;
    tlen = 4463;
    
    farr = zeros(1,10);
    
    if tlen-(cInd+flen) >0
        for i=0:42:flen
            temp = hist(cInd+(i-1)+1:cInd+i+1);
            tmax = max(temp);
            farr(1,(i/42)+1) = tmax;
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