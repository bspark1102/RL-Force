function new_arr = pushBack(arr,new_val)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A function that pushes back everything one timestep
% -----------------------------------------------------------------------
% Inputs:
%  - arr: an array containing n values (1 x n)
%  - new_val: the most recent value 
%
% Outputs:
%  - new_arr: a new array that has its values pushed back one time step
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    arr_size = size(arr,2);
    new_arr = zeros(1,arr_size);
    
    
    for i=1:arr_size-1
        new_arr(1,i) = arr(1,i+1); 
    end
    
    new_arr(1,end) = new_val;


end