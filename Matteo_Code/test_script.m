nan_labelled_data=clean_NAN(new_labelled_data1);
exp_filename = 'ADL2Opportunity_locomotion.csv';
csvwrite(exp_filename, nan_labelled_data);

% 
% new_labelled_data = labelled_data3;
% nan_labelled_data=clean_NAN(new_labelled_data);
% exp_filename = 'ADL3Opportunity_locomotion.csv';
% csvwrite(exp_filename, nan_labelled_data);
% 
% new_labelled_data = labelled_data4;
% nan_labelled_data=clean_NAN(new_labelled_data);
% exp_filename = 'ADL4Opportunity_locomotion.csv';
% csvwrite(exp_filename, nan_labelled_data);
% for column (i.e: temporal data acquired by a sensor)
% replace NaN with last valid value in the sequence

%% 



function nan_labelled_data=clean_NAN(new_labelled_data)
nan_labelled_data = zeros(size(new_labelled_data,1), size(new_labelled_data,2));
for i=1:size(new_labelled_data,2)
   
     %select the column
     
    col = new_labelled_data(:,i);
    indexes=1:length(col);
    idxs = find(isnan(col));
    
    if(length(idxs)~=length(col)) %check if the column is composed whole by NaN
    if ~isempty(idxs)        %  %check if there is no NaN Value
        
        if(idxs(1)==1) % if it is the first 
           start_index=get_next_valid_value(idxs,1);   % get the 
        else
            start_index=1;
        end
        disp(idxs(end))
        if(idxs(end)==length(col)) % if the last index is Nan
           last_index=get_last_valid_value(idxs,length(idxs));   % get the last Value
        else
            last_index=lenght(idxs);
        end
        
        idxs_1=idxs(start_index:last_index);
        j=1;
        next_index=start_index;
        while(next_index+1<length(idxs_1))
           
            next_index=get_next_valid_value(idxs_1,j);
            %linear interpolation
            x1=idxs_1(j)-1;
            x2=idxs_1(next_index)+1;
            y1=col(x1);
            y2=col(x2);
            
            coefficients = polyfit([x1, x2], [y1, y2], 1);
            m = coefficients (1);
            q = coefficients (2);
            index_v=(x1+1):(x2-1);
            
            col(index_v)=m*index_v+q;
            
            j=next_index+1;
        end
    order=10;
    k_step_ahead=length(col)-idxs(last_index)+1;
    forcasted_signal=get_predicted_Nan_series(col(1:idxs(last_index)-1),k_step_ahead,order);
    col=cat(1,col(1:idxs(last_index)-1),forcasted_signal);
    end
    nan_labelled_data(:,i) = col;
    end
end
end 




function forcasted_signal=get_predicted_Nan_series(signal,k_step_ahead,order)
    system=ar(signal,order);
    forcasted_signal = forecast(system,signal,k_step_ahead);
    

end

function next_index=get_next_valid_value(idxs,index)
   j=idxs(index);
   i=index;
   disp(i)
   disp(j)
   while((idxs(i)==j))
      j=j+1;
      i=i+1;
   end
   next_index=i-1;
    
 end




function last_index=get_last_valid_value(idxs,index)
   j=idxs(index);
   i=length(idxs);
  
   while(i>=1 && idxs(i)==j)
      j=j-1;
      i=i-1;

   end
   last_index=i+1;

end
