function nan_labelled_data=clean_NAN(new_labelled_data)
nan_labelled_data=zeros(size(new_labelled_data));
for i=1:size(new_labelled_data,2)
%     disp('Processing column: ')
%     disp(i)

    
     %select the column
     
    col = new_labelled_data(:,i);
    indexes=1:length(col);
    idxs = find(isnan(col));
%     disp('Number of NaNs detected: ')
%     disp(length(idxs))
   
   
   if(length(idxs)<0.95*length(col)) %check if the column is composed whole by NaN
    if ~isempty(idxs)        %  %check if there is no NaN Value
        
        if(idxs(1)==1) % if it is the first 
           start_index=get_next_valid_value(idxs,1);   % get the 
        else
            start_index=1;
        end
 
        if(idxs(end)==length(col)) % if the last index is Nan
           last_index=get_last_valid_value(idxs,length(idxs));   % get the last Value
        else
            last_index=length(idxs);
        end
        
        idxs_1=idxs(start_index:last_index);
        j=1;
        next_index=2;
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
        if(last_index~=length(idxs) || last_index==1)%check last
            x1=idxs(last_index)-1;
            x2=length(col);
            y1=col(x1);
            if(start_index~=1)
                y2=col(idxs(start_index)+1);
            else
                y2=col(1);
            end
            coefficients = polyfit([x1, x2], [y1, y2], 1);
            m = coefficients (1);
            q = coefficients (2);
            index_v=(x1+1):(length(col));
            col(index_v)=m*index_v+q;
        end
        
        if(start_index~=1)%check first
            x1=1;
            x2=idxs(start_index)+1;
            if(start_index==last_index)
             y1=col(idxs(last_index)+1);
            else
             y1=col(idxs(last_index)-1);
            end
            y2=col(x2);
            
            coefficients = polyfit([x1, x2], [y1, y2], 1);
            m = coefficients (1);
            q = coefficients (2);
            index_v=(1):(x2-1);
            col(index_v)=m*index_v+q;
        end
    
  
%     forcasted_signal=get_predicted_Nan_series(col(1:idxs(last_index)-1),k_step_ahead,order);
%     col=cat(1,col(1:idxs(last_index)-1),forcasted_signal);
    end
   
    nan_labelled_data(:,i)= col;
    
    if(~isempty(find(isnan(col))==1))
    disp(i);
    end
 
    end
end
if(~isempty(find(isnan(nan_labelled_data))==1))
    disp('Nan detected');
end
if(~isempty(find(isinf(nan_labelled_data))==1))
    disp('Infinict detected');
end
end 

function forcasted_signal=get_predicted_Nan_series(signal,k_step_ahead,order)
    number_of_iterations=floor(k_step_ahead/(3*order));
    forcasted_signal=zeros(k_step_ahead,1);
    system=ar(signal,order);
    for i=1:number_of_iterations
       disp(i)
       forcasted_signal((3*order*(i-1)+1):(3*order*(i)))= forecast(system,signal(1:(length(signal)-order*i)),3*order);
    end
    last_processed=3*order*(i);
    forcasted_signal((last_processed+1):k_step_ahead)=forecast(system,signal,k_step_ahead-last_processed);
    
    
  
    

end    
    
function next_index=get_next_valid_value(idxs,index)
   j=idxs(index);
   i=index;
 
   while( i<=length(idxs) && (idxs(i)==j) )
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