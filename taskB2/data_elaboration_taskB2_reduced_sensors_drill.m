

%% prepare data for python and tensorflow

[~, directory_name] = uigetfile('*.dat');
data1 = load(fullfile(directory_name, 'S1-Drill.dat'));
data2 = load(fullfile(directory_name, 'S2-Drill.dat'));
data3 = load(fullfile(directory_name, 'S3-Drill.dat'));
data4 = load(fullfile(directory_name, 'S4-Drill.dat'));
%data5 = load(fullfile(directory_name, 'S5-Drill.dat'));
%%
% vector of classes
% 0 corresponds to the non-activity class
classes = [0 506616 506617 504616 504617 506620 504620 506605 504605 506619 504619 506611 504611 506608 504608 508612 507621 505606];
labels_col = 116; % columns with labels for the gestures
number_of_drills = 4;

% accelerometers 
acc_columns = 5:7;

% IMUs
imu_arms_columns = [56:64 74:82];
imu_shoe_columns = 83:114;

all_columns = [acc_columns imu_arms_columns imu_shoe_columns];
num_cols = size(all_columns,2) + 1;

labelled_data1 = zeros(size(data1,1), num_cols);
data=data1;
for i=1:size(data,1)
        labelled_data1(i,1:end-1) = data(i, all_columns);
        labelled_data1(i,end) = find(classes == data(i, labels_col));
end


labelled_data2 = zeros(size(data2,1), num_cols);
data=data2;
for i=1:size(data,1)
        labelled_data2(i,1:end-1) = data(i, all_columns);
        labelled_data2(i,end) = find(classes == data(i, labels_col));
end


labelled_data3 = zeros(size(data3,1), num_cols);
data=data3;
for i=1:size(data,1)
        labelled_data3(i,1:end-1) = data(i, all_columns);
        labelled_data3(i,end) = find(classes == data(i, labels_col));
end


labelled_data4 = zeros(size(data4,1), num_cols);
data=data4;
for i=1:size(data,1)
        labelled_data4(i,1:end-1) = data(i, all_columns);
        labelled_data4(i,end) = find(classes == data(i, labels_col));
end


% labelled_data5 = zeros(size(data5,1), num_cols);
% data=data5;
% for i=1:size(data,1)
%         labelled_data5(i,1:end-1) = data(i, all_columns);
%         labelled_data5(i,end) = find(classes == data(i, labels_col));
% end



%% treat the NaN
% column 34 35 36 are always NaN

disp('Processing column: ')
new_labelled_data = labelled_data1;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'S1-drill-reduced.csv';
csvwrite(exp_filename, nan_labelled_data);

% 
nan_labelled_data=clean_NAN(labelled_data2);
exp_filename = 'S2-drill-reduced.csv';
csvwrite(exp_filename, nan_labelled_data);


new_labelled_data = labelled_data3;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'S3-drill-reduced.csv';
csvwrite(exp_filename, nan_labelled_data);

new_labelled_data = labelled_data4;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'S4-drill-reduced.csv';
csvwrite(exp_filename, nan_labelled_data);


% new_labelled_data = labelled_data5;
% nan_labelled_data=clean_NAN(new_labelled_data);
% exp_filename = 'S5-drill.csv';
% csvwrite(exp_filename, nan_labelled_data);
% for column (i.e: temporal data acquired by a sensor)
% replace NaN with last valid value in the sequence

%% 

function nan_labelled_data=clean_NAN(new_labelled_data)
nan_labelled_data=zeros(size(new_labelled_data));
for i=1:size(new_labelled_data,2)
    disp('Processing column: ')
    disp(i)

    
     %select the column
     
    col = new_labelled_data(:,i);
    indexes=1:length(col);
    idxs = find(isnan(col));
    disp('Number of NaNs detected: ')
    disp(length(idxs))
   
   
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
            last_index=0;
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
        if(last_index~=0)%check last
            x1=idxs(last_index)-1;
            x2=length(col);
            y1=col(x1);
            y2=col(1);
            coefficients = polyfit([x1, x2], [y1, y2], 1);
            m = coefficients (1);
            q = coefficients (2);
            index_v=(x1+1):(length(col));
            col(index_v)=m*index_v+q;
        end
        
        if(start_index~=1)%check first
            x1=length(col);
            x2=idxs(start_index)+1;
            y1=col(x1);
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
 
    end
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


