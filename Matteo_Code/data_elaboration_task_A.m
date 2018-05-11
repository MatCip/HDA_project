

%% prepare data for python and tensorflow

[~, directory_name] = uigetfile('*.dat');
data1 = load(fullfile(directory_name, 'S1-Drill.dat'));
data2 = load(fullfile(directory_name, 'S2-Drill.dat'));
data3 = load(fullfile(directory_name, 'S3-Drill.dat'));
data4 = load(fullfile(directory_name, 'S4-Drill.dat'));

%%
% vector of classes
% 0 corresponds to the non-activity class
classes = [0 101 102 104 105];

% labelled data is data without time column and labels 1-18 (number of classes)
num_cols = 114; % 113 are feature columns - last one labels columns
labels_col = 115;


labelled_data1 = zeros(size(data1,1), num_cols);
data=data1;

for i=1:size(data,1)
        labelled_data1(i,1:end-1) = data(i,2:num_cols);
        labelled_data1(i,end) = find(classes == data(i,labels_col));
end


labelled_data2 = zeros(size(data2,1), num_cols);
data=data2;

for i=1:size(data,1)
        labelled_data2(i,1:end-1) = data(i,2:num_cols);
        labelled_data2(i,end) = find(classes == data(i,labels_col));
end


labelled_data3 = zeros(size(data3,1), num_cols);
data=data3;

for i=1:size(data,1)
        labelled_data3(i,1:end-1) = data(i,2:num_cols);
        labelled_data3(i,end) = find(classes == data(i,labels_col));
end


labelled_data4 = zeros(size(data4,1), num_cols);
data=data4;
for i=1:size(data,1)
        labelled_data4(i,1:end-1) = data(i,2:num_cols);
        labelled_data4(i,end) = find(classes == data(i,labels_col));
end


%% treat the NaN
% column 34 35 36 are always NaN
new_labelled_data = labelled_data1;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'drill1Opportunity_locomotion.csv';
csvwrite(exp_filename, nan_labelled_data);

new_labelled_data = labelled_data2;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'drill2Opportunity_locomotion.csv';
csvwrite(exp_filename, nan_labelled_data);


new_labelled_data = labelled_data3;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'drill3Opportunity_locomotion.csv';
csvwrite(exp_filename, nan_labelled_data);

new_labelled_data = labelled_data4;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'drill4Opportunity_locomotion.csv';
csvwrite(exp_filename, nan_labelled_data);
% for column (i.e: temporal data acquired by a sensor)
% replace NaN with last valid value in the sequence

%% 



function nan_labelled_data=clean_NAN(new_labelled_data)
nan_labelled_data = zeros(size(new_labelled_data,1), size(new_labelled_data,2));
for i=1:size(new_labelled_data,2)
    
    col = new_labelled_data(:,i);
    idxs = find(isnan(col));
    if ~isempty(idxs)
        last = col(idxs(1)-1);
        for j=1:length(idxs)
            col(idxs(j)) = last;
            if j < length(idxs) 
                if idxs(j+1) - idxs(j) > 1
                    last = col(idxs(j+1)-1);
                end
            end
        end
    end
    nan_labelled_data(:,i) = col;
end
end 
