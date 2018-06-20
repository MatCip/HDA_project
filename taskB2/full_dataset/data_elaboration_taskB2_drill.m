%% prepare data for python and tensorflow

[~, directory_name] = uigetfile('*.dat');
data1 = load(fullfile(directory_name, 'S1-Drill.dat'));
data2 = load(fullfile(directory_name, 'S2-Drill.dat'));
data3 = load(fullfile(directory_name, 'S3-Drill.dat'));
data4 = load(fullfile(directory_name, 'S4-Drill.dat'));

%%

% 0 corresponds to the non-activity class
classes = [0 406516 406517 404516 404517 406520 404520 406505 404505 406519 404519 406511 404511 406508 404508 408512 407521 405506];

% labelled data is data without time column and labels 1-18 (number of classes)
keep_cols = [2:46 51:59 64:72 77:85 90:98 103:134]; % 113 are feature columns - last one labels columns
labels_col = 250;

% INTERPOLATION
method = 'linear';
end_values = 'previous';

labelled_data1 = zeros(size(data1,1), size(keep_cols,2)+1);
data=data1;
for i=1:size(data,1)
        labelled_data1(i,1:end-1) = data(i,keep_cols);
        labelled_data1(i,end) = find(classes == data(i,labels_col));
end

labelled_data2 = zeros(size(data2,1), size(keep_cols,2)+1);
data=data2;
for i=1:size(data,1)
        labelled_data2(i,1:end-1) = data(i,keep_cols);
        labelled_data2(i,end) = find(classes == data(i,labels_col));
end

labelled_data3 = zeros(size(data3,1), size(keep_cols,2)+1);
data=data3;
for i=1:size(data,1)
        labelled_data3(i,1:end-1) = data(i,keep_cols);
        labelled_data3(i,end) = find(classes == data(i,labels_col));
end

labelled_data4 = zeros(size(data4,1), size(keep_cols,2)+1);
data=data4;
for i=1:size(data,1)
        labelled_data4(i,1:end-1) = data(i,keep_cols);
        labelled_data4(i,end) = find(classes == data(i,labels_col));
end

labelled_data5 = zeros(size(data5,1), size(keep_cols,2)+1);
data=data5;
for i=1:size(data,1)
        labelled_data5(i,1:end-1) = data(i,keep_cols);
        labelled_data5(i,end) = find(classes == data(i,labels_col));
end

interpolated_data=fillmissing(check_number_nan(labelled_data1), method,1, 'EndValues', end_values);
exp_filename = 'Drill1Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data2), method,1, 'EndValues', end_values);
exp_filename = 'Drill2Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data3), method,1, 'EndValues', end_values);
exp_filename = 'Drill3Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data4), method,1, 'EndValues', end_values);
exp_filename = 'Drill4Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);
