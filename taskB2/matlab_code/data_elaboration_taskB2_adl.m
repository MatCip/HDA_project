% prepare data for python and tensorflow

%% Subject 1 

[~, directory_name] = uigetfile('*.dat');
data1 = load(fullfile(directory_name, 'S1-ADL1.dat'));
data2 = load(fullfile(directory_name, 'S1-ADL2.dat'));
data3 = load(fullfile(directory_name, 'S1-ADL3.dat'));
data4 = load(fullfile(directory_name, 'S1-ADL4.dat'));
data5 = load(fullfile(directory_name, 'S1-ADL5.dat'));

% 0 corresponds to the non-activity class
classes = [0 506616 506617 504616 504617 506620 504620 506605 504605 506619 504619 506611 504611 506608 504608 508612 507621 505606];

% labelled data is data without time column and labels 1-18 (number of classes)
num_cols = 114; % 113 are feature columns - last one labels columns
labels_col = 116;

% INTERPOLATION
method = 'pchip';
end_values = 'nearest';

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

labelled_data5 = zeros(size(data5,1), num_cols);
data=data5;
for i=1:size(data,1)
        labelled_data5(i,1:end-1) = data(i,2:num_cols);
        labelled_data5(i,end) = find(classes == data(i,labels_col));
end

interpolated_data=fillmissing(check_number_nan(labelled_data1), method,1, 'EndValues', end_values);
exp_filename = 'ADL1Opportunity_taskB2_S1.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data2), method,1, 'EndValues', end_values);
exp_filename = 'ADL2Opportunity_taskB2_S1.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data3), method,1, 'EndValues', end_values);
exp_filename = 'ADL3Opportunity_taskB2_S1.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data4), method,1, 'EndValues', end_values);
exp_filename = 'ADL4Opportunity_taskB2_S1.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data5), method,1, 'EndValues', end_values);
exp_filename = 'ADL5Opportunity_taskB2_S1.csv';
csvwrite(exp_filename, interpolated_data);
disp('Subject 1 Processed...');

%% Subject 2

data1 = load(fullfile(directory_name, 'S2-ADL1.dat'));
data2 = load(fullfile(directory_name, 'S2-ADL2.dat'));
data3 = load(fullfile(directory_name, 'S2-ADL3.dat'));
data4 = load(fullfile(directory_name, 'S2-ADL4.dat'));
data5 = load(fullfile(directory_name, 'S2-ADL5.dat'));

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

labelled_data5 = zeros(size(data5,1), num_cols);
data=data5;
for i=1:size(data,1)
        labelled_data5(i,1:end-1) = data(i,2:num_cols);
        labelled_data5(i,end) = find(classes == data(i,labels_col));
end

interpolated_data=fillmissing(check_number_nan(labelled_data1), method,1, 'EndValues', end_values);
exp_filename = 'ADL1Opportunity_taskB2_S2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data2), method,1, 'EndValues', end_values);
exp_filename = 'ADL2Opportunity_taskB2_S2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data3), method,1, 'EndValues', end_values);
exp_filename = 'ADL3Opportunity_taskB2_S2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data4), method,1, 'EndValues', end_values);
exp_filename = 'ADL4Opportunity_taskB2_S2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data5), method,1, 'EndValues', end_values);
exp_filename = 'ADL5Opportunity_taskB2_S2.csv';
csvwrite(exp_filename, interpolated_data);
disp('Subject 2 Processed...');

%% Subject 3

data1 = load(fullfile(directory_name, 'S3-ADL1.dat'));
data2 = load(fullfile(directory_name, 'S3-ADL2.dat'));
data3 = load(fullfile(directory_name, 'S3-ADL3.dat'));
data4 = load(fullfile(directory_name, 'S3-ADL4.dat'));
data5 = load(fullfile(directory_name, 'S3-ADL5.dat'));

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

labelled_data5 = zeros(size(data5,1), num_cols);
data=data5;
for i=1:size(data,1)
        labelled_data5(i,1:end-1) = data(i,2:num_cols);
        labelled_data5(i,end) = find(classes == data(i,labels_col));
end

interpolated_data=fillmissing(check_number_nan(labelled_data1), method,1, 'EndValues', end_values);
exp_filename = 'ADL1Opportunity_taskB2_S3.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data2), method,1, 'EndValues', end_values);
exp_filename = 'ADL2Opportunity_taskB2_S3.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data3), method,1, 'EndValues', end_values);
exp_filename = 'ADL3Opportunity_taskB2_S3.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data4), method,1, 'EndValues', end_values);
exp_filename = 'ADL4Opportunity_taskB2_S3.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data5), method,1, 'EndValues', end_values);
exp_filename = 'ADL5Opportunity_taskB2_S3.csv';
csvwrite(exp_filename, interpolated_data);
disp('Subject 3 Processed...');

%% Subject 4

data1 = load(fullfile(directory_name, 'S4-ADL1.dat'));
data2 = load(fullfile(directory_name, 'S4-ADL2.dat'));
data3 = load(fullfile(directory_name, 'S4-ADL3.dat'));
data4 = load(fullfile(directory_name, 'S4-ADL4.dat'));
data5 = load(fullfile(directory_name, 'S4-ADL5.dat'));

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

labelled_data5 = zeros(size(data5,1), num_cols);
data=data5;
for i=1:size(data,1)
        labelled_data5(i,1:end-1) = data(i,2:num_cols);
        labelled_data5(i,end) = find(classes == data(i,labels_col));
end

interpolated_data=fillmissing(check_number_nan(labelled_data1), method,1, 'EndValues', end_values);
exp_filename = 'ADL1Opportunity_taskB2_S4.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data2), method,1, 'EndValues', end_values);
exp_filename = 'ADL2Opportunity_taskB2_S4.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data3), method,1, 'EndValues', end_values);
exp_filename = 'ADL3Opportunity_taskB2_S4.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data4), method,1, 'EndValues', end_values);
exp_filename = 'ADL4Opportunity_taskB2_S4.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data5), method,1, 'EndValues', end_values);
exp_filename = 'ADL5Opportunity_taskB2_S4.csv';
csvwrite(exp_filename, interpolated_data);
disp('Subject 4 Processed...');
