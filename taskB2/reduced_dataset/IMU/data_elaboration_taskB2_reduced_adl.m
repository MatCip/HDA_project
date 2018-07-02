% prepare data for python and tensorflow

%% Subject 1 

[~, directory_name] = uigetfile('*.dat');
data1 = load(fullfile(directory_name, 'S1-ADL1.dat'));
data2 = load(fullfile(directory_name, 'S1-ADL2.dat'));
data3 = load(fullfile(directory_name, 'S1-ADL3.dat'));
data4 = load(fullfile(directory_name, 'S1-ADL4.dat'));
data5 = load(fullfile(directory_name, 'S1-ADL5.dat'));

% 0 corresponds to the non-activity class
classes = [0 406516 406517 404516 404517 406520 404520 406505 404505 406519 404519 406511 404511 406508 404508 408512 407521 405506];

% accelerometers 
% acc_columns = 5:7; 
% 23:25 32:34 accelerometri polso
% 14:16 35:37 accelerometri palmo

% IMUs
imu_arms_columns = [64:72 90:98];

all_columns = imu_arms_columns;
num_cols = size(all_columns,2) + 1;
labels_col = 250;

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

labelled_data5 = zeros(size(data5,1), num_cols);
data=data5;
for i=1:size(data,1)
        labelled_data5(i,1:end-1) = data(i, all_columns);
        labelled_data5(i,end) = find(classes == data(i, labels_col));
end

interpolated_data=clean_NAN(labelled_data1);
exp_filename = 'ADL1Opportunity_taskB2_S1.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data2);
exp_filename = 'ADL2Opportunity_taskB2_S1.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data3);
exp_filename = 'ADL3Opportunity_taskB2_S1.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data4);
exp_filename = 'ADL4Opportunity_taskB2_S1.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data5);
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

labelled_data5 = zeros(size(data5,1), num_cols);
data=data5;
for i=1:size(data,1)
        labelled_data5(i,1:end-1) = data(i, all_columns);
        labelled_data5(i,end) = find(classes == data(i, labels_col));
end

interpolated_data=clean_NAN(labelled_data1);
exp_filename = 'ADL1Opportunity_taskB2_S2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data2);
exp_filename = 'ADL2Opportunity_taskB2_S2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data3);
exp_filename = 'ADL3Opportunity_taskB2_S2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data4);
exp_filename = 'ADL4Opportunity_taskB2_S2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data5);
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

labelled_data5 = zeros(size(data5,1), num_cols);
data=data5;
for i=1:size(data,1)
        labelled_data5(i,1:end-1) = data(i, all_columns);
        labelled_data5(i,end) = find(classes == data(i, labels_col));
end

interpolated_data=clean_NAN(labelled_data1);
exp_filename = 'ADL1Opportunity_taskB2_S3.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data2);
exp_filename = 'ADL2Opportunity_taskB2_S3.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data3);
exp_filename = 'ADL3Opportunity_taskB2_S3.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data4);
exp_filename = 'ADL4Opportunity_taskB2_S3.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data5);
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

labelled_data5 = zeros(size(data5,1), num_cols);
data=data5;
for i=1:size(data,1)
        labelled_data5(i,1:end-1) = data(i, all_columns);
        labelled_data5(i,end) = find(classes == data(i, labels_col));
end

interpolated_data=clean_NAN(labelled_data1);
exp_filename = 'ADL1Opportunity_taskB2_S4.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data2);
exp_filename = 'ADL2Opportunity_taskB2_S4.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data3);
exp_filename = 'ADL3Opportunity_taskB2_S4.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data4);
exp_filename = 'ADL4Opportunity_taskB2_S4.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data5);
exp_filename = 'ADL5Opportunity_taskB2_S4.csv';
csvwrite(exp_filename, interpolated_data);
disp('Subject 4 Processed...');
