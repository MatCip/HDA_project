

%% prepare data for python and tensorflow

[~, directory_name] = uigetfile('*.dat');
data1 = load(fullfile(directory_name, 'S1-Drill.dat'));
data2 = load(fullfile(directory_name, 'S2-Drill.dat'));
data3 = load(fullfile(directory_name, 'S3-Drill.dat'));
data4 = load(fullfile(directory_name, 'S4-Drill.dat'));

%%
% 0 corresponds to the non-activity class
classes = [0 406516 406517 404516 404517 406520 404520 406505 404505 406519 404519 406511 404511 406508 404508 408512 407521 405506];

% accelerometers 
% acc_columns = 5:7; 
% 23:25 32:34 accelerometri polso
% 14:16 35:37 accelerometri palmo

% IMUs
imu_arms_columns = [51:59 64:72 77:85 90:98];

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

interpolated_data=clean_NAN(labelled_data1);
exp_filename = 'Drill1Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data2);
exp_filename = 'Drill2Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data3);
exp_filename = 'Drill3Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=clean_NAN(labelled_data4);
exp_filename = 'Drill4Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);
