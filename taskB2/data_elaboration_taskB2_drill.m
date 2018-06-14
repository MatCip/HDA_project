%% prepare data for python and tensorflow

[~, directory_name] = uigetfile('*.dat');
data1 = load(fullfile(directory_name, 'S1-Drill.dat'));
data2 = load(fullfile(directory_name, 'S2-Drill.dat'));
data3 = load(fullfile(directory_name, 'S3-Drill.dat'));
data4 = load(fullfile(directory_name, 'S4-Drill.dat'));

%%

% 0 corresponds to the non-activity class
classes = [0 506616 506617 504616 504617 506620 504620 506605 504605 506619 504619 506611 504611 506608 504608 508612 507621 505606];
% labelled data is data without time column and labels 1-18 (number of classes)
num_cols = 114; % 113 are feature columns - last one labels columns
labels_col = 116;

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

% fillmissing(A,'linear',2,'EndValues','nearest')
interpolated_data=fillmissing(check_number_nan(labelled_data1), 'linear',1, 'EndValues', 'previous');
exp_filename = 'Drill1Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data2), 'linear',1, 'EndValues', 'previous');
exp_filename = 'Drill2Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data3), 'linear',1, 'EndValues', 'previous');
exp_filename = 'Drill3Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);

interpolated_data=fillmissing(check_number_nan(labelled_data4), 'linear',1, 'EndValues', 'previous');
exp_filename = 'Drill4Opportunity_taskB2.csv';
csvwrite(exp_filename, interpolated_data);