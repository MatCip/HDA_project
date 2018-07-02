

%% prepare data for python and tensorflow

%Subject 1 
[~, directory_name] = uigetfile('*.dat');
data1 = load(fullfile(directory_name, 'S1-ADL1.dat'));
data2 = load(fullfile(directory_name, 'S1-ADL2.dat'));
data3 = load(fullfile(directory_name, 'S1-ADL3.dat'));
data4 = load(fullfile(directory_name, 'S1-ADL4.dat'));
data5 = load(fullfile(directory_name, 'S1-ADL5.dat'));

%%
% vector of classes
% 0 corresponds to the non-activity class
classes = [0 1 2 4 5];

% labelled data is data without time column and labels 1-18 (number of classes)
keep_cols = [2:46 51:59 64:72 77:85 90:98 103:134]; % 113 are feature columns - last one labels columns
labels_col = 244;

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

disp('Processing Subject 1..')
disp('ADL1')
new_labelled_data = labelled_data1;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL1Opportunity_locomotion_S1.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL2')
nan_labelled_data=clean_NAN(labelled_data2);
exp_filename = 'ADL2Opportunity_locomotion_S1.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL3')
new_labelled_data = labelled_data3;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL3Opportunity_locomotion_S1.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL4')
new_labelled_data = labelled_data4;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL4Opportunity_locomotion_S1.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL5')
new_labelled_data = labelled_data5;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL5Opportunity_locomotion_S1.csv';
csvwrite(exp_filename, nan_labelled_data);

%% Subject 2

data1 = load(fullfile(directory_name, 'S2-ADL1.dat'));
data2 = load(fullfile(directory_name, 'S2-ADL2.dat'));
data3 = load(fullfile(directory_name, 'S2-ADL3.dat'));
data4 = load(fullfile(directory_name, 'S2-ADL4.dat'));
data5 = load(fullfile(directory_name, 'S2-ADL5.dat'));

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

disp('Processing Subject 2..')
disp('ADL1')
new_labelled_data = labelled_data1;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL1Opportunity_locomotion_S2.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL2')
nan_labelled_data=clean_NAN(labelled_data2);
exp_filename = 'ADL2Opportunity_locomotion_S2.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL3')
new_labelled_data = labelled_data3;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL3Opportunity_locomotion_S2.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL4')
new_labelled_data = labelled_data4;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL4Opportunity_locomotion_S2.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL5')
new_labelled_data = labelled_data5;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL5Opportunity_locomotion_S2.csv';
csvwrite(exp_filename, nan_labelled_data);

%% Subject 3

data1 = load(fullfile(directory_name, 'S3-ADL1.dat'));
data2 = load(fullfile(directory_name, 'S3-ADL2.dat'));
data3 = load(fullfile(directory_name, 'S3-ADL3.dat'));
data4 = load(fullfile(directory_name, 'S3-ADL4.dat'));
data5 = load(fullfile(directory_name, 'S3-ADL5.dat'));

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

disp('Processing Subject 3..')
disp('ADL1')
new_labelled_data = labelled_data1;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL1Opportunity_locomotion_S3.csv';
csvwrite(exp_filename, nan_labelled_data);

% 
disp('ADL2')
nan_labelled_data=clean_NAN(labelled_data2);
exp_filename = 'ADL2Opportunity_locomotion_S3.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL3')
new_labelled_data = labelled_data3;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL3Opportunity_locomotion_S3.csv';
csvwrite(exp_filename, nan_labelled_data);
disp('ADL4')
new_labelled_data = labelled_data4;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL4Opportunity_locomotion_S3.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL5')
new_labelled_data = labelled_data5;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL5Opportunity_locomotion_S3.csv';
csvwrite(exp_filename, nan_labelled_data);


%% Subject 4

data1 = load(fullfile(directory_name, 'S4-ADL1.dat'));
data2 = load(fullfile(directory_name, 'S4-ADL2.dat'));
data3 = load(fullfile(directory_name, 'S4-ADL3.dat'));
data4 = load(fullfile(directory_name, 'S4-ADL4.dat'));
data5 = load(fullfile(directory_name, 'S4-ADL5.dat'));

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

disp('Processing Subject 4..')
disp('ADL1')
new_labelled_data = labelled_data1;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL1Opportunity_locomotion_S4.csv';
csvwrite(exp_filename, nan_labelled_data);

% 
disp('ADL2')
nan_labelled_data=clean_NAN(labelled_data2);
exp_filename = 'ADL2Opportunity_locomotion_S4.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL3')
new_labelled_data = labelled_data3;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL3Opportunity_locomotion_S4.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL4')
new_labelled_data = labelled_data4;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL4Opportunity_locomotion_S4.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('ADL5')
new_labelled_data = labelled_data5;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'ADL5Opportunity_locomotion_S4.csv';
csvwrite(exp_filename, nan_labelled_data);
