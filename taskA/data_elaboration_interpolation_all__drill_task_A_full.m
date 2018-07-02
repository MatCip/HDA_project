

%% prepare data for python and tensorflow

[~, directory_name] = uigetfile('*.dat');
data1 = load(fullfile(directory_name, 'S1-Drill.dat'));
data2 = load(fullfile(directory_name, 'S2-Drill.dat'));
data3 = load(fullfile(directory_name, 'S3-Drill.dat'));
data4 = load(fullfile(directory_name, 'S4-Drill.dat'));

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

disp('Drill1 ')
new_labelled_data = labelled_data1;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'Drill1Opportunity_locomotion.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('Drill2 ')
nan_labelled_data=clean_NAN(labelled_data2);
exp_filename = 'Drill2Opportunity_locomotion.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('Drill3 ')
new_labelled_data = labelled_data3;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'Drill3Opportunity_locomotion.csv';
csvwrite(exp_filename, nan_labelled_data);

disp('Drill4 ')
new_labelled_data = labelled_data4;
nan_labelled_data=clean_NAN(new_labelled_data);
exp_filename = 'Drill4Opportunity_locomotion.csv';
csvwrite(exp_filename, nan_labelled_data);
