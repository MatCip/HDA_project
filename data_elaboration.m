clc;

%% prepare data for python and tensorflow

[~, directory_name] = uigetfile('*.dat');
data1 = load(fullfile(directory_name, 'S1-Drill.dat'));
data2 = load(fullfile(directory_name, 'S2-Drill.dat'));
data3 = load(fullfile(directory_name, 'S3-Drill.dat'));
data4 = load(fullfile(directory_name, 'S4-Drill.dat'));

%%
% vector of classes
% 0 corresponds to the non-activity class
classes = [0 506616 506617 504616 504617 506620 504620 506605 504605 506619 504619 506611 504611 506608 504608 508612 507621 505606];

% labelled data is data without time column and labels 1-18 (number of classes)
num_cols = 114; % 113 are feature columns - last one labels columns
labels_col = 116;
labelled_data1 = zeros(size(data1,1), num_cols);
for i=1:size(data1,1)
    labelled_data1(i,1:end-1) = data1(i,2:num_cols);
    labelled_data1(i,end) = find(classes == data1(i,labels_col));
end

%% 
% export data on  a csv file
exp_filename = 'drill1Opportunity.csv';
csvwrite(exp_filename, labelled_data1);





