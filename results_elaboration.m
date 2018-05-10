clc;

%% this a matalb routine for results elaboration

predictions = csvread('preds_test.csv');
trues = csvread('true_test.csv');

%%

hot_predictions = zeros(size(predictions,1), size(predictions,2));

for i=1:size(predictions,1)
    
    pred = predictions(i,:);
    [max_val, index] = max(pred);
    pred(index) = 1;
    pred(pred < max_val) = 0;
    hot_predictions(i,:) = pred;
end

%%
% plot confusion matrix
labels = {'Null Class', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'Cloose Door 2', 'Open Fridge', 'Close Fridge',...
        'Open DishWas', 'Close DishWas', 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2', 'Close Drawer 2',...
        'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink Cup', 'Toggle Switch'};
numlabels = 18;
figure;
plotconfusion(trues', hot_predictions');
set(gca,'XTick',1:numlabels,...
    'XTickLabel',labels,...
    'YTick',1:numlabels,...
    'YTickLabel',labels,...
    'TickLength',[0 0]);






