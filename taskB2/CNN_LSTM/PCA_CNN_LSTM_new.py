import pandas as pd
import numpy as np
import pickle as pk
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

import keras as ke
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, CuDNNLSTM, Dropout, Reshape, PReLU, ELU, BatchNormalization, Flatten
from keras import optimizers
from keras import initializers
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.utils import to_categorical

is_most_freq = False
if is_most_freq:
    print('You have chosen to select most frequent label of the window as segment label')

class My_History(ke.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.f1_scores = []
        self.test_acc = []
        self.f1_scores_avg = []
        self.f1_scores_epoch = []

    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        val_accuracy = logs.get('val_acc')
        self.test_acc.append(val_accuracy)
        class_predictions = []
        class_true = []
        predictions = self.model.predict(self.validation_data[0])
        for pair in zip(predictions, self.validation_data[1]):
            class_predictions.append(np.argmax(pair[0]))
            class_true.append(np.argmax(pair[1]))

        f1_scores_i = f1_score(class_predictions, class_true, average=None)
        self.f1_scores_epoch.append(f1_scores_i)
        self.f1_scores_avg.append(np.mean(f1_scores_i))
        self.f1_scores.append(f1_score(class_true, class_predictions, average='weighted'))

    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return

def slidingWindow(sequence, labels, winSize, step, noNull):

    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # number of chunks
    numOfChunks = ((len(sequence)-winSize)//step)+1

    # Do the work
    for i in range(0,numOfChunks*step,step):
        segment = sequence[i:i+winSize]
        seg_labels = labels[i:i+winSize]
        if noNull:
            if seg_labels[-1] != 0:
                yield segment, seg_labels
        else:
            yield segment, seg_labels

def get_most_frequent(labels):

    (values, counts) = np.unique(labels, return_counts=True)
    index = np.argmax(counts)
    return values[index]

def segment_data(X_train, y_train, X_val, y_val, X_test, y_test, winSize, step, noNull=False):
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    # obtain chunks of data
    train_chunks = slidingWindow(X_train, y_train , winSize, step, noNull)
    val_chunks = slidingWindow(X_val, y_val, winSize, step, noNull)
    test_chunks = slidingWindow(X_test, y_test, winSize, step, noNull)

    # segment the data
    train_segments = []
    train_labels = []
    for chunk in train_chunks:
        data = chunk[0]
        labels = chunk[1]
        train_segments.append(data)
        if is_most_freq:
            train_labels.append(get_most_frequent(labels))
        else:
            train_labels.append(labels[-1])

    val_segments = []
    val_labels = []
    for chunk in val_chunks:
        data = chunk[0]
        labels = chunk[1]
        val_segments.append(data)
        if is_most_freq:
            val_labels.append(get_most_frequent(labels))
        else:
            val_labels.append(labels[-1])

    test_segments = []
    test_labels = []
    for chunk in test_chunks:
        data = chunk[0]
        labels = chunk[1]
        test_segments.append(data)
        if is_most_freq:
            test_labels.append(get_most_frequent(labels))
        else:
            test_labels.append(labels[-1])

    return np.array(train_segments), np.array(train_labels), np.array(val_segments), np.array(val_labels), np.array(test_segments), np.array(test_labels)

def prepare_data(train_data, val_data, test_data, channels):

    print('Selecting channles: {}'.format(channels))

    train_data = train_data[channels]
    val_data = val_data[channels]
    test_data = test_data[channels]

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = pd.concat([train_data,val_data,test_data])
    scaler.fit(data)
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    return train_data, val_data, test_data

def get_channels_by_decreasing_variance(data):

    variances = []
    for i in range(data.shape[1]):
        variances.append(np.var(data[:,i]))

    variances = np.array(variances)
    indexes_by_variance = variances.argsort()
    return indexes_by_variance[::-1]

# ********************************************************************************

print('Importing data...')
# import train data
adl_1_1 = pd.read_csv("../full_dataset/ADL1Opportunity_taskB2_S1.csv",header=None)
adl_1_2 = pd.read_csv("../full_dataset/ADL2Opportunity_taskB2_S1.csv",header=None)
drill_1 = pd.read_csv("../full_dataset/Drill1Opportunity_taskB2.csv",header=None)

adl_2_1 = pd.read_csv("../full_dataset/ADL1Opportunity_taskB2_S2.csv",header=None)
adl_2_2 = pd.read_csv("../full_dataset/ADL2Opportunity_taskB2_S2.csv",header=None)
drill_2 = pd.read_csv("../full_dataset/Drill2Opportunity_taskB2.csv",header=None)

adl_3_1 = pd.read_csv("../full_dataset/ADL1Opportunity_taskB2_S3.csv",header=None)
adl_3_2 = pd.read_csv("../full_dataset/ADL2Opportunity_taskB2_S3.csv",header=None)
drill_3 = pd.read_csv("../full_dataset/Drill3Opportunity_taskB2.csv",header=None)

adl_4_1 = pd.read_csv("../full_dataset/ADL1Opportunity_taskB2_S4.csv",header=None)
adl_4_2 = pd.read_csv("../full_dataset/ADL2Opportunity_taskB2_S4.csv",header=None)
drill_4 = pd.read_csv("../full_dataset/Drill4Opportunity_taskB2.csv",header=None)

# import validation data
adl_1_3 = pd.read_csv("../full_dataset/ADL3Opportunity_taskB2_S1.csv",header=None)
adl_2_3 = pd.read_csv("../full_dataset/ADL3Opportunity_taskB2_S2.csv",header=None)
adl_3_3 = pd.read_csv("../full_dataset/ADL3Opportunity_taskB2_S3.csv",header=None)
adl_4_3 = pd.read_csv("../full_dataset/ADL3Opportunity_taskB2_S4.csv",header=None)

# import test data
adl_1_4 = pd.read_csv("../full_dataset/ADL4Opportunity_taskB2_S1.csv",header=None)
adl_1_5 = pd.read_csv("../full_dataset/ADL5Opportunity_taskB2_S1.csv",header=None)
adl_2_4 = pd.read_csv("../full_dataset/ADL4Opportunity_taskB2_S2.csv",header=None)
adl_2_5 = pd.read_csv("../full_dataset/ADL5Opportunity_taskB2_S2.csv",header=None)
adl_3_4 = pd.read_csv("../full_dataset/ADL4Opportunity_taskB2_S3.csv",header=None)
adl_3_5 = pd.read_csv("../full_dataset/ADL5Opportunity_taskB2_S3.csv",header=None)
adl_4_4 = pd.read_csv("../full_dataset/ADL4Opportunity_taskB2_S4.csv",header=None)
adl_4_5 = pd.read_csv("../full_dataset/ADL5Opportunity_taskB2_S4.csv",header=None)

train_frames = [adl_1_1, adl_1_2, drill_1, adl_2_1, adl_2_2, drill_2, adl_3_1, adl_3_2, drill_3, adl_4_1, adl_4_2, drill_4]
val_frames = [adl_1_3, adl_2_3, adl_3_3, adl_4_3]
test_frames = [adl_1_4, adl_1_5, adl_2_4, adl_2_5, adl_3_4, adl_3_5, adl_4_4, adl_4_5]
train_data = pd.concat(train_frames)
val_data = pd.concat(val_frames)
test_data = pd.concat(test_frames)
train_data.rename(columns ={113: 'labels'}, inplace =True)
val_data.rename(columns ={113: 'labels'}, inplace =True)
test_data.rename(columns ={113: 'labels'}, inplace =True)
train_labels_out = train_data['labels'].values
val_labels_out = val_data['labels'].values
test_labels_out = test_data['labels'].values
train_data.drop(['labels'], axis=1, inplace=True)
val_data.drop(['labels'], axis=1, inplace=True)
test_data.drop(['labels'], axis=1, inplace=True)
print("shapes: train {0}, val {1}, test {2}".format(train_data.shape, val_data.shape, test_data.shape))

# extracting increasing number of relevant sensory inputs
n_components = [5, 10, 20, 40, 80]
data = pd.concat([train_data, val_data, test_data]).values
data_centered = data - np.mean(data, axis=0)
channels_by_variance = get_channels_by_decreasing_variance(data_centered)

print('Channels ordered by decreasing variance: ')
print(channels_by_variance)

f1_scores_components = []
for components, in n_components:

    # saving filename
    filename = 'PCA_CNN_LSTM_new_' + str(components)

    # scale data between (0,1)
    scaled_train, scaled_val, scaled_test = prepare_data(train_data, val_data, test_data, channels_by_variance[:components])

    num_sensors = scaled_train.shape[1]
    print('Number of sensor channels is: {}'.format(scaled_train.shape[1]))
    window_size = 24
    step_size = 12
    classes = 18
    print("New shapes: train {0}, val {1}, test {2}".format(scaled_train.shape, scaled_val.shape, scaled_test.shape))

    # segment data in sliding windows of size: window_size
    train_segments, train_labels, val_segments, val_labels, test_segments, test_labels = segment_data(scaled_train, train_labels_out, scaled_val, val_labels_out,
                                                                                                      scaled_test, test_labels_out, window_size, step_size)
    encoder = OneHotEncoder()
    train_labels = encoder.fit_transform(train_labels.reshape(-1,1)).toarray()
    val_labels = encoder.transform(val_labels.reshape(-1,1)).toarray()
    test_labels = encoder.transform(test_labels.reshape(-1,1)).toarray()
    print('Data has been segmented and ready...')

    # reshape data for network input
    reshaped_train = train_segments.reshape(-1, window_size, num_sensors, 1)
    reshaped_val = val_segments.reshape(-1, window_size, num_sensors, 1)
    reshaped_test = test_segments.reshape(-1, window_size, num_sensors, 1)

    # network parameters
    kernel_layer_1 = (3,1)
    kernel_layer_2_3 = (4,1)
    pooling_size = (2,1)
    kernel_strides = 1
    dropout_prob = 0.5
    inputshape = (window_size, num_sensors, 1)

    # BUILDING MODEL USING KERAS AND TENSORFLOW BACKEND
    print('Building Model...')
    model = Sequential()



    print(model.summary())

    # TRAINING OF THE MODEL
    batchSize = 500
    train_epochs = 50
    model.fit(reshaped_train,train_labels,validation_data=(reshaped_val,val_labels),epochs=train_epochs,batch_size=batchSize,verbose=1)
    print('Calculating score.. ')
    score = model.evaluate(reshaped_test,test_labels,verbose=1)
    print('After {0} epochs test accuracy is: {1}'.format(train_epochs, score[1]))

    print('Traning model again adding validation data...')
    train_epochs = 40
    checkpoint_1 = ModelCheckpoint('temporary_model_' + str(components) + '.h5', monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr_1 = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, mode='max', verbose=1, min_lr=0)
    all_train = np.concatenate((reshaped_train, reshaped_val))
    all_labels = np.concatenate((train_labels, val_labels))
    model.fit(all_train,all_labels,validation_data=(reshaped_test,test_labels),epochs=train_epochs,batch_size=batchSize,callbacks=[reduce_lr_1, checkpoint_1],verbose=1)

    print('Traning model again adding validation data...')
    model = load_model('temporary_model_' + str(components) + '.h5')
    train_epochs = 30
    my_callback = My_History()
    checkpoint_2 = ModelCheckpoint(filename + '.h5', monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr_2 = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, mode='max', verbose=1, min_lr=0)
    model.fit(all_train,all_labels,validation_data=(reshaped_test,test_labels),epochs=train_epochs,batch_size=batchSize,callbacks=[reduce_lr_2, my_callback, checkpoint_2],verbose=1)

    print('After other {0} epochs BEST test accuracy is: {1}, and BEST f1-score is {2}'.format(train_epochs, np.amax(my_callback.test_acc), np.amax(my_callback.f1_scores)))

    f1_scores_components.append(np.amax(my_callback.f1_scores))
    # saving variables
    open_file = 'results_' + filename + '.pkl'
    print('Saving results to: ' + open_file)
    with open(open_file, 'wb') as f:
        pk.dump([my_callback.test_acc, my_callback.f1_scores, my_callback.f1_scores_avg, my_callback.f1_scores_epoch], f)

print('The f1-scores for increasing number of components are: ')
print(np.array(f1_scores_components))

with open('f1_scores_per_components.pkl', 'wb') as s:
    pk.dump([np.array(f1_scores_components)], s)

with open('channels_by_variance.pkl', 'wb') as ef:
    pk.dump([channels_by_variance], ef)
