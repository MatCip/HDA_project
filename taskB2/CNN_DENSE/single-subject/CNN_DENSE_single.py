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
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

is_most_freq = True
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

def segment_data(X_train, y_train, X_test, y_test, winSize, step, noNull=False):
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    # obtain chunks of data
    train_chunks = slidingWindow(X_train, y_train , winSize, step, noNull)
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

    return np.array(train_segments), np.array(train_labels), np.array(test_segments), np.array(test_labels)

def prepare_data(train_data, test_data):

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data

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
adl_1_1 = pd.read_csv("../../full_dataset/ADL1Opportunity_taskB2_S1.csv",header=None)
adl_1_2 = pd.read_csv("../../full_dataset/ADL2Opportunity_taskB2_S1.csv",header=None)
drill_1 = pd.read_csv("../../full_dataset/Drill1Opportunity_taskB2.csv",header=None)

adl_2_1 = pd.read_csv("../../full_dataset/ADL1Opportunity_taskB2_S2.csv",header=None)
adl_2_2 = pd.read_csv("../../full_dataset/ADL2Opportunity_taskB2_S2.csv",header=None)
drill_2 = pd.read_csv("../../full_dataset/Drill2Opportunity_taskB2.csv",header=None)

adl_3_1 = pd.read_csv("../../full_dataset/ADL1Opportunity_taskB2_S3.csv",header=None)
adl_3_2 = pd.read_csv("../../full_dataset/ADL2Opportunity_taskB2_S3.csv",header=None)
drill_3 = pd.read_csv("../../full_dataset/Drill3Opportunity_taskB2.csv",header=None)

adl_4_1 = pd.read_csv("../../full_dataset/ADL1Opportunity_taskB2_S4.csv",header=None)
adl_4_2 = pd.read_csv("../../full_dataset/ADL2Opportunity_taskB2_S4.csv",header=None)
drill_4 = pd.read_csv("../../full_dataset/Drill4Opportunity_taskB2.csv",header=None)

# import validation data
adl_1_3 = pd.read_csv("../../full_dataset/ADL3Opportunity_taskB2_S1.csv",header=None)
adl_2_3 = pd.read_csv("../../full_dataset/ADL3Opportunity_taskB2_S2.csv",header=None)
adl_3_3 = pd.read_csv("../../full_dataset/ADL3Opportunity_taskB2_S3.csv",header=None)
adl_4_3 = pd.read_csv("../../full_dataset/ADL3Opportunity_taskB2_S4.csv",header=None)

# import test data
adl_1_4 = pd.read_csv("../../full_dataset/ADL4Opportunity_taskB2_S1.csv",header=None)
adl_1_5 = pd.read_csv("../../full_dataset/ADL5Opportunity_taskB2_S1.csv",header=None)
adl_2_4 = pd.read_csv("../../full_dataset/ADL4Opportunity_taskB2_S2.csv",header=None)
adl_2_5 = pd.read_csv("../../full_dataset/ADL5Opportunity_taskB2_S2.csv",header=None)
adl_3_4 = pd.read_csv("../../full_dataset/ADL4Opportunity_taskB2_S3.csv",header=None)
adl_3_5 = pd.read_csv("../../full_dataset/ADL5Opportunity_taskB2_S3.csv",header=None)
adl_4_4 = pd.read_csv("../../full_dataset/ADL4Opportunity_taskB2_S4.csv",header=None)
adl_4_5 = pd.read_csv("../../full_dataset/ADL5Opportunity_taskB2_S4.csv",header=None)

train_1 = [adl_1_1, adl_1_2, adl_1_3, drill_1]
test_1 = [adl_1_4, adl_1_5]
train_2 = [adl_2_1, adl_2_2, adl_2_3, drill_2]
test_2 = [adl_2_4, adl_2_5]
train_3 = [adl_3_1, adl_3_2, adl_3_3, drill_3]
test_3 = [adl_3_4, adl_3_5]
train_4 = [adl_4_1, adl_4_2, adl_4_3, drill_4]
test_4 = [adl_4_4, adl_4_5]

def get_user_data(user):
    if user == 1:
        return pd.concat(train_1), pd.concat(test_1)
    elif user == 2:
        return pd.concat(train_2), pd.concat(test_2)
    elif user == 3:
        return pd.concat(train_3), pd.concat(test_3)
    else:
        return pd.concat(train_4), pd.concat(test_4)

f1_scores_users = []
users = [1, 2, 3, 4]
for user in users:

    # saving filename
    filename = 'CNN_DENSE_user_' + str(user)

    train_data, test_data = get_user_data(user)
    train_data.rename(columns ={113: 'labels'}, inplace =True)
    test_data.rename(columns ={113: 'labels'}, inplace =True)
    train_labels = train_data['labels'].values
    test_labels = test_data['labels'].values
    train_data.drop(['labels'], axis=1, inplace=True)
    test_data.drop(['labels'], axis=1, inplace=True)

    # scale data between (0,1)
    scaled_train, scaled_test = prepare_data(train_data, test_data)
    num_sensors = scaled_train.shape[1]
    print('Number of sensor channels is: {}'.format(scaled_train.shape[1]))
    window_size = 24
    step_size = 6
    classes = 18
    print("shapes: train {0}, test {1}".format(scaled_train.shape, scaled_test.shape))

    # segment data in sliding windows of size: window_size
    train_segments, train_labels, test_segments, test_labels = segment_data(scaled_train, train_labels,
                                                                                scaled_test, test_labels, window_size, step_size)
    encoder = OneHotEncoder()
    train_labels = encoder.fit_transform(train_labels.reshape(-1,1)).toarray()
    test_labels = encoder.transform(test_labels.reshape(-1,1)).toarray()
    print('Data has been segmented and ready...')

    # reshape data for network input
    reshaped_train = train_segments.reshape(-1, window_size, num_sensors, 1)
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

    model.add(BatchNormalization(input_shape=inputshape))
    model.add(Conv2D(50, kernel_size=kernel_layer_1, strides=kernel_strides,
                     kernel_initializer='glorot_normal', name='1_conv_layer'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pooling_size))

    model.add(Conv2D(40, kernel_size=kernel_layer_2_3, strides=kernel_strides,
                     kernel_initializer='glorot_normal',name='2_conv_layer'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pooling_size))

    model.add(Conv2D(30, kernel_size=kernel_layer_2_3, strides=kernel_strides,
                     kernel_initializer='glorot_normal',name='3_conv_layer'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(1,1)))

    model.add(Flatten())

    model.add(Dense(1024,kernel_initializer='glorot_normal', bias_initializer=initializers.Constant(value=0.1), activation='relu', name='dense_layer'))

    model.add(Dropout(dropout_prob, name='3_dropout_layer'))

    model.add(Dense(classes,kernel_initializer='glorot_normal',
                    bias_initializer=initializers.Constant(value=0.1),activation='softmax', name='softmax_layer'))

    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    # TRAINING OF THE MODEL
    batchSize = 500
    train_epochs = 50
    my_callback = My_History()
    checkpoint_1 = ModelCheckpoint(filename + '.h5', monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr_1 = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, mode='max', verbose=1, min_lr=0)
    model.fit(reshaped_train,train_labels,validation_data=(reshaped_test,test_labels),epochs=train_epochs,batch_size=batchSize,callbacks=[reduce_lr_1, my_callback, checkpoint_1],verbose=1)
    print('After {0} epochs BEST test accuracy is: {1}, and BEST f1-score is {2}'.format(train_epochs, np.amax(my_callback.test_acc), np.amax(my_callback.f1_scores)))

    f1_scores_users.append(np.amax(my_callback.f1_scores))
    # saving variables
    open_file = 'results_' + filename + '.pkl'
    print('Saving results to: ' + open_file)
    with open(open_file, 'wb') as f:
        pk.dump([my_callback.test_acc, my_callback.f1_scores, my_callback.f1_scores_avg, my_callback.f1_scores_epoch], f)

print('The f1-scores from user 1 to 4 are: ')
print(np.array(f1_scores_users))

with open('f1_scores_per_user.pkl', 'wb') as s:
    pk.dump([np.array(f1_scores_users)], s)
