import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Conv2D, LSTM, CuDNNLSTM, Flatten, Dropout, Reshape
from keras import optimizers
from keras import initializers
from keras.utils import to_categorical

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
        train_labels.append(labels[-1])

    val_segments = []
    val_labels = []
    for chunk in val_chunks:
        data = chunk[0]
        labels = chunk[1]
        val_segments.append(data)
        val_labels.append(labels[-1])

    test_segments = []
    test_labels = []
    for chunk in test_chunks:
        data = chunk[0]
        labels = chunk[1]
        test_segments.append(data)
        test_labels.append(labels[-1])

    return np.array(train_segments), np.array(train_labels), np.array(val_segments), np.array(val_labels), np.array(test_segments), np.array(test_labels)

def prepare_data(train_data, val_data, test_data):
    encoder = OneHotEncoder()
    train_labels = encoder.fit_transform(train_data['labels'].values.reshape(-1,1)).toarray()
    val_labels = encoder.transform(val_data['labels'].values.reshape(-1,1)).toarray()
    test_labels = encoder.transform(test_data['labels'].values.reshape(-1,1)).toarray()
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data.drop(['labels'], axis=1, inplace=True)
    val_data.drop(['labels'], axis=1, inplace=True)
    test_data.drop(['labels'], axis=1, inplace=True)
    data = pd.concat([train_data,val_data,test_data])
    scaler.fit(data)
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels

print('Importing data...')
# import train data
adl_1_1 = pd.read_csv("./full_dataset/CIP_interpolation/ADL1Opportunity_taskB2_S1.csv",header=None)
adl_1_2 = pd.read_csv("./full_dataset/CIP_interpolation/ADL2Opportunity_taskB2_S1.csv",header=None)
adl_1_3 = pd.read_csv("./full_dataset/CIP_interpolation/ADL3Opportunity_taskB2_S1.csv",header=None)
adl_1_4 = pd.read_csv("./full_dataset/CIP_interpolation/ADL4Opportunity_taskB2_S1.csv",header=None)
adl_1_5 = pd.read_csv("./full_dataset/CIP_interpolation/ADL5Opportunity_taskB2_S1.csv",header=None)
drill_1 = pd.read_csv("./full_dataset/CIP_interpolation/Drill1Opportunity_taskB2.csv",header=None)
adl_2_1 = pd.read_csv("./full_dataset/CIP_interpolation/ADL1Opportunity_taskB2_S2.csv",header=None)
adl_2_2 = pd.read_csv("./full_dataset/CIP_interpolation/ADL2Opportunity_taskB2_S2.csv",header=None)
drill_2 = pd.read_csv("./full_dataset/CIP_interpolation/Drill2Opportunity_taskB2.csv",header=None)
adl_3_1 = pd.read_csv("./full_dataset/CIP_interpolation/ADL1Opportunity_taskB2_S3.csv",header=None)
adl_3_2 = pd.read_csv("./full_dataset/CIP_interpolation/ADL2Opportunity_taskB2_S3.csv",header=None)
drill_3 = pd.read_csv("./full_dataset/CIP_interpolation/Drill3Opportunity_taskB2.csv",header=None)

# import validation data
adl_2_3 = pd.read_csv("./full_dataset/CIP_interpolation/ADL3Opportunity_taskB2_S2.csv",header=None)
adl_3_3 = pd.read_csv("./full_dataset/CIP_interpolation/ADL3Opportunity_taskB2_S3.csv",header=None)

# import test data
adl_2_4 = pd.read_csv("./full_dataset/CIP_interpolation/ADL4Opportunity_taskB2_S2.csv",header=None)
adl_2_5 = pd.read_csv("./full_dataset/CIP_interpolation/ADL5Opportunity_taskB2_S2.csv",header=None)
adl_3_4 = pd.read_csv("./full_dataset/CIP_interpolation/ADL4Opportunity_taskB2_S3.csv",header=None)
adl_3_5 = pd.read_csv("./full_dataset/CIP_interpolation/ADL5Opportunity_taskB2_S3.csv",header=None)

train_frames = [adl_1_1,adl_1_2,adl_1_3,adl_1_4,adl_1_5,drill_1,adl_2_1,adl_2_2,drill_2,adl_3_1,adl_3_2,drill_3]
val_frames = [adl_2_3,adl_3_3]
test_frames = [adl_2_4,adl_2_5,adl_3_4,adl_3_5]
train_data = pd.concat(train_frames)
val_data = pd.concat(val_frames)
test_data = pd.concat(test_frames)
train_data.rename(columns ={113: 'labels'}, inplace =True)
val_data.rename(columns ={113: 'labels'}, inplace =True)
test_data.rename(columns ={113: 'labels'}, inplace =True)
print("shapes: train {0}, val {1}, test {2}".format(train_data.shape, val_data.shape, test_data.shape))

# scale data between (0,1)
scaled_train, scaled_val, scaled_test, train_labels, val_labels, test_labels = prepare_data(train_data, val_data, test_data)

num_sensors = 113
window_size = 24
step_size = 12
classes = 18

# segment data in sliding windows of size: window_size
train_segments, train_labels, val_segments, val_labels, test_segments, test_labels = segment_data(scaled_train, train_labels, scaled_val, val_labels,
                                                                                                  scaled_test, test_labels, window_size, step_size)

print('Data has been segmented and ready...')

# reshape data for network input
reshaped_train = train_segments.reshape(-1, window_size, num_sensors, 1)
reshaped_val = val_segments.reshape(-1, window_size, num_sensors, 1)
reshaped_test = test_segments.reshape(-1, window_size, num_sensors, 1)

# network parameters
size_of_kernel = (5,1)
kernel_strides = 1
num_filters = 64
num_lstm_cells = 128
dropout_prob = 0.5
inputshape = (window_size, num_sensors, 1)

# BUILDING MODEL USING KERAS AND TENSORFLOW BACKEND
model = Sequential()

model.add(Conv2D(num_filters, kernel_size=size_of_kernel, strides=kernel_strides,
                 activation='relu', input_shape=inputshape,
                 kernel_initializer='glorot_normal', name='1_conv_layer'))
model.add(Conv2D(num_filters, kernel_size=size_of_kernel, strides=kernel_strides,
                 activation='relu',kernel_initializer='glorot_normal',
                 name='2_conv_layer'))
model.add(Conv2D(num_filters, kernel_size=size_of_kernel, strides=kernel_strides,
                 activation='relu',kernel_initializer='glorot_normal',
                 name='3_conv_layer'))
model.add(Conv2D(num_filters, kernel_size=size_of_kernel, strides=kernel_strides,
                 activation='relu',kernel_initializer='glorot_normal',
                 name='4_conv_layer'))
model.add(Reshape((8, num_filters*num_sensors)))

model.add(CuDNNLSTM(num_lstm_cells,kernel_initializer='glorot_normal', return_sequences=True, name='1_lstm_layer'))

model.add(Dropout(dropout_prob, name='2_dropout_layer'))

model.add(CuDNNLSTM(num_lstm_cells,kernel_initializer='glorot_normal',return_sequences=False, name='2_lstm_layer'))

model.add(Dropout(dropout_prob, name='3_dropout_layer'))

model.add(Dense(int(num_lstm_cells/2),kernel_initializer='glorot_normal',
                bias_initializer=initializers.Constant(value=0.1), name='dense_layer'))

model.add(Dense(classes,kernel_initializer='glorot_normal',
                bias_initializer=initializers.Constant(value=0.1),activation='softmax', name='softmax_layer'))

opt = optimizers.RMSprop(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# TRAINING OF THE MODEL
batchSize = 100
train_epochs = 50

model.fit(reshaped_train,train_labels,validation_data=(reshaped_val,val_labels),epochs=train_epochs,batch_size=batchSize,verbose=1)

print('Calculating score.. ')
score = model.evaluate(reshaped_test,test_labels,verbose=1)
print('After {1} epochs test accuracy is: {1}'.format(train_epochs, score[1]))

print('Traning model again over validation data')
train_epochs = 20
all_train = np.concatenate((reshaped_train, reshaped_val))
all_labels = np.concatenate((train_labels, val_labels))
model.fit(all_train,all_labels,validation_data=(reshaped_test,test_labels),epochs=train_epochs,batch_size=batchSize,verbose=1)

print('Calculating score.. ')
score = model.evaluate(reshaped_test,test_labels,verbose=1)
print('After other {1} epochs test accuracy is: {1}'.format(train_epochs, score[1]))

print('Saving model...')
model.save('taskB2_all_Subjects_CNN_LSTM_our_elaboration_full.h5')

# F1-score measure
predictions = model.predict(reshaped_test)
from sklearn.metrics import f1_score
num_classes = 18
class_predictions = []
class_true = []
tot_labels = 0.0
count = 0.0
for pair in zip(predictions, test_labels):
    class_predictions.append(np.argmax(pair[0]))
    class_true.append(np.argmax(pair[1]))
    if np.argmax(pair[0]) == np.argmax(pair[1]):
        count += 1.0
    tot_labels += 1.0

print('Standard accuracy is ' + str(count/tot_labels))

unique, counts = np.unique(class_true, return_counts=True)
counted_labels = dict(zip(unique, counts))
f1_scores = f1_score(class_predictions, class_true, average=None)

tot_f1_score = 0.0
weights_sum = 0.0
for i in range(num_classes):
    labels_class_i = counted_labels[i]
    weight_i = labels_class_i / tot_labels
    weights_sum += weight_i
    tot_f1_score += f1_scores[i]*weight_i
    print(str(i) + ' ' + str(weight_i) + ' ' + str(f1_scores[i]))


print('The weigths sum is ' + str(weights_sum))
print('The computed f1-score is {}'.format(tot_f1_score))
print('The f1-score with sklearn function is {}'.format(f1_score(class_true, class_predictions, average='weighted')))
