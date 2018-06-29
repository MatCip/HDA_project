import pandas as pd
import numpy as np
import pickle as pk
import sys
from scipy.stats import skew
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC

is_most_freq = True
if is_most_freq:
    print('You have chosen to select most frequent label of the window as segment label')

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

def get_features_from_segment(segments):
    num_features = 8
    features = []
    for i in range(segments.shape[1]):
        segment = segments[:,i]

        # features for each segment
        maxi = np.amax(segment)
        mini = np.amin(segment)
        avg = np.mean(segment)
        stdev = np.std(segment)
        vari = stdev**2
        mediano = np.median(segment)
        skewness = skew(segment)
        autocorr = np.correlate(segment, segment)
        features.append([maxi, mini, avg, stdev, vari, mediano, skewness, autocorr])

    return np.array(features).reshape(-1, num_features*segments.shape[1])


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
        train_segments.append(get_features_from_segment(data))
        if is_most_freq:
            train_labels.append(get_most_frequent(labels))
        else:
            train_labels.append(labels[-1])

    print('features extracted from training data')

    val_segments = []
    val_labels = []
    for chunk in val_chunks:
        data = chunk[0]
        labels = chunk[1]
        val_segments.append(get_features_from_segment(data))
        if is_most_freq:
            val_labels.append(get_most_frequent(labels))
        else:
            val_labels.append(labels[-1])

    print('features extracted from val data')

    test_segments = []
    test_labels = []
    for chunk in test_chunks:
        data = chunk[0]
        labels = chunk[1]
        test_segments.append(get_features_from_segment(data))
        if is_most_freq:
            test_labels.append(get_most_frequent(labels))
        else:
            test_labels.append(labels[-1])

    print('features extracted from test data')

    return np.array(train_segments), np.array(train_labels), np.array(val_segments), np.array(val_labels), np.array(test_segments), np.array(test_labels)

def prepare_data(train_data, val_data, test_data, channels):

    print('Selecting channles: {}'.format(channels))
    train_data = train_data[channels]
    val_data = val_data[channels]
    test_data = test_data[channels]

    return train_data.values, val_data.values, test_data.values

def get_channels_by_decreasing_variance(data):

    variances = []
    for i in range(data.shape[1]):
        variances.append(np.var(data[:,i]))

    variances = np.array(variances)
    indexes_by_variance = variances.argsort()
    return indexes_by_variance[::-1]

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
for components in n_components:

    # saving filename
    filename = 'PCA_MCF_new_' + str(components)

    # scale data between (0,1)
    scaled_train, scaled_val, scaled_test = prepare_data(train_data, val_data, test_data, channels_by_variance[:components])

    num_sensors = scaled_train.shape[1]
    window_size = 24
    step_size = 12
    classes = 18
    print('Number of sensor channels is: {}'.format(scaled_train.shape[1]))

    print("New shapes: train {0}, val {1}, test {2}".format(scaled_train.shape, scaled_val.shape, scaled_test.shape))

    # segment data in sliding windows of size: window_size
    train_segments, train_labels, val_segments, val_labels, test_segments, test_labels = segment_data(scaled_train, train_labels_out, scaled_val, val_labels_out,
                                                                                                      scaled_test, test_labels_out, window_size, step_size)

    # PCA + SVM elaboration
    num_features = 8
    new_train = train_segments.reshape(-1, num_sensors*num_features)
    new_val = val_segments.reshape(-1, num_sensors*num_features)
    new_test = test_segments.reshape(-1, num_sensors*num_features)

    all_train = np.concatenate([new_train, new_val])
    all_labels = np.concatenate([train_labels, val_labels])

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(all_train)
    scaled_test = scaler.transform(new_test)

    pca = PCA(0.99)
    pca.fit(scaled_train)
    print('PCA components: {}'.format(pca.n_components_))
    pca_train = pca.transform(scaled_train)
    pca_test = pca.transform(scaled_test)

    # train a one-vs-rest multi-class support vector machine
    clf = SVC(decision_function_shape='ovr', kernel='linear')
    clf.fit(pca_train, all_labels)

    # predict test data
    svm_pred = clf.predict(pca_test)

    # measure accuracy and f1-score
    num = 0.0
    den = 0.0
    for pair in zip(svm_pred, test_labels):
        if pair[0] == pair[1]:
            num += 1.0
        den += 1.0

    acc = num / den
    print('Test accuracy is: {}'.format(acc))
    f1_scores = f1_score(svm_pred, test_labels, average=None)
    weighted_f1_score = f1_score(test_labels, svm_pred, average='weighted')
    print('The f1-score with sklearn function is {}'.format(weighted_f1_score))
    print('Average f1-score is {}'.format(np.mean(f1_scores)))
    f1_scores_components.append(weighted_f1_score)
    # saving variables
    open_file = 'results_' + filename + '.pkl'
    print('Saving results to: ' + open_file)
    with open(open_file, 'wb') as f:
        pk.dump([acc, weighted_f1_score, np.mean(f1_scores), f1_scores], f)

print('The f1-scores for increasing number of components are: ')
print(np.array(f1_scores_components))

with open('f1_scores_per_components.pkl', 'wb') as s:
    pk.dump([np.array(f1_scores_components)], s)

with open('channels_by_variance.pkl', 'wb') as ef:
    pk.dump([channels_by_variance], ef)
