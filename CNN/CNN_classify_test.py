from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import time

import numpy as np
import pandas as pd

data_root = '/aws_data/'

train_photos = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
train_photo_to_biz = pd.read_csv(data_root+'train_photo_to_biz_ids.csv', index_col='photo_id')

train_df = pd.read_csv(data_root+"biz_label_features(train).csv")
test_df  = pd.read_csv(data_root+"test_biz_fc7features.csv")

y_train = train_df['label'].values
X_train = train_df['features'].values
X_test = test_df['feature vector'].values

def convert_label_to_array(str_label):
    str_label = str_label[1:-1]
    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x)>0]

def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]

y_train = np.array([convert_label_to_array(y) for y in train_df['label']])
X_train = np.array([convert_feature_to_vector(x) for x in train_df['features']])
X_test = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])

print "X_train: ", X_train.shape
print "y_train: ", y_train.shape
print "X_test: ", X_test.shape

t = time.time()

print "Setting up multi-label binarizer"
mlb = MultiLabelBinarizer()
y_train= mlb.fit_transform(y_train)  #Convert list of labels to binary matrix

classifier = OneVsRestClassifier(svm.SVC(kernel='linear'))
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)