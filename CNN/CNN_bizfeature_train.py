import numpy as np
import pandas as pd
import h5py
import time

DATA = '/aws_data/'

#photo to business mapping
train_photo_to_business = pd.read_csv(DATA + 'train_photo_to_biz_ids.csv')
#train labels
train_labels = pd.read_csv(DATA + 'train.csv').dropna()
train_labels['labels'] = train_labels['labels'].apply(lambda x: tuple(sorted(int(t) for t in x.split())))
train_labels.set_index('business_id', inplace=True)

#print len(train_labels)
f = h5py.File(DATA+'train_image_fc7features.h5','r')
train_image_features = np.copy(f['feature'])
f.close()
print "Input file shape: "
print train_image_features.shape

df = pd.DataFrame(columns=['business', 'label', 'features'])
business_id = train_labels.index
index = 0
for i in business_id:
    label = train_labels.loc[i]['labels']
    image_index = train_photo_to_business[train_photo_to_business['business_id']==i].index.tolist()
    features = train_image_features[image_index]
    features = list(np.mean(features, axis=0))

    df.loc[index] = [i, label, features]
    index = index + 1

with open(DATA+"biz_label_features(train).csv",'w') as f:
    df.to_csv(f, index=False)