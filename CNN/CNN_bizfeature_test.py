import numpy as np
import pandas as pd
import h5py
import time

data_root = '/aws_data/'

#photo to business mapping
test_photo_to_biz = pd.read_csv(data_root+'test_photo_to_biz.csv')
biz_ids = test_photo_to_biz['business_id'].unique()

## Load image features
f = h5py.File('/aws_data/test_image_fc7features.h5','r')
image_filenames = list(np.copy(f['photo_id']))
image_filenames = [name.split('/')[-1][:-4] for name in image_filenames]  #remove the full path and the str ".jpg"
image_features = np.copy(f['feature'])
f.close()
print "Number of business: ", len(biz_ids)

df = pd.DataFrame(columns=['business','feature vector'])
index = 0
t = time.time()

for biz in biz_ids:

    image_ids = test_photo_to_biz[test_photo_to_biz['business_id']==biz]['photo_id'].tolist()
    image_index = [image_filenames.index(str(x)) for x in image_ids]

    folder = '/aws_data/test_photos/'
    features = image_features[image_index]
    mean_feature =list(np.mean(features,axis=0))

    df.loc[index] = [biz, mean_feature]
    index+=1
    if index%1000==0:
        print "Input processed: ", index, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"

with open("/aws_data/test_biz_fc7features.csv",'w') as f:
    df.to_csv(f, index=False)