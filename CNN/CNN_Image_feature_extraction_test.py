import h5py
import caffe
import os
import pandas as pd
import numpy as np
import sys

CAFFE = '/home/ubuntu/caffe/'
DATA = '/aws_data/'
#https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipyb
caffe.set_mode_gpu()

# Assumes that Caffe has been installed. If caffe does not exist, program will terminate

model_def = CAFFE + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = CAFFE + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


def get_features(images):
        layer = 'fc7'

        Transform = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        Transform.set_transpose('data', (2,0,1))
        mu = np.load(CAFFE + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
        mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

        Transform.set_raw_scale('data', 255)
        Transform.set_channel_swap('data', (2,1,0))
        #Resizing the image to 3*227*227
        num_images= len(images)
        net.blobs['data'].reshape(num_images,3,227,227)
        net.blobs['data'].data[...] = map(lambda x: Transform.preprocess('data',caffe.io.load_image(x)), images)
        out = net.forward()

        return net.blobs[layer].data


#file initialisation

WRITE = "/mnt/"
f = h5py.File(WRITE+'test_image_fc7features_200k.h5','w')
filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
feature = f.create_dataset('feature',(0,4096), maxshape = (None,4096))
f.close()

# Credits to ncchen on kaggle for the starter code on test processing

test_photos = pd.read_csv(DATA + 'test_photo_to_biz.csv')
test_folder = DATA + 'test_photos/'
test_images = [os.path.join(test_folder, str(x)+'.jpg') for x in test_photos['photo_id']]

print "Test images: ", len(test_images)
num_train = len(test_images)
batch_size = 200

import time
start = time.time()

for i in range(0, num_train, batch_size):
    images = test_images[i: min(i+batch_size, num_train)]
    features = get_features(images)
    num_done = i+features.shape[0]
    f= h5py.File(WRITE+'test_image_fc7features.h5','r+')
    f['photo_id'].resize((num_done,))
    f['photo_id'][i: num_done] = np.array(images)
    f['feature'].resize((num_done,features.shape[1]))
    f['feature'][i: num_done, :] = features
    f.close()
    if num_done%20000==0 or num_done==num_train:
        print "Test images processed: ", num_done
        end = time.time()
        print "      |    Elapsed time: %fs" % (end - start)