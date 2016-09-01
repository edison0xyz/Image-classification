By EDISON LIM 

These files are ran on AWS. 

IMPORTANT: 
It is assumed that the user's computer have the latest version of Caffe model downloaded. 
Please visit caffe.berkeleyvision.org to download the latest Caffe model if you do not have it installed. 

INSTRUCTIONS: 

Files are divided into train or test. THESE FILES RUN DIFFERENT INSTRUCTIONS: 
Run these in sequence: 

For train learning: 
1. CNN_image_feature_extraction_train.py 
2. CNN_classify_train.py
3. CNN_bizfeature_train.py

For test learning: 
1. CNN_image_feature_extraction_test.py 
2. CNN_classify_test.py
3. CNN_bizfeature_test.py



==== Acknowledgements === 

Special acknowledgements to the following for helping in one way or another to make this project possible 

* Tutorials on Caffe: http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
* Stanford CS231n - for releasing your course notes and AMI to the public domain! 
* Kaggle 