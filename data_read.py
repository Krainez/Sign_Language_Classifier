import pandas as pd
import numpy as np
from keras import backend as be 
from sklearn.preprocessing import LabelBinarizer

label_binarizer=LabelBinarizer()
train=pd.read_csv('sign_mnist_train.csv')
test=pd.read_csv('sign_mnist_test.csv')

labels_train = train['label'].values
labels_test=test['label'].values

labels_train=label_binarizer.fit_transform(labels_train)
labels_test=label_binarizer.fit_transform(labels_test)

train.drop('label', axis = 1, inplace = True)
test.drop('label', axis = 1, inplace = True)

TRAIN =train.values
TEST=test.values

TRAIN=TRAIN/255
TEST=TEST/255

img_rows, img_cols = 28, 28

if be.image_data_format() == 'channels_first':
    TRAIN = TRAIN.reshape(TRAIN.shape[0], 1, img_rows, img_cols)
    TEST = TEST.reshape(TEST.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    TRAIN = TRAIN.reshape(TRAIN.shape[0], img_rows, img_cols, 1)
    TEST = TEST.reshape(TEST.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

