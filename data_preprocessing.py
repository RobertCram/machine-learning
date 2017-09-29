import os, sys
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.datasets import mnist

# Naming Conventions
# X = input
# Y = output
# Tc = target (classifications)
# T = target (one hot encodings)
# P = prediction (one hot encoding probabilities)
# Pc = prediction (classifications)

def getdata_arnold():
    TEST_SAMPLES = 600
    Raw = np.loadtxt('data/arnold_ka-data.csv', delimiter=',')
    np.random.shuffle(Raw)
    X = Raw[:-TEST_SAMPLES,0:-1]
    Tc = Raw[:-TEST_SAMPLES,-1].reshape(-1,1).astype(np.int32)
    X_test = Raw[-TEST_SAMPLES:,0:-1]
    Tc_test = Raw[-TEST_SAMPLES:,-1].reshape(-1,1).astype(np.int32)
    return X, Tc, X_test, Tc_test

def getdata_mnist():
    Raw = np.loadtxt('data/mnist_train.csv'.format(type), delimiter=',')
    X = Raw[:,1:]
    Tc = Raw[:,0].reshape(-1,1).astype(np.int32)
    Raw_test = np.loadtxt('data/mnist_test.csv'.format(type), delimiter=',')
    X_test = Raw_test[:,1:]
    Tc_test = Raw_test[:,0].reshape(-1,1).astype(np.int32)
    return X, Tc, X_test, Tc_test

def getdata_mnist_from_keras():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = x_train.reshape(-1, 28*28).astype(np.float64)
    Tc = y_train.reshape(-1, 1)
    X_test = x_test.reshape(-1, 28*28).astype(np.float64)
    Tc_test = y_test.reshape(-1, 1)
    return X, Tc, X_test, Tc_test

def one_hot_encode(Yc):
    ohe = OneHotEncoder(sparse=False)
    return ohe.fit_transform(Yc)

def scale_data(X, X_test):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X), scaler.transform(X_test)

def convert_to_img (vec, mode, xs, ys, rotation=0):
   byte_image = np.array (vec, dtype = np.uint8)
   img = Image.frombuffer (mode, (xs, ys), byte_image, 'raw', mode, 0, 1).rotate (rotation)
   return img

def save_images_to_disk(Ximg, Tc, width, height, maindir, subdir, rotation=0):
    assert Tc.shape[1] == 1, 'Expected Tc to be a (samples x 1) vector of categories but got {}'.format(Tc.shape)
    assert Ximg.shape[0] == Tc.shape[0], 'Expected same number of samples (rows) for Ximg and Tc, but got {} and {}'.format(Ximg.shape[0], Tc.shape[0])
    mode = 'L' if Ximg.shape[1] == width * height else 'RGB'
    for row in range(Ximg.shape[0]):
        dir = 'data/pictures/{}/{}/class{}'.format(maindir, subdir, Tc[row, 0])
        if not os.path.exists(dir):
            os.makedirs(dir)
        img = convert_to_img(Ximg[row,:], mode, width, height, rotation)
        img.save(dir+'/pic{}.png'.format(row), 'PNG')


