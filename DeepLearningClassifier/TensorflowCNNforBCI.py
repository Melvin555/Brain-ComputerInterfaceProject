# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:55:23 2019

@author: Melvin
"""


import mne
import mxnet as mx
import numpy as np
import pyeeg
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import LinearSegmentedColormap
from numpy.fft import fft2

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.time_frequency import AverageTFR
from mne.time_frequency import tfr_morlet
from mne.viz import ClickableImage
from mne.time_frequency import csd_fourier, csd_multitaper, csd_morlet
from mne import io, compute_raw_covariance, read_events
from mne.preprocessing import Xdawn
from mne.viz import plot_epochs_image
from mne.baseline import rescale
from mne.stats import _bootstrap_ci
from mne.decoding import Vectorizer
from mne.viz import tight_layout
from mne.decoding import UnsupervisedSpatialFilter
from mne.decoding import GeneralizingEstimator
from mne import EvokedArray
from mne.preprocessing import ICA
from mne.decoding import EMS, compute_ems, LinearModel, SlidingEstimator, get_coef
from mne.decoding import Scaler, cross_val_multiscore

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from pyriemann.estimation import Covariances

def center_cmap(cmap, vmin, vmax):
    """Center given colormap (ranging from vmin to vmax) at value 0.

    Note that eventually this could also be achieved by re-normalizing a given
    colormap by subclassing matplotlib.colors.Normalize as described here:
    https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    """  # noqa: E501
    vzero = abs(vmin) / (vmax - vmin)
    index_old = np.linspace(0, 1, cmap.N)
    index_new = np.hstack([np.linspace(0, vzero, cmap.N // 2, endpoint=False),
                           np.linspace(vzero, 1, cmap.N // 2)])
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}
    for old, new in zip(index_old, index_new):
        r, g, b, a = cmap(old)
        cdict["red"].append((new, r, r))
        cdict["green"].append((new, g, g))
        cdict["blue"].append((new, b, b))
        cdict["alpha"].append((new, a, a))
    return LinearSegmentedColormap("erds", cdict)

def make_classifiers():
    """

    :return:
    """

    names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]

    nh = 10

    # pass user defined transfer func
    sinsq = (lambda x: np.power(np.sin(x), 2.0))
    srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)

    # use internal transfer funcs
    srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
    srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
    srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

    # use gaussian RBF
    srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)
    log_reg = LogisticRegression()

    classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                   GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                   GenELMClassifier(hidden_layer=srhl_sinsq),
                   GenELMClassifier(hidden_layer=srhl_tribas),
                   GenELMClassifier(hidden_layer=srhl_hardlim),
                   GenELMClassifier(hidden_layer=srhl_rbf)]

    return names, classifiers

def plot_confusion_matrix_melv(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# load and preprocess the data

event_ids = dict(zero=4, one=5)
subject = 1
tmin, tmax = 10, 20
montage = mne.channels.read_montage('standard_1020')
d1 = {}
d2 = {}
for ext in range(6):
    d1['raw_zero'+str(ext)] = mne.io.read_raw_edf('F:\BCI_Data\Jado\zero\exp'+str(ext)+'\exp'+str(ext)+'.edf',preload=True)
for ext in range(6):
    d2['raw_one'+str(ext)] = mne.io.read_raw_edf('F:\BCI_Data\Jado\one\exp'+str(ext)+'\exp'+str(ext)+'.edf',preload=True)

list_zero = []
list_one = []
for ext in range(len(d1)):
    list_zero.append(d1['raw_zero'+str(ext)])
for ext in range(len(d2)):
    list_one.append(d2['raw_one'+str(ext)])
    
raw_zero = mne.concatenate_raws(list_zero)
events_zero = find_events(raw_zero,shortest_event=0, stim_channel='STI 014', initial_event=True)
raw_one = mne.concatenate_raws(list_one)
events_one = find_events(raw_one,shortest_event=0, stim_channel='STI 014', initial_event=True)
for num in range(len(events_one)):
    events_one[num,2]=5
raw = mne.concatenate_raws(list_zero,list_one) 
events = np.concatenate((events_zero,events_one),0)
raw.set_montage(montage)

raw.info['bads'] = ['COUNTER','INTERPOLATED','RAW_CQ','GYROX','GYROY','MARKER','MARKER_HARDWARE','SYNC','TIME_STAMP_s','TIME_STAMP_ms','CQ_AF3','CQ_F7','CQ_F3','CQ_FC5','CQ_T7','CQ_P7','CQ_O1','CQ_O2','CQ_P8','CQ_T8','CQ_FC6','CQ_F4','CQ_F8','CQ_AF4','CQ_CMS','STI 014']
#raw.info['bads'] = ['COUNTER','INTERPOLATED','RAW_CQ','GYROX','GYROY','MARKER','MARKER_HARDWARE','SYNC','TIME_STAMP_s','TIME_STAMP_ms','CQ_AF3','CQ_F7','CQ_F3','CQ_FC5','CQ_T7','CQ_P7','CQ_O1','CQ_O2','CQ_P8','CQ_T8','CQ_FC6','CQ_F4','CQ_F8','CQ_AF4','CQ_CMS','STI 014']

iir_params = dict(order=2,ftype='butter')
raw.filter(l_freq=8, h_freq=30, method='iir', iir_params=iir_params)
#raw.filter(l_freq=8, h_freq=30, method='fir', fir_window='hann', fir_design='firwin')
##if cropped, caution! Events cannot be detected.
#raw.crop(tmin, tmax).load_data()

#pick all channel
picks = mne.pick_channels(raw.info["ch_names"],["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"])

#picks = mne.pick_channels(raw.info["ch_names"],["AF3","F7","F3","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"])
#picks = mne.pick_channels(raw.info["ch_names"],["O1","O2","P8"])
#picks = mne.pick_channels(raw.info["ch_names"],["O1"])

epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5, picks=picks, baseline=None, preload=True, verbose=True, reject=None)
epochs.equalize_event_counts(event_ids,method='mintime')

labels = epochs.events[:,-1]
evoked = epochs.average()
evoked_zero = epochs['zero'].average()
evoked_one = epochs['one'].average()

#plot evoked
evoked.plot(time_unit='s')
evoked.plot_topomap(time_unit='s')
evoked.plot_image()
evoked.plot_topo()

evoked_zero.plot(time_unit='s')
evoked_zero.plot_topomap(time_unit='s')

evoked_one.plot(time_unit='s')
evoked_one.plot_topomap(time_unit='s')


epochs_data = epochs.get_data()
epochs_data_reshape = epochs_data.reshape((epochs_data.shape[0],epochs_data.shape[1]*epochs_data.shape[2]))


joint_kwargs = dict(ts_args=dict(time_unit='s'),
                    topomap_args=dict(time_unit='s'))
mne.combine_evoked([evoked_zero, -evoked_one], weights='equal').plot_joint(**joint_kwargs)

##Apply fft
#fft_data = []
#for epochs_idx in range(len(epochs_data)):
#    fft_data.append(abs(fft2(epochs_data[epochs_idx]))/sum(epochs_data[epochs_idx]))
#    
#fft_data = np.array(fft_data)




##Apply PCA to the epochs_data
#pca = UnsupervisedSpatialFilter(PCA(14), average=False)
#pca_data = pca.fit_transform(epochs_data)

#Apply ICA to the epochs_data
ica = UnsupervisedSpatialFilter(FastICA(len(picks)), average=False)
ica_data = ica.fit_transform(epochs_data)

##normalizing ICA data
#for epochs_idx in range(len(ica_data)):
#    for channels_idx in range(14):
#        ica_data[epochs_idx,channels_idx] /= ica_data[epochs_idx].sum()

 
ica_data_reshape = ica_data.reshape((ica_data.shape[0],ica_data.shape[1]*ica_data.shape[2]))

#------------------------------------------------------------------------------

#Checking ICA through plot

method = 'fastica'
random_state = 42
ica = ICA(n_components=13, method=method, random_state=random_state)
ica.fit(epochs)
ica.plot_components()
ica.plot_properties(epochs, picks=0)
ica.plot_overlay(evoked, title='Plot Overlay', show=True)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from PIL import Image
import os.path, sys

#zero-produceERP-Image
steps = np.arange(9.5,20.5,0.001)
for step in steps:
    evoked_zero.plot_topomap(times=step, show=False, colorbar=False, time_unit='s')
    plt.savefig('F:\ERP_Test_Subject\Yulia\ERP_test_zero/ERP_zero_time_'+str(step)+'.jpg')
    plt.close('all')

#one-produceERP-Image
steps = np.arange(9.5,20.5,0.001)
for step in steps:
    evoked_one.plot_topomap(times=step, show=False, colorbar=False, time_unit='s')
    plt.savefig('F:\ERP_Test_Subject\Yulia\ERP_test_one/ERP_one_time_'+str(step)+'.jpg')
    plt.close('all')

#zero-Crop
path = "F:\ERP_Test_Subject\Yulia\ERP_test_zero/"
dirs = os.listdir(path)

def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((0, 40, 150, 195)) 
            imCrop = imCrop.crop((0, 0, 150, 135))
            imCrop = imCrop.crop((0, 0, 135, 135))
            imCrop = imCrop.crop((15, 0, 135, 135))
            imCrop.save('F:\ERP_Test_Subject\Yulia\ERP_test_zero_ready/' + item, "JPEG", quality=100)
            
crop()

#one-Crop
path = "F:\ERP_Test_Subject\Yulia\ERP_test_one/"
dirs = os.listdir(path)

def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((0, 40, 150, 195)) 
            imCrop = imCrop.crop((0, 0, 150, 135))
            imCrop = imCrop.crop((0, 0, 135, 135))
            imCrop = imCrop.crop((15, 0, 135, 135))
            imCrop.save('F:\ERP_Test_Subject\Yulia\ERP_test_one_ready/' + item, "JPEG", quality=100)
            
crop()

#zero-Rename
import os
path = "F:\ERP_Test_Subject\Yulia\ERP_test_zero_ready/"
dirs = os.listdir(path)
i = 0
      
for filename in dirs: 
    dst = "zer" + "." + str(i) + ".jpg"
    src = path + filename 
    dst = path + dst 
    os.rename(src, dst) 
    i += 1

#one-Rename
import os
path = "F:\ERP_Test_Subject\Yulia\ERP_test_one_ready/"
dirs = os.listdir(path)
i = 0
      
for filename in dirs: 
    dst = "one" + "." + str(i) + ".jpg"
    src = path + filename 
    dst = path + dst 
    os.rename(src, dst) 
    i += 1
    
#test-folder-rename
#path = "F:\\ERP_Plot\\test/"
#path = "F:\ERP_Test_Subject\TestData/"
path = "F:\Data\A\moment/"
dirs = os.listdir(path)
i = 4896
      
for filename in dirs: 
    dst = "zer." + str(i) + ".jpg"
    src = path + filename 
    dst = path + dst 
    os.rename(src, dst) 
    i += 1
    


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import tensorflow as tf
print(tf.__version__)

import numpy as np 
import os 
from random import shuffle
from tqdm import tqdm
import cv2

#os.chdir("F:\Data")
os.chdir("F:\ERP_Plot")

#Train_dir = r'F:\Data\Ammar\train/'
#Test_dir = r'F:\Data\Ammar\test/'
Train_dir = r'F:\ERP_Plot\train/'
Test_dir = r'F:\ERP_Plot\test/'
IMG_SIZE = 50
LR = 0.0003

MODEL_NAME ='OneZeroImagination-{}-{}.model'.format(LR, 'Recognition-Melvin')

def label_img(img):
    #dog.93.png for labelling
    word_label = img.split('.')[-3]
    if word_label == 'zer': return [1,0]
    elif word_label == 'one': return [0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(Train_dir)):
        label = label_img(img)
        path = os.path.join(Train_dir,img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
        
        
    shuffle(training_data)
    np.save('F:\\ERP_Plot\\train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(Test_dir)):
        label = label_img(img)
        path = os.path.join(Test_dir,img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), np.array(label)])
        
        
    shuffle(testing_data)
    np.save('F:\\ERP_Test_Subject\\Jado\\test_data.npy', testing_data)
    return testing_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(Test_dir)):
        path = os.path.join(Test_dir, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    np.save('F:\\ERP_Test_Subject\\Jado\\train_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
#test_data = create_test_data()
train_data = np.load('F:\\ERP_Plot\\train_data.npy')
#test_data = np.load('F:\\Data\\test_data.npy')

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


#this is the convulation network 

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir ='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model Loaded! YeaY!')

#train = train_data
#test = create_test_data()
train = train_data[:-1400]
test = train_data[-1400:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=50, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

import matplotlib.pyplot as plt
#test_data = process_test_data()
#test_data = np.load('F:\\ERP_Plot\\test_data.npy')

count_one=0; count_zero=0;
fig = plt.figure();
for num, data in enumerate(test_data[:10]):
    # cat is 1:0, 
    # dog is 0:1
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='One'
    else: str_label = 'Zero'
    
    if str_label=='One':
        count_one = count_one+1
    if str_label=='Zero':
        count_zero = count_zero+1
    
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
print('Zero = '+str(count_zero)+' One = '+str(count_one))

#tensorboard --logdir=melvin:C:\Users\Melvin\log




from numpy import random
writer_val = tf.summary.FileWriter('./log/plot_val')
writer_train = tf.summary.FileWriter('./log/plot_train')
loss_var = tf.Variable(0.0)
tf.summary.scalar("loss", loss_var)
write_op = tf.summary.merge_all()
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
for i in range(100):
 # loss validation
 summary = session.run(write_op, {loss_var: random.rand()})
 writer_val.add_summary(summary, i)
 writer_val.flush()
 # loss train
 summary = session.run(write_op, {loss_var: random.rand()})
 writer_train.add_summary(summary, i)
 writer_train.flush()

#------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense

model = Sequential()

model.add(Conv2D(32,(2,2), input_shape=(IMG_SIZE,IMG_SIZE,1) , activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(64,(2,2),activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(32,(2,2),activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(64,(2,2),activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(32,(2,2),activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(64,(2,2),activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(32,(2,2),activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(64,(2,2),activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(32,(2,2),activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(64,(2,2),activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dense(units=1024, activation='relu'))
model.Dropout(0.8)

model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2 , zoom_range=0.2 , horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('train/',(50,50),batch_size=32,class_mode='binary')
model.fit_generator(training_set , steps_per_epoch = 8000,epochs = 25 , validation_steps =2000)

print(history.history.keys())  
   
plt.figure(1)  
   
# summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
  
# summarize history for loss  
  
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()  



















