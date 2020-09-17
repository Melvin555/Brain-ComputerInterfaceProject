#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:43:59 2018

@author: Melvin

This is a code to classify the imagined digit using the traditional machine 
learning method. SVM, ANN-MLP, Logistic Regression, and LDA are used as the 
classifier.
The data is first read, band-passfiltered within 10-13 Hz, and passsed to ICA
and CSP before being classified using the classifier.


"""

import time
import mne
#import mxnet as mx
import numpy as np
#import pyeeg
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import LinearSegmentedColormap
from numpy.fft import fft2

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
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
for ext in range(100):
    d1['raw_zero'+str(ext)] = mne.io.read_raw_edf('F:\BCI_Data\Melvin\zero\exp'+str(ext)+'\exp'+str(ext)+'.edf',preload=True)
for ext in range(100):
    d2['raw_one'+str(ext)] = mne.io.read_raw_edf('F:\BCI_Data\Melvin\one\exp'+str(ext)+'\exp'+str(ext)+'.edf',preload=True)

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

raw.info['bads'] = ['COUNTER','INTERPOLATED','RAW_CQ','GYROX','GYROY','MARKER','MARKER_HARDWARE','SYNC','TIME_STAMP_s','TIME_STAMP_ms','CQ_AF3','CQ_F7','CQ_F3','CQ_FC5','CQ_T7','CQ_P7','CQ_O1','CQ_O2','CQ_P8','CQ_T8','CQ_FC6','CQ_F4','CQ_F8','CQ_AF4','CQ_CMS','STI 014','FC5']
#raw.info['bads'] = ['COUNTER','INTERPOLATED','RAW_CQ','GYROX','GYROY','MARKER','MARKER_HARDWARE','SYNC','TIME_STAMP_s','TIME_STAMP_ms','CQ_AF3','CQ_F7','CQ_F3','CQ_FC5','CQ_T7','CQ_P7','CQ_O1','CQ_O2','CQ_P8','CQ_T8','CQ_FC6','CQ_F4','CQ_F8','CQ_AF4','CQ_CMS','STI 014']

iir_params = dict(order=2,ftype='butter')
raw.filter(l_freq=10, h_freq=13, method='iir', iir_params=iir_params)
#raw.filter(l_freq=8, h_freq=30, method='fir', fir_window='hann', fir_design='firwin')
##if cropped, caution! Events cannot be detected.
#raw.crop(tmin, tmax).load_data()

#pick all channel
#picks = mne.pick_channels(raw.info["ch_names"],["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"])

#picks = mne.pick_channels(raw.info["ch_names"],["AF3","F7","F3","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"])
picks = mne.pick_channels(raw.info["ch_names"],["O1","O2","P8"])
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
ica = ICA(n_components=3, method=method, random_state=random_state)
ica.fit(epochs)
ica.plot_components()
ica.plot_properties(epochs, picks=0)
ica.plot_overlay(evoked, title='Plot Overlay', show=True)


#------------------------------------------------------------------------------


#Try to differentiate zero and one epochs
#event_id_zero = dict(zero=4)
#event_id_one = dict(one=5)
#epochs_zero = mne.Epochs(raw_zero, events_zero, event_id_zero, tmin - 0.5, tmax + 0.5, picks=picks, baseline=None, preload=True, verbose=True, reject=None)
#epochs_one = mne.Epochs(raw_one, events_one, event_id_one, tmin - 0.5, tmax + 0.5, picks=picks, baseline=None, preload=True, verbose=True, reject=None)


#zero = epochs['zero'].average()
#one = epochs['one'].average()   
#
#joint_kwargs = dict(ts_args=dict(time_unit='s'),
#                    topomap_args=dict(time_unit='s'))
#
#mne.combine_evoked([zero, -one], weights='equal').plot_joint(**joint_kwargs)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Applying pyRiemann XdawnCovariance+TangentSpace+LogisticRegression

# Decoding in tangent space with a logistic regression
start = time.time()
n_components = 1  # pick some components

# Define a monte-carlo cross-validation generator (reduce variance):
cv = KFold(len(labels), 10, shuffle=True, random_state=42)

clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds = np.zeros(len(labels))

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]
    clf.fit(ica_data[train_idx], y_train)
    preds[test_idx] = clf.predict(ica_data[test_idx])
    print(train_idx, test_idx)

##use ShuffleSpit instead of K-Fold
#cv = ShuffleSplit(15, test_size=0.2, random_state=42)
#scores = cross_val_score(clf, ica_data, labels, cv=cv)

# Printing the results
acc = np.mean(preds == labels)
print('Classification accuracy: ' + str(round(acc*100,2)) + '%')

end = time.time()
print("The elapsed time is {} seconds".format(end - start))

tn, fp, fn, tp = confusion_matrix(labels,preds).ravel()
print('True Negative: '+str(tn)+'\n'+
      'False Positive: '+str(fp)+'\n'+
      'False Negative: '+str(fn)+'\n'+
      'True Positive: '+str(tp))

#------------------------------------------------------------------------------
##Another method to plot confusion matrix

#cnf = confusion_matrix(labels,preds)
#plt.figure()
#plot_confusion_matrix_melv(cnf,classes=names,title='confusion_matrix_without_normalization')
#
#plt.figure()
#plot_confusion_matrix_melv(cnf, classes=names, normalize=True,
#                      title='Normalized confusion matrix')

#------------------------------------------------------------------------------

names = ['zero', 'one']
plot_confusion_matrix(preds, labels, names, title='Logistic Regression confusion matrix')

print('Classification report: ')
print(classification_report(labels, preds, target_names=names))

for name in ('patterns_', 'filters_'):
    # The `inverse_transform` parameter will call this method on any estimator
    # contained in the pipeline, in reverse order.
    coef = get_coef(clf, name)
    evoked = EvokedArray(coef, epochs.info, tmin=epochs.tmin)
    evoked.plot_topomap(title='EEG %s' % name[:-1], time_unit='s')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Applying CSP+LDA

#Another method to checck Accuracy
#scores = []
#cv = ShuffleSplit(15, test_size=0.2, random_state=42)

start  = time.time()
# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
cv = KFold(len(labels), 10, shuffle=True, random_state=42)
clf = Pipeline([('CSP', csp), ('LDA', lda)])

##Apply xDawnCovriance and TangentSpace instead of CSP
#n_components = 1
#clf = make_pipeline(XdawnCovariances(n_components),
#                    TangentSpace(metric='riemann'),
#                    LinearDiscriminantAnalysis())

preds = np.zeros(len(labels))

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]
    clf.fit(ica_data[train_idx], y_train)
    preds[test_idx] = clf.predict(ica_data[test_idx])

# Printing the results
acc = np.mean(preds == labels)
print('Classification accuracy: ' + str(round(acc*100,2)) + '%')

end = time.time()
print("The elapsed time is {} seconds".format(end - start))

tn, fp, fn, tp = confusion_matrix(labels,preds).ravel()
print('True Negative: '+str(tn)+'\n'+
      'False Positive: '+str(fp)+'\n'+
      'False Negative: '+str(fn)+'\n'+
      'True Positive: '+str(tp))

names = ['zero', 'one']
plot_confusion_matrix(preds, labels, names, title='LDA confusion matrix')

print('Classification report: ')
print(classification_report(labels, preds, target_names=names))

##Another method to check Accuracy
#scores = cross_val_score(clf, ica_data, labels, cv=cv, n_jobs=5)

## Printing the results
#class_balance = np.mean(labels == labels[0])
#class_balance = max(class_balance, 1. - class_balance)
#print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
#                                                          class_balance))
#
## plot CSP patterns estimated on full data for visualization
#csp.fit_transform(ica_data, labels)
#
#layout = read_layout('EEG1005')
#csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
#                  units='Patterns (AU)', size=1.5)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Applying SVM

#scores = []
#cv = ShuffleSplit(15, test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(ica_data_reshape, labels, test_size=0.2, random_state=0)

start = time.time()

cv = KFold(len(labels), 10, shuffle=True, random_state=42)
clf = make_pipeline(CSP(n_components=len(picks), reg=None, log=True, norm_trace=False),
                    svm.SVC(kernel='rbf', C=1))

#clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#scores = cross_val_score(clf, ica_data_reshape, labels, cv=cv)

##Apply xDawnCovriance and TangentSpace instead of CSP
#n_components = 1
#clf = make_pipeline(XdawnCovariances(n_components),
#                    TangentSpace(metric='riemann'),
#                    svm.SVC(kernel='rbf', C=1))


preds = np.zeros(len(labels))

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]
    clf.fit(ica_data[train_idx], y_train)
    preds[test_idx] = clf.predict(ica_data[test_idx])

# Printing the results
acc = np.mean(preds == labels)
print('Classification accuracy: ' + str(round(acc*100,2)) + '%')

end = time.time()
print("The elapsed time is {} seconds".format(end - start))

tn, fp, fn, tp = confusion_matrix(labels,preds).ravel()
print('True Negative: '+str(tn)+'\n'+
      'False Positive: '+str(fp)+'\n'+
      'False Negative: '+str(fn)+'\n'+
      'True Positive: '+str(tp))

names = ['zero', 'one']
plot_confusion_matrix(preds, labels, names, title='SVM confusion matrix')

print('Classification report: ')
print(classification_report(labels, preds, target_names=names))

#scores = cross_val_score(clf, ica_data_reshape, labels, cv=5)
#
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Applying ANN-MLP


start = time.time()
# Define a monte-carlo cross-validation generator (reduce variance):
cv = KFold(len(labels), 10, shuffle=True, random_state=42)
clf = make_pipeline(CSP(n_components=3, reg=None, log=True, norm_trace=False),
                    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(14, 10, 5, 2), random_state=42))

#clf = make_pipeline(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5, 2), random_state=42))

##Apply xDawnCovriance and TangentSpace instead of CSP
#n_components = 1
#clf = make_pipeline(XdawnCovariances(n_components),
#                    TangentSpace(metric='riemann'),
#                    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(14, 10, 5, 2), random_state=42))

preds = np.zeros(len(labels))

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]
    clf.fit(ica_data[train_idx], y_train)
    preds[test_idx] = clf.predict(ica_data[test_idx])

# Printing the results
acc = np.mean(preds == labels)
print('Classification accuracy: ' + str(round(acc*100,2)) + '%')

end = time.time()
print("The elapsed time is {} seconds".format(end - start))

tn, fp, fn, tp = confusion_matrix(labels,preds).ravel()
print('True Negative: '+str(tn)+'\n'+
      'False Positive: '+str(fp)+'\n'+
      'False Negative: '+str(fn)+'\n'+
      'True Positive: '+str(tp))

names = ['zero', 'one']
plot_confusion_matrix(preds, labels, names, title='ANN-MLP confusion matrix')

print('Classification report: ')
print(classification_report(labels, preds, target_names=names))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Applying ELM

# Define a monte-carlo cross-validation generator (reduce variance):
cv = KFold(len(labels), 10, shuffle=True, random_state=42)
elm_model = MLPRandomLayer(n_hidden=100000, activation_func='tanh')
clf = make_pipeline(CSP(n_components=len(picks), reg=None, log=True, norm_trace=False),
                    GenELMClassifier(hidden_layer=elm_model))

#Apply xDawnCovriance and TangentSpace instead of CSP
n_components = 1
elm_model = MLPRandomLayer(n_hidden=100000, activation_func='tanh')
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    GenELMClassifier(hidden_layer=elm_model))


preds = np.zeros(len(labels))

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]
    clf.fit(ica_data[train_idx], y_train)
    preds[test_idx] = clf.predict(ica_data[test_idx])

# Printing the results
acc = np.mean(preds == labels)
print('Classification accuracy: ' + str(round(acc*100,2)) + '%')

tn, fp, fn, tp = confusion_matrix(labels,preds).ravel()
print('True Negative: '+str(tn)+'\n'+
      'False Positive: '+str(fp)+'\n'+
      'False Negative: '+str(fn)+'\n'+
      'True Positive: '+str(tp))

names = ['zero', 'one']
plot_confusion_matrix(preds, labels, names)

print('Classification report: ')
print(classification_report(labels, preds, target_names=names))
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Applying Linear Classifier + plot patterns & filters -> Vectorizer, StandardScaler, 
#LinearModel, LogisticRegression

clf = make_pipeline(
    Vectorizer(),                       # 1) vectorize across time and channels
    StandardScaler(),                   # 2) normalize features across trials
    LinearModel(
        LogisticRegression(solver='lbfgs')))  # 3) fits a logistic regression
#clf.fit(epochs_data, labels)
cv = KFold(len(labels), 10, shuffle=True, random_state=42)

preds = np.zeros(len(labels))

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]
    clf.fit(ica_data[train_idx], y_train)
    preds[test_idx] = clf.predict(ica_data[test_idx])

# Printing the results
acc = np.mean(preds == labels)
print('Classification accuracy: ' + str(round(acc*100,2)) + '%')

names = ['zero', 'one']
plot_confusion_matrix(preds, labels, names)

layout = read_layout('EEG1005')
topomap_args=dict(layout=layout)
# Extract and plot patterns and filters
for name in ('patterns_', 'filters_'):
    # The `inverse_transform` parameter will call this method on any estimator
    # contained in the pipeline, in reverse order.
    coef = get_coef(clf, name, inverse_transform=True)
    evoked = EvokedArray(coef, epochs.info, tmin=epochs.tmin)
    evoked.plot_topomap(title='EEG %s' % name[:-1], time_unit='s')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Applying StandardScaler, LogisticRegression

clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))

time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')
scores = cross_val_multiscore(time_decod, ica_data, labels, cv=5, n_jobs=1)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')

#------------------------------------------------------------------------------

clf = make_pipeline(StandardScaler(),
                    LinearModel(LogisticRegression(solver='lbfgs')))
time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')
time_decod.fit(ica_data, labels)

coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
evoked = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
joint_kwargs = dict(ts_args=dict(time_unit='s'),
                    topomap_args=dict(time_unit='s'))
evoked.plot_joint(times=np.arange(9.5, 20.5, 1.), title='patterns',
                  **joint_kwargs)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#plot power image (time frequency-analysis)
n_cycles = 5 #number of cycles in Morlet wavelet
freqs = np.arange(13, 30, 3)  # frequencies of interest

power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                        return_itc=True, decim=3, n_jobs=1, use_fft=True)
for electrode_position in power.ch_names:
    power.plot([power.ch_names.index(electrode_position)])

#------------------------------------------------------------------------------

#Compute a cross-spectral density matrix==covariance matrix
n_jobs = 1 #number of cores used
csd_fft = csd_fourier(epochs, fmin=7, fmax = 30, n_jobs=n_jobs)
csd_mt = csd_multitaper(epochs, fmin=7, fmax=30, adaptive=True, n_jobs=n_jobs)
frequencies = np.linspace(7.1,30,1000)
csd_wav = csd_morlet(epochs, frequencies, decim=10, n_jobs=n_jobs)

csd_fft.mean().plot()
plt.suptitle('short-term Fourier transform')

csd_mt.mean().plot()
plt.suptitle('adaptive multitapers')

csd_wav.mean().plot()
plt.suptitle('Morlet wavelet transform')
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


##Plotting the layout manually
#im = plt.imread('F:\BCI_EEG/brain.png')
#lt = mne.channels.read_layout('melvin_brain.lout', path='F:\BCI_EEG' , scale=False)
#x = lt.pos[:, 0] * float(im.shape[1])
#y = (1 - lt.pos[:, 1]) * float(im.shape[0])  # Flip the y-position
#fig, ax = plt.subplots()
#ax.imshow(im)
#ax.scatter(x, y, s=120, color='b')
#plt.autoscale(tight=True)
#ax.set_axis_off()
#plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Part for CSP classification -> checking time
sfreq = raw.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data[train_idx], y_train)
    X_test = csp.transform(epochs_data[test_idx])

    # fit classifier
    lda.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Advance artifact detection
from mne.preprocessing import create_eog_epochs, create_ecg_epochs

eog_average = create_eog_epochs(raw, reject=dict(mag=5e-12, grad=4000e-13),
                                picks=picks).average()

eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation

ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).

ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course

#------------------------------------------------------------------------------
















