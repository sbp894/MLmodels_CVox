# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:11:23 2022

@author: Satya Parida
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import h5py 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import copy 
from scipy.io import savemat
from sklearn import metrics as skmetric

## 
all_snrs= np.arange(-20.0, 11, 5)
all_snrs= np.append(all_snrs, np.inf)
snr_training_weights= np.flip(len(all_snrs)-np.arange(len(all_snrs)))/len(all_snrs)
print(snr_training_weights)

root_data_dir= '../Datasets/psd_gp_vIHC_mat/PyData/'


# Load clean data 
train_test_dataset = h5py.File(root_data_dir + 'PSD_data_clean.h5', "r")
data_x_clean_orig = np.array(train_test_dataset["data_psd_x"][:]) # your train set features
data_y_clean_orig = np.ravel(np.array(train_test_dataset["data_label_y"][:])) # your train set labels
data_clean_fnames = np.ravel(np.array(train_test_dataset["data_filename"][:])) # your train set labels

train_inds, test_inds = train_test_split(np.arange(data_x_clean_orig.shape[0]), test_size=0.25, random_state=1)

all_train_x_orig = data_x_clean_orig[train_inds,:]
all_train_y_orig = data_y_clean_orig[train_inds]

# Load all SNR data 
for snr_value,snr_weight in zip(all_snrs[0:-1], snr_training_weights[0:-1]):
    print(f"SNR={snr_value} dB and weight = {snr_weight}")
    out_psd_data_file = root_data_dir + 'PSD_data_SNR' + snr_value.astype('int').astype('str') + '.h5'    
    cur_dataset = h5py.File(out_psd_data_file, "r")
        
    cur_x = np.array(cur_dataset["data_psd_x"][:])
    cur_y = np.ravel(np.array(cur_dataset["data_label_y"][:]))
    
    sample_inds_training, _= train_test_split(np.arange(len(train_inds)), test_size= 1-snr_weight, random_state=1)

    all_train_x_orig = np.concatenate((all_train_x_orig, cur_x[train_inds[sample_inds_training],:]), axis=0)
    all_train_y_orig = np.concatenate((all_train_y_orig, cur_y[train_inds[sample_inds_training]]))
    print(f"shape:cur={cur_x.shape},cur_subsamp={len(sample_inds_training)}, full={len(train_inds)}, all_train_x_orig={all_train_x_orig.shape},type={type(cur_x)}")

minmax_scaler = MinMaxScaler()
minmax_scaler.fit(all_train_x_orig)

train_set_x_norm = minmax_scaler.transform(data_x_clean_orig)
all_train_x_norm = minmax_scaler.transform(all_train_x_orig)

X_train_clean, X_test_clean, y_train_clean, y_test_clean, train_inds, test_inds, train_fnames, test_fnames = train_test_split \
    (train_set_x_norm, data_y_clean_orig, np.arange(train_set_x_norm.shape[0]), data_clean_fnames, test_size=0.25, random_state=1)
print(f"Type: X_train_clean={type(X_train_clean)}, y_train_clean={type(y_train_clean)}")
print(f"Shapes: X_train_clean={X_train_clean.shape}, y_train_clean={y_train_clean.shape}, X_test_clean={X_test_clean.shape}, y_test_clean={y_test_clean.shape}")

unq_vals, unq_counts = np.unique(y_train_clean, return_counts=True)
print(dict(zip(unq_vals,unq_counts)))
    

## SVM Model 
svm_model = svm.SVC(kernel='rbf').fit(X_train_clean, y_train_clean)
y_train_pred = svm_model.predict(X_train_clean)
y_test_pred = svm_model.predict(X_test_clean)

print(f"Unique values in y_test_pred={np.unique(y_test_pred)}")
print(f"Unique values in y_train_clean={np.unique(y_train_clean)}")

fig, ax = plt.subplots(2,1,figsize=(4,4))
ax[0].plot(y_train_clean)
ax[1].plot(y_train_clean-y_train_pred, color = "orangered")
ax[1].set_xlabel("Call numbers")

print(f"Training accuracy={np.sum(y_train_clean==y_train_pred)/len(y_train_pred)}\n Testing accuracy={np.sum(y_test_clean==y_test_pred)/len(y_test_pred)}\n")


## 
# tf.random.set_random_seed(1234)  # applied to achieve consistent results
tf.random.set_seed(1234)  # applied to achieve consistent results

# Define the model 
MLPclean = Sequential(
    [
        Dense(10, activation = 'relu',   name = "L1"),
        Dense(5, activation = 'linear', name = "L2")
    ]
)

# Compile the model 
MLPclean.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

# Fit the model 
MLPclean.fit(
    X_train_clean,y_train_clean,
    epochs=200
)


##
y_train_pred_NN = np.argmax(MLPclean.predict(X_train_clean), axis=1)
y_test_pred_NN = np.argmax(MLPclean.predict(X_test_clean), axis=1)

print(f"Unique values in y_train_clean={np.unique(y_train_pred_NN)}")
print(f"Unique values in y_test_pred={np.unique(y_test_pred_NN)}")

fig, ax = plt.subplots(2,1,figsize=(4,4))
ax[0].plot(y_train_clean)
ax[1].plot(y_train_clean-y_train_pred_NN, color = "orangered")
ax[1].set_xlabel("Call numbers")

print(f"Training accuracy={np.sum(y_train_clean==y_train_pred_NN)/len(y_train_clean)}\n Testing accuracy={np.sum(y_test_clean==y_test_pred_NN)/len(y_test_clean)}\n")

##
# tf.random.set_random_seed(1234)  # applied to achieve consistent results
tf.random.set_seed(1234)  # applied to achieve consistent results

# Define the model 
MLPnoisy = Sequential(
    [
        Dense(10, activation = 'relu',   name = "L1"),
        Dense(5, activation = 'linear', name = "L2")
    ]
)

# Compile the model 
MLPnoisy.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=['accuracy'],
)

# Fit the model 
MLPnoisy.fit(
    all_train_x_norm, all_train_y_orig,
    epochs=50
)

##
def test_MLP_snr(MLPmodel, minmax_scaler, snr_value, test_inds_clean, test_fnames_clean):
    root_out_dir= '../Datasets/psd_gp_vIHC_mat/PyData/'
    if np.isinf(snr_value):
        out_psd_data_file = root_out_dir + 'PSD_data_clean.h5'
    else:
        out_psd_data_file = root_out_dir + 'PSD_data_SNR' + snr_value.astype('int').astype('str') + '.h5'
    cur_snr_dataset = h5py.File(out_psd_data_file, "r")
    print(cur_snr_dataset)
    data_x_orig = np.array(cur_snr_dataset["data_psd_x"][:]) # your train set features
    data_y_orig = np.ravel(np.array(cur_snr_dataset["data_label_y"][:])) # your train set labels
    data_fnames = np.ravel(np.array(cur_snr_dataset["data_filename"][:])) # your train set labels
    data_x_norm = minmax_scaler.transform(data_x_orig)

    X_test = data_x_norm[test_inds]
    y_test = data_y_orig[test_inds]
    fName_test= data_fnames[test_inds]

    print(f"-------------------\nWorking on SNR = {snr_value} dB\n--------------------------------------")
    for fileNum,fileName in zip(range(len(fName_test[:5])),fName_test[:5]):
        print(f"File #{fileNum}: {fileName}")
    
    rand_int = random.randint(0, len(test_fnames_clean))
    clean_name = test_fnames_clean[rand_int].decode("utf-8")
    noisy_name= fName_test[rand_int].decode("utf-8")

    # print(f"file number={rand_int} |-| clean={clean_name}|noisy={noisy_name}")
    clean_name= clean_name[clean_name.rfind('/')+1:-3]
    noisy_name= noisy_name[noisy_name.rfind('/')+1:-3]
    if clean_name!=noisy_name:
        print("Should be same")
    y_test_pred_NN = np.argmax(MLPmodel.predict(X_test), axis=1)
    # accuracy = np.sum(y_test==y_test_pred_NN)/len(y_test_clean)
    # auc = skmetric.roc_auc_score(y_test, y_test_pred_NN)
    accuracy = skmetric.accuracy_score(y_test, y_test_pred_NN)
    print(f"[SNR={snr_value} dB] Accuracy={accuracy}\n")
    
    return accuracy

test_accuracy_cleanMLP = np.zeros(all_snrs.shape)
test_accuracy_noisyMLP = np.zeros(all_snrs.shape)
# test_auc_cleanMLP = np.zeros(all_snrs.shape)
# test_auc_noisyMLP = np.zeros(all_snrs.shape)

for snr_value,iter in zip(all_snrs,range(len(all_snrs))):
    test_accuracy_cleanMLP[iter]= test_MLP_snr(MLPclean, minmax_scaler, snr_value, test_inds, test_fnames)
    test_accuracy_noisyMLP[iter]= test_MLP_snr(MLPnoisy, minmax_scaler, snr_value, test_inds, test_fnames)
    # print(f"SNR={snr_value} dB | Accuracy= clean = {test_accuracy_cleanMLP[iter]}, noisy = {test_accuracy_noisyMLP[iter]}")
    
##    
plot_snr= copy.deepcopy(all_snrs)
plot_snr[np.isinf(plot_snr)] = 15
plt.figure(5)
plt.plot(plot_snr, test_accuracy_cleanMLP, label="Clean-trained")
plt.plot(plot_snr, test_accuracy_noisyMLP, label="Noise-trained")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.title("Only carrier frequency (place)")
plt.legend()