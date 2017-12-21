"""
DNN-based speech enhancement training script based on keras for clean-clean mapping verification
Author: YONG XU
Date: 2017.12.21
"""
import sys
import pickle
import numpy as np
np.random.seed(1515)
import os
import config as cfg
import prepare_fea as pp_data


import keras

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import h5py
from keras.optimizers import SGD,Adam

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, fea_dim*agg_num) )




# hyper-params
fe_fd = cfg.dev_fe_mel_fd
#fe_fd_ori = cfg.dev_fe_mel_fd_ori
agg_num = 7        # concatenate frames
hop = 1            # step_len

n_out = 257

fold = 1           # 1 will be the test set;;; 0 will be the training set
fea_dim=257

# prepare data
scaler = pp_data.GetScaler( fe_fd, fold )

#tr_X, te_X = pp_data.GetAllData_NAT( fe_fd, agg_num, hop, fold, scaler,fea_dim)
tr_X, te_X= pp_data.GetAllData( fe_fd, agg_num, hop, fold, scaler,fea_dim)
#tr_X, te_X = pp_data.GetAllData_noMVN( fe_fd, agg_num, hop, fold,fea_dim)



#tr_X=np.concatenate((tr_X,te_X),axis=0)

### target fea, the middle frame
tr_y=tr_X[:,((agg_num-1)/2)*fea_dim:((agg_num-1)/2+1)*fea_dim] ### arrive at 4*fea_dim-1
te_y=te_X[:,((agg_num-1)/2)*fea_dim:((agg_num-1)/2+1)*fea_dim]
#tr_y=tr_X
#te_y=te_X


print tr_X.shape, tr_y.shape
print te_X.shape, te_y.shape
#(852844, 1799) (852844, 257)
#(311714, 1799) (311714, 257)



###build model by keras
input_audio=Input(shape=(agg_num*fea_dim,))
encoded = Dropout(0.1)(input_audio)

encoded = Dense(2048,activation='relu')(encoded)
encoded = Dropout(0.2)(encoded)

encoded = Dense(2048,activation='relu')(encoded)
encoded = Dropout(0.2)(encoded)

decoded = Dense(2048,activation='relu')(encoded)
#decoded = Dense(fea_dim*agg_num,activation='linear')(decoded) ###7frame out
decoded = Dense(fea_dim,activation='linear')(decoded)   ### the middle frame out

autoencoder=Model(input=input_audio,output=decoded)

autoencoder.summary()

sgd = SGD(lr=0.01, decay=0, momentum=0.9)
autoencoder.compile(optimizer=sgd,loss='mse')

dump_fd=cfg.model_fd+'/sednn_keras_logMag_Relu2048_1outFr_7inFr_dp0.2_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
eachmodel=ModelCheckpoint(dump_fd,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto') 

autoencoder.fit(tr_X,tr_y,nb_epoch=10000,batch_size=100,shuffle=True,validation_data=(te_X,te_y), callbacks=[eachmodel])



