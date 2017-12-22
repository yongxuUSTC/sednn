"""
DNN-based speech enhancement decoding script based on keras for clean-clean mapping verification
Author: YONG XU
Date: 2017.12.21
"""

import sys

import pickle
import numpy as np
np.random.seed(1515)
import os
import config as cfg
from preprocessing import reshape_3d_to_4d
import prepare_fea as pp_data
import matplotlib.pyplot as plt
from matplotlib.pyplot import matshow, colorbar,clim,title
from keras.models import load_model
#from prepare_data import load_data
from preprocessing import sparse_to_categorical, mat_2d_to_3d, mat_2d_to_3d_paddingzeros

import keras
import shutil
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import h5py
from keras.optimizers import SGD,Adam
import csv
import cPickle
from keras import backend as K

from wav_reconstruction import wav_reco


# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, fea_dim*agg_num) )

# hyper-params
fe_fd = cfg.dev_fe_mel_fd
agg_num = 7        # concatenate frames
hop = 1            # step_len
n_out = 257
fold = 1          # 1 will be test;;; 0 will be the training set
fea_dim=257


dnn=load_model(cfg.model_fd+'/sednn_keras_logMag_Relu2048layer1_1outFr_7inFr_dp0.2_weights.08-0.01.hdf5')

#scaler = pp_data.GetScaler( fe_fd, fold )
#with open('tr_norm.pickle', 'wb') as handle:
#    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./tr_norm.pickle', 'rb') as handle:
    scaler = pickle.load(handle)
print scaler.mean_, scaler.scale_

def recognize():
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        # read one line
        for li in lis:
            na = li[1]
            curr_fold = int(li[2])
            
            if fold ==curr_fold:          
                fe_path = fe_fd + '/' + na + '.fea'
                print na
                X = cPickle.load( open( fe_path, 'rb' ) )
                X=X.T

                
                print X.shape
	        len_X, n_in = X.shape
      		#padding zeros in the left begining with (agg_num-1)/2 zeros:
      		X = np.concatenate( ( np.zeros(((agg_num-1)/2, n_in))+ (-50.0), X  ) )  ### due to log, so -50.0
       		#padding zeros in the right end with (hop-1)/2 zeros:
       		X = np.concatenate( ( X, np.zeros(((agg_num-1)/2, n_in))+ (-50.0)  ) )  ### due to log, so -50.0
	        #print X

                X = scaler.transform( X )

                #X3d = mat_2d_to_3d( X, agg_num, hop )
                X3d = mat_2d_to_3d_paddingzeros( X, agg_num, hop ) # to make the frame num is the same after frame expansion and enhancement
                print X3d.shape
                X3d= reshapeX(X3d)
                
                pred = dnn.predict( X3d ) 
                print pred.shape #(203, 257)

                # inverse_transform to reconstruct spectorgram
                pred = scaler.inverse_transform(pred)
         	## DEBUG. print mel-spectrogram
        	#print pred
        	#plt.matshow(pred.T, origin='lower', aspect='auto')
        	#plt.show()
       		#sys.exit()   
            
                out_path = cfg.enh_fea_wav_fd + '/' + na + '.enh_fea'
                cPickle.dump( pred, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
                wav_reco(na)
                print 'write fea and reconstruct wav done!'
                sys.exit()              


if __name__ == '__main__':
    recognize()
