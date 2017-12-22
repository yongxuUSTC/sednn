'''
SUMMARY:  extract log-magnitude spectrogram feature and prepare fea for training
AUTHOR:   YONG XU, QIUQIANG KONG
--------------------------------------
'''
import sys
from preprocessing import mat_2d_to_3d, reshape_3d_to_4d, mat_2d_to_3d_paddingzeros
import numpy as np
from scipy import signal
import cPickle
import os
import sys
import matplotlib.pyplot as plt
from scipy import signal
import wavio
import librosa
import config as cfg
import csv
import scipy.stats
from sklearn import preprocessing
#import scikits.talkbox.features.mfcc as mfcc
#import htkmfc


def reshapeX( X ,fea_dim,agg_num):
    N = len(X)
    return X.reshape( (N, agg_num*fea_dim) )
    
### readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

# calculate mel feature
def GetMel( wav_fd, fe_fd, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.WAV') ]
    extlen=len('WAV')+1
    #print extlen
    #sys.exit()
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        #print wav.shape
        #print fs
        #sys.exit()
        if ( wav.ndim==2 ): 
            wav_m = np.mean( wav, axis=-1 ) # mean
        assert fs==cfg.fs
        ham_win = np.hamming(cfg.win)
        
        
        [f_m, t_m, X_m] = signal.spectral.spectrogram( wav_m, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=True, mode='magnitude' )
        #X_m = X_m.T
        #[ceps, mspec, spec]=mfcc(wav_m, nwin=cfg.win, overlap=cfg.win/2, nfft=512, fs=16000, nceps=24)
        #mspec: Log-spectrum in the mel-domain;; ceps: Mel-cepstrum coefficients
        #X_m=ceps
        #print ceps.shape, mspec.shape, spec.shape #(399, 24) (399, 40) (399, 512), why is 512 for spec ?
        #(99, 24) (99, 40) (99, 512)
        #sys.exit()

        X_m=np.log(X_m+1.0e-9) # loge for better hearing perception
        # DEBUG. print mel-spectrogram
        #print X_m
        #plt.matshow(X_m, origin='lower', aspect='auto')
        #plt.show()
        #sys.exit()
        
        out_path = fe_fd + '/' + na[0:-extlen] + '.fea'
        cPickle.dump( X_m, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )

###
# get chunk data, size: N*agg_num*n_in
def GetAllData( fe_fd, agg_num, hop, fold , scaler,fea_dim):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1] ### file name
        curr_fold = int(li[2])  ### train (0) or test (1)
        
        # get features
        fe_path = cfg.dev_fe_mel_fd + '/' + na + '.fea'

        X = cPickle.load( open( fe_path, 'rb' ) )
        X=X.T 
        #print X.shape
        
        

        len_X, n_in = X.shape
        #print X.shape
        #padding zeros in the left begining with (agg_num-1)/2 zeros:
        X = np.concatenate( ( np.zeros(((agg_num-1)/2, n_in))+ (-50.0), X  ) )  ### due to log, so -50.0
        #padding zeros in the right end with (hop-1)/2 zeros:
        X = np.concatenate( ( X, np.zeros(((agg_num-1)/2, n_in))+ (-50.0)  ) )  ### due to log, so -50.0
        #print X
        #sys.exit()
        

        X = scaler.transform( X )
        

        # aggregate data
        #print X_l.shape #(nframe=125,ndim=257)
        #X3d = mat_2d_to_3d( X, agg_num, hop )
        X3d = mat_2d_to_3d_paddingzeros( X, agg_num, hop )
        X3d= reshapeX(X3d,fea_dim, agg_num)
        #print X3d_l.shape # (nsampelPERutt=10,contextfr=33,ndim=257)
        # reshape 3d to 4d
        #X4d_l = reshape_3d_to_4d( X3d_l)
        #X4d_r = reshape_3d_to_4d( X3d_r)
        #X4d_m = reshape_3d_to_4d( X3d_m)
        #X4d_d = reshape_3d_to_4d( X3d_d)
        # concatenate
        #X4d=mat_concate_multiinmaps4in(X3d_l, X3d_r, X3d_m, X3d_d)
        #print X4d.shape      
        #sys.exit()       
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            #te_ylist += [ y ] * len( X3d )
        else:
            tr_Xlist.append( X3d )
            #tr_ylist += [ y ] * len( X3d )

#    return np.concatenate( tr_Xlist, axis=0 ), np.array( tr_ylist ),\
#           np.concatenate( te_Xlist, axis=0 ), np.array( te_ylist )
    return np.concatenate( tr_Xlist, axis=0 ), np.concatenate( te_Xlist, axis=0 )
           
def GetScaler( fe_fd, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist = []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = cfg.dev_fe_mel_fd + '/' + na + '.fea'
        X = cPickle.load( open( fe_path, 'rb' ) )
        X = X.T  ##(nframe, 257)
        if curr_fold!=fold:
            tr_Xlist.append( X )
            #print np.array(tr_Xlist).shape
            
    Xall = np.concatenate( tr_Xlist, axis=0 )
    print Xall.shape  ##(321794, 257)
    scaler = preprocessing.StandardScaler( with_mean=True, with_std=True ).fit( Xall)
    print scaler.mean_, scaler.scale_
    return scaler
    
###
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
            
if __name__ == "__main__":
    CreateFolder( cfg.scrap_fd + '/fe' )
    CreateFolder( cfg.scrap_fd + '/fe/log_mag_spec' )
    CreateFolder( cfg.scrap_fd + '/results' )
    CreateFolder( cfg.scrap_fd + '/md' )
    GetMel( cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0 )
    #GetHTKfea( cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0 )
