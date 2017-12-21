"""
sub-function for wav reconstruction from enhanced log-magnitude spectrogram and noisy phase
Author: YONG XU, QIUQIANG KONG
Date: 2017.12.21
"""

import os
import librosa
import glob
import numpy as np
import cPickle
import scipy
import sys
from istft import *
from scipy import signal
import soundfile
import spectrogram_to_wave
import config as cfg
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def wav_reco(fn):
    # stft parameters
    fs      = 16000
    n_window= 512
    n_overlap= n_window/2
    #n_fft   = 512
    ham_win = np.hamming(n_window)
    recover_scaler = np.sqrt((ham_win**2).sum())
    
    #fn="TEST_DR1_MRJO0_SI1364"
    (audio, fs) = soundfile.read(cfg.dev_wav_fd+"/%s.WAV"%fn)
    with open(cfg.enh_fea_wav_fd+"/%s.enh_fea"%fn, "rb") as feafile:
    	enh_fea=cPickle.load(feafile)
    enh_fea=enh_fea.T
    enh_fea=np.exp(enh_fea)-1.0e-9 # transform back from log to linear domain
    print "sampling rate:", fs
    print enh_fea.shape

    [f, t, Y_test] = signal.spectral.spectrogram(x=audio, 
                                        window=ham_win, 
                                        nperseg=n_window, 
                                        noverlap=n_overlap, 
                                        detrend=False, 
                                        return_onesided=True, 
                                        scaling='density', 
                                        mode='complex')

    print(Y_test.shape)
    
    enh_wav = spectrogram_to_wave.recover_wav(enh_fea.T, Y_test.T, n_overlap=n_overlap, winfunc=np.hamming, wav_len=len(audio))

    enh_wav *= recover_scaler

    write_audio(os.path.join(cfg.enh_fea_wav_fd, "%s_dnnEnh.wav" %fn), enh_wav, fs)
    #sys.exit()
    ############################################################################################end


