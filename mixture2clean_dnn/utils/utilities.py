import numpy as np
import soundfile
import librosa
import os
import h5py

from stft import stft
import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


def get_stft_window_func(window_type):
    if window_type == 'hamming':
        return np.hamming
        
    else:
        raise Exception('Incorrect window_type!')


def calculate_spectrogram(audio, mode):
    """Calculate spectrogram. 
    
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    window_size = config.window_size
    overlap = config.overlap
    hop_size = window_size - overlap
    window = get_stft_window_func(config.window_type)(window_size)
    
    x = stft(x=audio, window_size=window_size, hop_size=hop_size, 
        window=window, mode=mode)
    
    if mode == 'magnitude':
        x = x.astype(np.float32)
        
    elif mode == 'complex':
        x = x.astype(np.complex64)
        
    else:
        raise Exception('Incorrect mode!')
        
    return x
    
    
def log_sp(x):
    return np.log(x + 1e-08)
    
    
def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments. 
    """
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)


def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value. 
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)
    
    
def calculate_scaler(x, axis):
    mean = np.mean(x, axis)
    std = np.std(x, axis)
    scaler = {'mean': mean, 'std': std}
    return scaler
    
    
def scale(x, scaler):
    return (x - scaler['mean']) / scaler['std']
    
    
def inverse_scale(x, scaler):
    return x * scaler['std'] + scaler['mean']
    
    
def load_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf['x'][:]      # (n_segs, n_concat, n_freq)
        y = hf['y'][:]      # (n_segs, n_freq)
        
    return x, y


def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))