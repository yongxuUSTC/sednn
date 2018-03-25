"""
Summary:  Prepare data. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: - 
"""
import os
import soundfile
import numpy as np
import argparse
import csv
import time
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import h5py
from sklearn import preprocessing
import torch
from torch.autograd import Variable
import math

import config as cfg
import stft


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

def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def normalize(x):
    return x / np.max(np.abs(x))
    
    
def calculate_energy(x):
    return np.mean(np.abs(x) ** 2)


def calculate_features(args):
    
    workspace = args.workspace
    tr_speech_dir = args.tr_speech_dir
    tr_noise_dir = args.tr_noise_dir
    te_speech_dir = args.te_speech_dir
    te_noise_dir = args.te_noise_dir
    
    sample_rate = cfg.sample_rate
    fft_size = cfg.fft_size
    hop_size = cfg.hop_size
    window_type = cfg.window_type
    
    # Create hdf5
    hdf5_path = os.path.join(workspace, "features", "cmplx_spectrogram.h5")
    create_folder(os.path.dirname(hdf5_path))
    
    with h5py.File(hdf5_path, 'w') as hf:
        
        hf.attrs['sample_rate'] = sample_rate
        hf.attrs['fft_size'] = fft_size
        hf.attrs['hop_size'] = hop_size
        hf.attrs['window_type'] = window_type
    
    # Write out features to hdf5
    write_features_to_hdf5(tr_speech_dir, hdf5_path, 'train', 'speech', sample_rate, fft_size, hop_size, window_type)
    write_features_to_hdf5(tr_noise_dir, hdf5_path, 'train', 'noise', sample_rate, fft_size, hop_size, window_type)
    write_features_to_hdf5(te_speech_dir, hdf5_path, 'test', 'speech', sample_rate, fft_size, hop_size, window_type)
    write_features_to_hdf5(te_noise_dir, hdf5_path, 'test', 'noise', sample_rate, fft_size, hop_size, window_type)
    
    print("Write out to hdf5_path: %s" % hdf5_path)
    
    
def write_features_to_hdf5(audio_dir, hdf5_path, data_type, audio_type, sample_rate, fft_size, hop_size, window_type):
    
    n_freq = fft_size // 2 + 1
    
    if window_type == 'hamming':
        window = np.hamming(fft_size)
        
    print("--- %s, %s ---" % (data_type, audio_type))
    
    # Create group
    with h5py.File(hdf5_path, 'a') as hf:
        
        if data_type not in hf.keys():
            hf.create_group(data_type)
            
        if audio_type not in hf[data_type].keys():
            hf[data_type].create_group(audio_type)
        
        hf[data_type][audio_type].create_dataset(
            name='data', 
            shape=(0, n_freq), 
            maxshape=(None, n_freq), 
            dtype=np.complex64)
            
        hf[data_type][audio_type].create_dataset(
            name='raw', 
            shape=(0, fft_size), 
            maxshape=(None, fft_size), 
            dtype=np.float32)
        
        hf_data = hf[data_type][audio_type]['data']
        hf_raw = hf[data_type][audio_type]['raw']
        
        name_list = []
        bgn_fin_indices  = []
        energy_list = []
        
        # Spectrogram of a song
        names = os.listdir(audio_dir)
        
        for na in names:
            
            # Extract spectrogram & raw audio frames
            audio_path = os.path.join(audio_dir, na)
            (audio, _) = read_audio(audio_path, sample_rate)
            
            audio = normalize(audio)
            sp = calc_sp(audio, fft_size, hop_size, window)
            frames = stft.enframe(audio, fft_size, hop_size)
            energy = calculate_energy(audio)
            
            print(audio_path, sp.shape, "eng:", energy)
            
            # Write spectrogram & raw audio frames out to hdf5
            bgn_indice = hf_data.shape[0]
            fin_indice = bgn_indice + sp.shape[0]
            
            hf_data.resize((fin_indice, n_freq))
            hf_raw.resize((fin_indice, fft_size))
            
            hf_data[bgn_indice : fin_indice] = sp
            hf_raw[bgn_indice : fin_indice] = frames

            name_list.append(na)
            energy_list.append(energy)
            bgn_fin_indices.append((bgn_indice, fin_indice))
            
        # Write out bin_fin_indices to hdf5
        hf[data_type][audio_type].create_dataset(
            name='bgn_fin_indices', 
            data=bgn_fin_indices, 
            dtype=np.int32)
            
        # Write out energy_list
        hf[data_type][audio_type].create_dataset(
            name='energies', 
            data=energy_list, 
            dtype=np.float32)
            
        # Write out name_list to hdf5
        hf[data_type][audio_type].create_dataset(
            name='names', 
            data=name_list, 
            dtype='S64')


def calc_sp(audio, fft_size, hop_size, window):
    
    sp = stft.stft(x=audio, 
                    window_size=fft_size, 
                    hop_size=hop_size, 
                    window=window, 
                    mode='complex')

    sp = sp.astype(np.complex64)

    return sp


class DataLoader(object):
    def __init__(self, hdf5_file, data_type, audio_type, stack_num, hop_frames, center_only, batch_size, shuffle=True, snr_generator=None, load_raw=False):
        
        self.audio_type = audio_type
        self.stack_num = stack_num
        self.hop_frames = hop_frames
        self.center_only = center_only
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.snr_generator = snr_generator
        
        if load_raw:
            group = 'raw'
        else:
            group = 'data'
        
        with h5py.File(hdf5_file, 'r') as hf:
            self.speech_x = hf[data_type]['speech'][group][:]
            self.noise_x = hf[data_type]['noise'][group][:]
            self.speech_bgn_fin_indices = hf[data_type]['speech']['bgn_fin_indices'][:]
            self.noise_bgn_fin_indices = hf[data_type]['noise']['bgn_fin_indices'][:]
            self.speech_engs = hf[data_type]['speech']['energies'][:]
            self.noise_engs = hf[data_type]['noise']['energies'][:]
            
        self.speech_indices = self.get_indices(self.speech_bgn_fin_indices)
        self.noise_indices = self.get_indices(self.noise_bgn_fin_indices)
        print("Num of speech indices:", len(self.speech_indices))
        print("Num of noise indices:", len(self.noise_indices))
        
        self.speech_indice_to_audio_id_hash = self.calculate_indice_to_audio_hash(self.speech_bgn_fin_indices)
        self.noise_indice_to_audio_id_hash = self.calculate_indice_to_audio_hash(self.noise_bgn_fin_indices)
        
        
    def get_indices(self, bgn_fin_indices):
        stack_num = self.stack_num
        hop_frames = self.hop_frames
        
        indices = np.array([], dtype=np.int32)
        for n in range(bgn_fin_indices.shape[0]):
            [bgn, fin] = bgn_fin_indices[n]
            indices = np.append(indices, range(bgn, fin - stack_num + 1, hop_frames))
            
        return indices
            
    def get_stacked_indices(self, indices):
        full_indices = indices[:, None] + np.arange(self.stack_num)
        return full_indices
            
    def calculate_indice_to_audio_hash(self, bgn_fin_indices):
        
        indice_to_audio_id_hash = np.array([], np.int64)
        
        for n in range(len(bgn_fin_indices)):
            [bgn, fin] = bgn_fin_indices[n]
            indice_to_audio_id_hash = np.append(indice_to_audio_id_hash, np.ones(fin - bgn, dtype=np.int64) * n)
        
        return indice_to_audio_id_hash
            
    def generate(self):
        batch_size = self.batch_size
        shuffle = self.shuffle
        snr_generator = self.snr_generator
        
        seed = 1234
        rs = np.random.RandomState(seed)
        
        speech_indices_queue = np.array([], dtype=np.int64)
        noise_indices_queue = np.array([], dtype=np.int64)
        
        while(True):
            while len(speech_indices_queue) < batch_size:
                if shuffle:
                    rs.shuffle(self.speech_indices)
                speech_indices_queue = np.append(speech_indices_queue, self.speech_indices)

            while len(noise_indices_queue) < batch_size:
                if shuffle:
                    rs.shuffle(self.noise_indices)
                noise_indices_queue = np.append(noise_indices_queue, self.noise_indices)

            # Speech indices
            batch_speech_indices = speech_indices_queue[0 : batch_size]
            speech_indices_queue = np.delete(speech_indices_queue, np.arange(batch_size), axis=0)
            
            stacked_batch_speech_indices = self.get_stacked_indices(batch_speech_indices)
            batch_speech_x = self.speech_x[stacked_batch_speech_indices]
            
            batch_speech_audio_id = self.speech_indice_to_audio_id_hash[batch_speech_indices]
            batch_speech_eng = self.speech_engs[batch_speech_audio_id]
            
            # Noise indices
            batch_noise_indices = noise_indices_queue[0 : batch_size]
            noise_indices_queue = np.delete(noise_indices_queue, np.arange(batch_size), axis=0)
            
            stacked_batch_noise_indices = self.get_stacked_indices(batch_noise_indices)
            batch_noise_x = self.noise_x[stacked_batch_noise_indices]
        
            batch_noise_audio_id = self.noise_indice_to_audio_id_hash[batch_noise_indices]
            batch_noise_eng = self.noise_engs[batch_noise_audio_id]
        
            # Mix
            if snr_generator is None:
                snr = 0
            else:
                snr = snr_generator.__next__()
                
            (batch_mix_x, scaled_batch_noise) = self.mix_speech_noise(batch_speech_x, batch_noise_x, batch_speech_eng, batch_noise_eng, snr)
            
            if self.audio_type == 'speech':
                batch_y = batch_speech_x
            elif self.audio_type == 'noise':
                batch_y = scaled_batch_noise
            
            if self.center_only:
                yield batch_mix_x, batch_y[:, self.stack_num // 2, :]
            else:
                yield batch_mix_x, batch_y 
            
    def mix_speech_noise(self, speech_x, noise_x, speech_eng, noise_eng, snr):

        # Scale coefficient of noise
        r = np.sqrt(speech_eng / noise_eng) * (10. ** (float(-snr) / 20.))
        
        # Obtain mixture
        scaled_noise_x = r[:, None, None] * noise_x
        mix_x = speech_x + scaled_noise_x
        
        # Debug plot
        if False:
            for i1 in range(speech_x.shape[0]):
                fig, axs = plt.subplots(1, 3, sharex=True)
                vmin = -5
                vmax = 5
                axs[0].matshow(np.log(np.abs(speech_x)[i1].T), origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                axs[1].matshow(np.log(np.abs(r[:, None, None] * noise_x)[i1].T), origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                axs[2].matshow(np.log(np.abs(mix_x)[i1].T), origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                plt.show()
                
        return mix_x, scaled_noise_x
        
        
def log(x, type):
    eps = 1e-10
    if type == 'numpy':
        return np.log(x + eps)
    elif type == 'torch':
        return torch.log(x + eps)
    else:
        raise Exception("Incorrect type!")
        
        
def exp(x, type):
    if type == 'numpy':
        return np.exp(x)
    elif type == 'torch':
        return torch.exp(x)
    else:
        raise Exception("Incorrect type!")

    
def calculate_scalar(x):
    if x.ndim == 2:
        mean_ = np.mean(x, axis=0)
        std_ = np.std(x, axis=0)
    elif x.ndim == 3:
        mean_ = np.mean(x, axis=(0, 1))
        std_ = np.std(x, axis=(0, 1))
    return mean_, std_
    
def scale(x, mean, std):
    return (x - mean) / std
    
def inv_scale(x, mean, std):
    return (x * std + mean)
    
def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value. 
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)
    
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
    
class DFT(object):
    def __init__(self, N, cuda, volatile=False):
        self.W = self.dft_matrix(N)
        self.inv_W = self.idft_matrix(N)
        
        self.W_real = np.real(self.W)
        self.W_imag = np.imag(self.W)
        self.inv_W_real = np.real(self.inv_W)
        self.inv_W_imag = np.imag(self.inv_W)

        self.W_real = self.move_data_to_gpu(self.W_real, cuda, volatile)
        self.W_imag = self.move_data_to_gpu(self.W_imag, cuda, volatile)
        self.inv_W_real = self.move_data_to_gpu(self.inv_W_real, cuda, volatile)
        self.inv_W_imag = self.move_data_to_gpu(self.inv_W_imag, cuda, volatile)
        
        self.N = N
        self.cuda = cuda
        self.volatile = volatile
        self.norm = math.sqrt(float(N))
        
    def dft_matrix(self, N):
        (x, y) = np.meshgrid(np.arange(N), np.arange(N))
        omega = np.exp(-2 * np.pi * 1j / N)
        W = np.power(omega, x * y) / np.sqrt(N)
        return W
        
    def idft_matrix(self, N):
        (x, y) = np.meshgrid(np.arange(N), np.arange(N))
        omega = np.exp(2 * np.pi * 1j / N)
        W = np.power(omega, x * y) / np.sqrt(N)
        return W
        
    def move_data_to_gpu(self, x, cuda, volatile):
        x = torch.Tensor(x)
        if cuda:
            x = x.cuda()
        x = Variable(x, volatile=volatile)
        return x
        
    def dft(self, x_real, x_imag):
        """x_real, x_imag: (..., n_fft)
        """
        z_real = torch.matmul(x_real, self.W_real) - torch.matmul(x_imag, self.W_imag)
        z_imag = torch.matmul(x_imag, self.W_real) + torch.matmul(x_real, self.W_imag)
        return z_real, z_imag
        
    def idft(self, x_real, x_imag):
        """x_real, x_imag: (..., n_fft)
        """
        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)
        z_imag = torch.matmul(x_imag, self.inv_W_real) + torch.matmul(x_real, self.inv_W_imag)
        return z_real, z_imag
        
    def rdft(self, x_real):
        """x_real, x_imag: (..., n_fft)
        """
        n_rfft = self.N // 2 + 1
        z_real = torch.matmul(x_real, self.W_real[:, 0 : n_rfft])
        z_imag = torch.matmul(x_real, self.W_imag[:, 0 : n_rfft])
        return z_real, z_imag        
        
    def irdft(self, x_real, x_imag):
        """x_real, x_imag: (..., n_fft // 2 + 1)
        """
        n_rfft = self.N // 2 + 1
        
        flip_x_real = self.flip(x_real)
        x_real = torch.cat((x_real, flip_x_real[..., 1 : n_rfft - 1]), dim=-1)
        
        flip_x_imag = self.flip(x_imag)
        x_imag = torch.cat((x_imag, -1. * flip_x_imag[..., 1 : n_rfft - 1]), dim=-1)
        
        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)
        return z_real
    
    def flip(self, x):
        """Flip a tensor along the last dimension. 
        """
        inv_idx = torch.arange(x.size(-1) - 1, -1, -1).long()
        if self.cuda:
            inv_idx = inv_idx.cuda()
        inv_idx = Variable(inv_idx, self.volatile)
        
        flip_x = x.index_select(dim=-1, index=inv_idx)
        return flip_x

    
###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_calculate_features = subparsers.add_parser('calculate_features')
    parser_calculate_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_features.add_argument('--tr_speech_dir', type=str, required=True)
    parser_calculate_features.add_argument('--tr_noise_dir', type=str, required=True)
    parser_calculate_features.add_argument('--te_speech_dir', type=str, required=True)
    parser_calculate_features.add_argument('--te_noise_dir', type=str, required=True)

    args = parser.parse_args()
    if args.mode == 'calculate_features':
        calculate_features(args)
    else:
        raise Exception("Error!")