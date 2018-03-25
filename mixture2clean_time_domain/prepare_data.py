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
import logging

import config as cfg
import stft


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def create_logging(log_dir, filemode):
    # Write out to file
    i1 = 0
    while os.path.isfile(os.path.join(log_dir, "%05d.log" % i1)):
        i1 += 1
    log_path = os.path.join(log_dir, "%05d.log" % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging
        
        
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

    # Create hdf5
    hdf5_path = os.path.join(workspace, "features", "data.h5")
    create_folder(os.path.dirname(hdf5_path))
    
    with h5py.File(hdf5_path, 'w') as hf:
        
        hf.attrs['sample_rate'] = sample_rate
      
    # Write out features to hdf5
    write_features_to_hdf5(tr_speech_dir, hdf5_path, 'train', 'speech', sample_rate)
    write_features_to_hdf5(tr_noise_dir, hdf5_path, 'train', 'noise', sample_rate)
    write_features_to_hdf5(te_speech_dir, hdf5_path, 'test', 'speech', sample_rate)
    write_features_to_hdf5(te_noise_dir, hdf5_path, 'test', 'noise', sample_rate)
    
    print("Write out to hdf5_path: %s" % hdf5_path)
    
    
def write_features_to_hdf5(audio_dir, hdf5_path, data_type, audio_type, sample_rate):
    
    print("--- %s, %s ---" % (data_type, audio_type))
    
    # Create group
    with h5py.File(hdf5_path, 'a') as hf:
        
        if data_type not in hf.keys():
            hf.create_group(data_type)
            
        if audio_type not in hf[data_type].keys():
            hf[data_type].create_group(audio_type)
        
        hf[data_type][audio_type].create_dataset(
            name='data', 
            shape=(0,), 
            maxshape=(None,), 
            dtype=np.float32)
            
        hf_data = hf[data_type][audio_type]['data']
        
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
            energy = calculate_energy(audio)
            
            print(audio_path, "eng:", energy)
            
            # Write spectrogram & raw audio frames out to hdf5
            bgn_indice = hf_data.shape[0]
            fin_indice = bgn_indice + audio.shape[0]
            
            hf_data.resize((fin_indice,))
            
            hf_data[bgn_indice : fin_indice] = audio

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


class Tree(object):
    def __init__(self, array, ids):
        if len(array) > 1:
            center = len(array) // 2
            self.node = array[center]
            self.left = Tree(array[0 : center], ids[0 : center])
            self.right = Tree(array[center : ], ids[center :])
        else:
            self.id = ids[0]
            self.left = None
            self.right = None
            
    def is_leaf(self):
        if (self.left is None) and (self.right is None):
            return True
        else:
            return False
         
    def search(self, value):
        return self._search(self, value)
            
    def _search(self, obj, value):
        if obj.is_leaf():
            return obj.id
        else:
            if value < obj.node:
                return obj._search(obj.left, value)
            else:
                return obj._search(obj.right, value)
        


class DataLoader(object):
    def __init__(self, hdf5_file, data_type, audio_type, filter_len, out_len, hop_samples, batch_size, shuffle=True, snr_generator=None):

        self.audio_type = audio_type
        self.filter_len = filter_len
        self.out_len = out_len
        self.hop_samples = hop_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.snr_generator = snr_generator
        self.seed = 1234

        with h5py.File(hdf5_file, 'r') as hf:
            self.speech_x = hf[data_type]['speech']['data'][:]
            self.noise_x = hf[data_type]['noise']['data'][:]
            self.speech_bgn_fin_indices = hf[data_type]['speech']['bgn_fin_indices'][:]
            self.noise_bgn_fin_indices = hf[data_type]['noise']['bgn_fin_indices'][:]
            self.speech_engs = hf[data_type]['speech']['energies'][:]
            self.noise_engs = hf[data_type]['noise']['energies'][:]
            
        self.speech_indices = self.get_indices(self.speech_bgn_fin_indices)
        self.noise_indices = self.get_indices(self.noise_bgn_fin_indices)
        print("Num of speech indices:", len(self.speech_indices))
        print("Num of noise indices:", len(self.noise_indices))
        
        self.speech_indices_tree = Tree(self.speech_bgn_fin_indices[:, 0], np.arange(len(self.speech_bgn_fin_indices)))
        self.noise_indices_tree = Tree(self.noise_bgn_fin_indices[:, 0], np.arange(len(self.noise_bgn_fin_indices)))

    def get_indices(self, bgn_fin_indices):
        
        filter_len = self.filter_len
        out_len = self.out_len
        hop_samples = self.hop_samples
        
        indices = np.array([], dtype=np.int32)
        for n in range(bgn_fin_indices.shape[0]):
            [bgn, fin] = bgn_fin_indices[n]
            indices = np.append(indices, range(bgn, fin - filter_len - out_len + 1, hop_samples))
            
        return indices
            
    def get_stacked_indices(self, indices):
        full_indices = indices[:, None] + np.arange(self.filter_len + self.out_len - 1)
        return full_indices
            
    def generate(self):
        batch_size = self.batch_size
        shuffle = self.shuffle
        snr_generator = self.snr_generator
        
        
        rs = np.random.RandomState(self.seed)
        
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
            
            batch_speech_audio_id = np.array([self.speech_indices_tree.search(indice) for indice in batch_speech_indices])
            batch_speech_eng = self.speech_engs[batch_speech_audio_id]
            
            # Noise indices
            batch_noise_indices = noise_indices_queue[0 : batch_size]
            noise_indices_queue = np.delete(noise_indices_queue, np.arange(batch_size), axis=0)
            
            stacked_batch_noise_indices = self.get_stacked_indices(batch_noise_indices)
            batch_noise_x = self.noise_x[stacked_batch_noise_indices]
        
            batch_noise_audio_id = np.array([self.noise_indices_tree.search(indice) for indice in batch_noise_indices])
            batch_noise_eng = self.noise_engs[batch_noise_audio_id]
        
            # Mix
            if snr_generator is None:
                snr = 0
            else:
                snr = snr_generator.__next__()
                
            (batch_mix_x, scaled_batch_noise) = self.mix_speech_noise(batch_speech_x, batch_noise_x, batch_speech_eng, batch_noise_eng, snr)
            (batch_mix_x, batch_speech_x, scaled_batch_noise) = self.scale(batch_mix_x, batch_speech_x, scaled_batch_noise)
            
            # Target
            center_bgn = self.filter_len // 2
            center_fin = center_bgn + self.out_len
            
            if self.audio_type == 'speech':
                batch_y = batch_speech_x[:, center_bgn : center_fin]
            elif self.audio_type == 'noise':
                batch_y = scaled_batch_noise[:, center_bgn : center_fin]

            yield batch_mix_x, batch_y 
#             
    def mix_speech_noise(self, speech_x, noise_x, speech_eng, noise_eng, snr):

        # Scale coefficient of noise
        r = np.sqrt(speech_eng / noise_eng) * (10. ** (float(-snr) / 20.))
        
        # Obtain mixture
        scaled_noise_x = r[:, None] * noise_x
        mix_x = speech_x + scaled_noise_x
        
        # Debug plot
        if False:
            for i1 in range(speech_x.shape[0]):
                fig, axs = plt.subplots(1, 3, sharex=True)
                vmin = -5
                vmax = 5
                axs[0].plot(speech_x[i1])
                axs[1].plot(scaled_noise_x[i1])
                axs[2].plot(mix_x[i1])
                plt.show()
                
        return mix_x, scaled_noise_x
        
    def scale(self, mix_x, speech_x, noise_x):
        mix_max = np.max(np.abs(mix_x))
        speech_max = np.max(np.abs(speech_x))
        noise_max = np.max(np.abs(noise_x))
        max_ = max(mix_max, speech_max, noise_max)
        return mix_x / max_, speech_x / max_, noise_x / max_


def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value. 
    """
    x_pad_list = [np.zeros(n_pad), x, np.zeros(n_pad)]
    return np.concatenate(x_pad_list, axis=0)

    
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