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
import h5py
from sklearn import preprocessing
import pickle
try:
    import cPickle
except:
    import _pickle as cPickle

from utilities import (create_folder, read_audio, write_audio, 
    calculate_spectrogram, log_sp, mat_2d_to_3d, pad_with_border, 
    calculate_scaler)
import config


###
def create_mixture_csv(args):
    """Create csv containing mixture information. 
    Each line in the .csv file contains [speech_name, noise_name, noise_onset, noise_offset]
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      magnification: int, only used when data_type='train', number of noise 
          selected to mix with a speech. E.g., when magnication=3, then 4620
          speech with create 4620*3 mixtures. magnification should not larger 
          than the species of noises. 
    """
    
    # Arguments & parameters
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    magnification = args.magnification
    
    rs = np.random.RandomState(0)
    
    # Paths
    out_csv_path = os.path.join(workspace, 'mixture_csvs', '{}.csv'.format(data_type))
    create_folder(os.path.dirname(out_csv_path))
    
    speech_names = [name for name in os.listdir(speech_dir) if name.lower().endswith('.wav')]
    noise_names = [name for name in os.listdir(noise_dir) if name.lower().endswith('.wav')]
    
    cnt = 0
    f = open(out_csv_path, 'w')
    f.write('{}\t{}\t{}\t{}\n'.format('speech_name', 'noise_name', 'noise_onset', 'noise_offset'))
    
    for speech_na in speech_names:
        
        # Read speech
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path)
        len_speech = len(speech_audio)
        
        # For training data, mix each speech with randomly picked #magnification noises
        if data_type == 'train':
            selected_noise_names = rs.choice(noise_names, size=magnification, replace=False)
            
        # For test data, mix each speech with all noises
        elif data_type == 'test':
            selected_noise_names = noise_names
        else:
            raise Exception('data_type must be train | test!')

        # Mix one speech with different noises many times
        for noise_na in selected_noise_names:
            noise_path = os.path.join(noise_dir, noise_na)
            (noise_audio, _) = read_audio(noise_path)
            
            len_noise = len(noise_audio)

            # If noise shorter than speech then noise will be repeated in calculating features
            if len_noise <= len_speech:
                noise_onset = 0
                nosie_offset = len_speech
                
            # If noise longer than speech then randomly select a segment of noise
            else:
                noise_onset = rs.randint(0, len_noise - len_speech, size=1)[0]
                nosie_offset = noise_onset + len_speech
            
            if cnt % 100 == 0:
                print(cnt)
                
            cnt += 1
            f.write('{}\t{}\t{}\t{}\n'.format(speech_na, noise_na, noise_onset, nosie_offset))
    
    f.close()
    
    print('Write {} mixture csv to {}!'.format(data_type, out_csv_path))
    
    
###
def calculate_mixture_features(args):
    """Calculate spectrogram for mixed, speech and noise audio. Then write the 
    features to disk. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
    """
    
    # Arguments & parameters
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    snr = args.snr
    sample_rate = config.sample_rate
    
    # Paths
    mixture_csv_path = os.path.join(workspace, 'mixture_csvs', '{}.csv'.format(data_type))
    
    # Open mixture csv. 
    with open(mixture_csv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
    
    t1 = time.time()
    cnt = 0
    
    for i1 in range(1, len(lis)):
        
        [speech_na, noise_na, noise_onset, noise_offset] = lis[i1]
        noise_onset = int(noise_onset)
        noise_offset = int(noise_offset)
        
        # Read speech audio
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, target_fs=sample_rate)
        
        # Read noise audio
        noise_path = os.path.join(noise_dir, noise_na)
        (noise_audio, _) = read_audio(noise_path, target_fs=sample_rate)
        
        # Repeat noise to the same length as speech
        if len(noise_audio) < len(speech_audio):
            n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
            noise_audio_repeat = np.tile(noise_audio, n_repeat)
            noise_audio = noise_audio_repeat[0 : len(speech_audio)]
            
        # Truncate noise to the same length as speech
        else:
            noise_audio = noise_audio[noise_onset : noise_offset]
        
        # Scale speech to given snr
        scaler = get_amplitude_scaling_factor(speech_audio, noise_audio, snr=snr)
        speech_audio *= scaler
        
        # Get normalized mixture, speech, noise
        (mixed_audio, speech_audio, noise_audio, alpha) = additive_mixing(speech_audio, noise_audio)

        # Write out mixed audio
        out_bare_name = os.path.join('{}.{}'.format(
            os.path.splitext(speech_na)[0], os.path.splitext(noise_na)[0]))
            
        out_audio_path = os.path.join(workspace, 'mixed_audios', 
            data_type, '{}db'.format(int(snr)), '{}.wav'.format(out_bare_name))
            
        create_folder(os.path.dirname(out_audio_path))
        write_audio(out_audio_path, mixed_audio, sample_rate)
        print('Write mixture wav to: {}'.format(out_audio_path))

        # Extract spectrogram
        mixed_complx_x = calculate_spectrogram(mixed_audio, mode='complex')
        speech_x = calculate_spectrogram(speech_audio, mode='magnitude')
        noise_x = calculate_spectrogram(noise_audio, mode='magnitude')

        # Write out features
        out_feature_path = os.path.join(workspace, 'features', 'spectrogram', 
            data_type, '{}db'.format(int(snr)), '{}.p'.format(out_bare_name))
            
        create_folder(os.path.dirname(out_feature_path))
        data = [mixed_complx_x, speech_x, noise_x, alpha, out_bare_name]
        cPickle.dump(data, open(out_feature_path, 'wb'))
        
        # Print
        if cnt % 100 == 0:
            print(cnt)
            
        cnt += 1

    print('Extracting feature time: %s' % (time.time() - t1))
    
    
def rms(y):
    """Root mean square. 
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))


def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    """Given s and n, return the scaler s according to the snr. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      snr: float, SNR. 
      method: 'rms'. 
      
    Outputs:
      float, scaler. 
    """
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio =  10. ** (float(snr) / 20.)    # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor


def additive_mixing(s, n):
    """Mix normalized source1 and source2. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      
    Returns:
      mix_audio: ndarray, mixed audio. 
      s: ndarray, pad or truncated and scalered source1. 
      n: ndarray, scaled source2. 
      alpha: float, normalize coefficient. 
    """
    mixed_audio = s + n
        
    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha
    s *= alpha
    n *= alpha
    return mixed_audio, s, n, alpha
    
    
###
def pack_features(args):
    """Load all features, apply log and conver to 3D tensor: 
    (samples, n_concat, freq_bins) and write it out to an hdf5 file. 
    
    Args:
      workspace: str, path of workspace. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
      n_concat: int, number of frames to be concatenated. 
      n_hop: int, hop frames. 
    """
    
    # Arguments & parameters
    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    n_concat = args.n_concat
    n_hop = args.n_hop
    
    # Paths
    feature_dir = os.path.join(workspace, 'features', 'spectrogram', data_type, 
        '{}db'.format(int(snr)))
        
    hdf5_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        data_type, '{}db'.format(int(snr)), 'data.h5')
        
    create_folder(os.path.dirname(hdf5_path))
    
    x_all = []  # (n_segs, n_concat, n_freq)
    y_all = []  # (n_segs, n_freq)
    
    cnt = 0
    t1 = time.time()
    
    # Load all features
    names = os.listdir(feature_dir)
    
    for name in names:
        
        # Load feature. 
        feature_path = os.path.join(feature_dir, name)
        data = cPickle.load(open(feature_path, 'rb'))
        [mixed_complx_x, speech_x, noise_x, alpha, name] = data
        mixed_x = np.abs(mixed_complx_x)

        # Pad start and finish of the spectrogram with boarder values. 
        n_pad = (n_concat - 1) // 2
        mixed_x = pad_with_border(mixed_x, n_pad)
        speech_x = pad_with_border(speech_x, n_pad)
    
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=n_hop)
        x_all.append(mixed_x_3d)
        
        # Cut target spectrogram and take the center frame of each 3D segment. 
        speech_x_3d = mat_2d_to_3d(speech_x, agg_num=n_concat, hop=n_hop)
        y = speech_x_3d[:, (n_concat - 1) // 2, :]
        y_all.append(y)
        
        # Print. 
        if cnt % 100 == 0:
            print(cnt)
            
        # if cnt == 3: break
        cnt += 1
        
    x_all = np.concatenate(x_all, axis=0)   # (n_segs, n_concat, n_freq)
    y_all = np.concatenate(y_all, axis=0)   # (n_segs, n_freq)
    
    x_all = log_sp(x_all).astype(np.float32)
    y_all = log_sp(y_all).astype(np.float32)
    
    # Write out data to hdf5 file. 
    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
    
    print('Write out to {}'.format(hdf5_path))
    print('Pack features finished! {} s'.format(time.time() - t1,))
    

###
def write_out_scaler(args):
    """Compute and write out scaler of data. 
    """
    
    # Arguments & parameters
    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    
    # Paths
    hdf5_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        data_type, '{}db'.format(int(snr)), 'data.h5')
    
    scaler_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        data_type, '{}db'.format(int(snr)), 'scaler.p')
        
    create_folder(os.path.dirname(scaler_path))
    
    # Load data
    t1 = time.time()
    
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf['x'][:]     # (n_segs, n_concat, n_freq)
    
    # Compute scaler
    scaler = calculate_scaler(x, axis=(0, 1))
    
    print(scaler)
    
    # Write out scaler
    pickle.dump(scaler, open(scaler_path, 'wb'))
    
    print('Save scaler to {}'.format(scaler_path))
    print('Compute scaler finished! {} s'.format(time.time() - t1,))
    
    
###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_mixture_csv = subparsers.add_parser('create_mixture_csv')
    parser_create_mixture_csv.add_argument('--workspace', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--noise_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--data_type', type=str, required=True)
    parser_create_mixture_csv.add_argument('--magnification', type=int, default=1)

    parser_calculate_mixture_features = subparsers.add_parser('calculate_mixture_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--noise_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--data_type', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--snr', type=float, required=True)
    
    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)
    
    parser_write_out_scaler = subparsers.add_parser('write_out_scaler')
    parser_write_out_scaler.add_argument('--workspace', type=str, required=True)
    parser_write_out_scaler.add_argument('--data_type', type=str, required=True)
    parser_write_out_scaler.add_argument('--snr', type=float, required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'create_mixture_csv':
        create_mixture_csv(args)
        
    elif args.mode == 'calculate_mixture_features':
        calculate_mixture_features(args)
        
    elif args.mode == 'pack_features':
        pack_features(args)       
        
    elif args.mode == 'write_out_scaler':
        write_out_scaler(args)
        
    else:
        raise Exception('Error!')