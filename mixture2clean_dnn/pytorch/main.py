import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import pickle
import matplotlib.pyplot as plt
try:
    import cPickle
except:
    import _pickle as cPickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utilities import (create_folder, load_hdf5, scale, np_mean_absolute_error, 
    pad_with_border, log_sp, mat_2d_to_3d, inverse_scale, get_stft_window_func, 
    write_audio, read_audio, calculate_spectrogram)
from models import DNN, move_data_to_gpu
from data_generator import DataGenerator
from stft import real_to_complex, istft, get_cola_constant, overlap_add

import config


def mae_loss(output, target):
    """Mean absolute error loss. 
    """
    return torch.mean(torch.abs(output - target))


def evaluate(model, generator, max_iteration, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      max_iteration: int, maximum iteration for validation
      cuda: bool.
      
    Returns:
      accuracy: float
    """
    
    # Generate function
    generate_func = generator.generate_validate(
        shuffle=True, max_iteration=max_iteration)
            
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=True)

    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)
    
    loss = np_mean_absolute_error(outputs, targets)
    
    return loss


def forward(model, generate_func, cuda, return_target):
    """Forward data to a model.
    
    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs = []
    audio_names = []
    
    if return_target:
        targets = []
    
    # Evaluate on mini-batch
    for data in generate_func:
            
        if return_target:
            (batch_x, batch_y) = data
            
        else:
            batch_x = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_x)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        
        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets
        
    return dict


def train(args):

    # Arugments & parameters
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    batch_size = 500
    lr = 1e-4
    cuda = torch.cuda.is_available()

    # Paths
    train_hdf5_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'train', '{}db'.format(int(tr_snr)), 'data.h5')
        
    test_hdf5_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'test', '%ddb' % int(te_snr), 'data.h5')

    scaler_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'train', '{}db'.format(int(tr_snr)), 'scaler.p')
        
    models_dir = os.path.join(workspace, 'models', '{}db'.format(int(tr_snr)))
    create_folder(models_dir)

    # Load data
    load_data_time = time.time()
    
    (train_x, train_y) = load_hdf5(train_hdf5_path)
    (test_x, test_y) = load_hdf5(test_hdf5_path)
    (_, n_concat, freq_bins) = train_x.shape
    
    print('train_x.shape: {}, train_y.shape: {}'.format(train_x.shape, train_y.shape))
    print('test_x.shape: {}, test_y.shape: {}'.format(test_x.shape, test_y.shape))
    print('Load data time: {} s'.format(time.time() - load_data_time,))
    print('{} iterations / epoch'.format(int(train_x.shape[0] / batch_size)))
    
    # Debug plot
    if False:
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].matshow(train_x[0 : 1000, n_concat // 2, :].T, origin='lower', aspect='auto', cmap='jet')
        axs[1].matshow(train_y[0 : 1000, :].T, origin='lower', aspect='auto', cmap='jet')
        axs[0].set_title('Mixture')
        axs[1].set_title('Clean')
        plt.tight_layout()
        plt.show()
        pause

    # Scale data
    scale_time = time.time()
    scaler = pickle.load(open(scaler_path, 'rb'))
    train_x = scale(train_x, scaler)
    train_y = scale(train_y, scaler)
    test_x = scale(test_x, scaler)
    test_y = scale(test_y, scaler)
    print('Scale data time: {} s'.format(time.time() - scale_time,))

    # Model
    model = DNN(n_concat, freq_bins)
    
    if cuda:
        model.cuda()

    # Data generator
    train_generator = DataGenerator(train_x, train_y, batch_size)
    test_generator = DataGenerator(test_x, test_y, batch_size)
    
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0., amsgrad=True)

    # Train on mini batches
    train_bgn_time = time.time()
    
    for iteration, (batch_x, batch_y) in enumerate(train_generator.generate_train()):
        
        # Evaluate
        if iteration % 1000 == 0:

            train_fin_time = time.time()

            train_loss = evaluate(
                model=model,
                generator=train_generator,
                max_iteration=None,
                cuda=cuda)
                
            test_loss = evaluate(
                model=model,
                generator=test_generator,
                max_iteration=None,
                cuda=cuda)

            print('Iteration: {}, tr_loss: {:.3f}, te_loss: {:.3f}, '
                'tr_time: {:.3f}, te_time: {:.3f}'.format(
                iteration, train_loss, test_loss, 
                train_fin_time - train_bgn_time, time.time() - train_fin_time))
                
            train_bgn_time = time.time()

        # Save model
        if iteration % 5000 == 0 and iteration > 0:

            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
                             
            model_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
                
            torch.save(save_out_dict, model_path)
            print('Model saved to {}'.format(model_path))

        # Train
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        model.train()
        batch_output = model(batch_x)
        loss = mae_loss(batch_output, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 10000:
            break


def inference(args):
    """Inference all test data, write out recovered wavs to disk. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      n_concat: int, number of frames to concatenta, should equal to n_concat 
          in the training stage. 
      iter: int, iteration of model to load. 
      visualize: bool, plot enhanced spectrogram for debug. 
    """
    
    # Arguments & parameters
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    n_concat = args.n_concat
    iteration = args.iteration
    visualize = args.visualize
    data_type = 'test'
    
    window_size = config.window_size
    overlap = config.overlap
    hop_size = window_size - overlap
    cuda = torch.cuda.is_available()
    freq_bins = window_size // 2 + 1
    sample_rate = config.sample_rate
    window = get_stft_window_func(config.window_type)(window_size)
    
    # Paths
    mixed_audios_dir = os.path.join(workspace, 'mixed_audios', data_type, 
        '{}db'.format(int(te_snr)))
        
    features_dir = os.path.join(workspace, 'features', 'spectrogram', data_type, 
        '{}db'.format(int(te_snr)))
    
    scaler_path = os.path.join(workspace, 'packed_features', 'spectrogram', 
        'train', '{}db'.format(int(tr_snr)), 'scaler.p')
    
    model_path = os.path.join(workspace, 'models', '{}db'.format(int(tr_snr)), 
        'md_{}_iters.tar'.format(iteration))
        
    enh_audios_dir = os.path.join(workspace, 'enh_wavs', data_type, 
        '{}db'.format(int(te_snr)))

    # Load model
    model = DNN(n_concat, freq_bins)    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()
    
    # Load scaler
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    feature_names = os.listdir(features_dir)

    for (cnt, audio_name) in enumerate(feature_names):
        
        print(cnt, audio_name)

        # Load feature
        feature_path = os.path.join(features_dir, audio_name)
        data = cPickle.load(open(feature_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        
        '''Or calculate feature from audio by:
        mixed_cmplx_x = calculate_spectrogram(mixed_audio, mode='complex')
        '''
        
        mixed_x = np.abs(mixed_cmplx_x)
        
        # Process data
        n_pad = (n_concat - 1) // 2
        mixed_x = pad_with_border(mixed_x, n_pad)
        mixed_x = log_sp(mixed_x)
        speech_x = log_sp(speech_x)
        
        # Scale data
        mixed_x = scale(mixed_x, scaler)
        
        # Cut input spectrogram to 3D segments with n_concat
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        
        # Move data to GPU
        mixed_x_3d = move_data_to_gpu(mixed_x_3d, cuda)
        
        # Predict
        prediction = model(mixed_x_3d)
        prediction = prediction.data.cpu().numpy()
        
        # Inverse scale
        mixed_x = inverse_scale(mixed_x, scaler)
        prediction = inverse_scale(prediction, scaler)
        
        # Debug plot
        if args.visualize:
            
            fig, axs = plt.subplots(3,1, sharex=False)
            axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(prediction.T, origin='lower', aspect='auto', cmap='jet')
            axs[0].set_title('{}db mixture log spectrogram'.format(int(te_snr)))
            axs[1].set_title('Clean speech log spectrogram')
            axs[2].set_title('Enhanced speech log spectrogram')
            for j1 in range(3):
                axs[j1].xaxis.tick_bottom()
            plt.tight_layout()
            plt.show()

        # Recover enhanced wav
        prediction_sp = np.exp(prediction)
        complex_sp = real_to_complex(prediction_sp, mixed_cmplx_x)
        frames = istft(complex_sp)
        
        # Overlap add
        cola_constant = get_cola_constant(hop_size, window)
        enh_audio = overlap_add(frames, hop_size, cola_constant)
        
        # Write out enhanced wav
        bare_name = os.path.splitext(audio_name)[0]
        out_path = os.path.join(enh_audios_dir, '{}.enh.wav'.format(bare_name))
        create_folder(os.path.dirname(out_path))
        write_audio(out_path, enh_audio, sample_rate)
        print('Write enhanced audio to {}'.format(out_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--tr_snr', type=float, required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)
    
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--visualize', action='store_true', default=False)

    args = parser.parse_args()
    print(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference':
        inference(args)

    else:
        raise Exception('Error argument!')