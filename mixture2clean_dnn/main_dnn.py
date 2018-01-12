"""
Summary:  Train, inference and evaluate speech enhancement. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import numpy as np
import os
import pickle
import cPickle
import h5py
import argparse
import time
import glob
import matplotlib.pyplot as plt

import prepare_data as pp_data
import config as cfg
from data_generator import DataGenerator
from spectrogram_to_wave import recover_wav

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.models import load_model


def eval(model, gen, x, y):
    """Validation function. 
    
    Args:
      model: keras model. 
      gen: object, data generator. 
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []
    
    # Inference in mini batch. 
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)
        
    # Concatenate mini batch prediction. 
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    # Compute loss. 
    loss = pp_data.np_mean_absolute_error(y_all, pred_all)
    return loss
    

def train(args):
    """Train the neural network. Write out model every several iterations. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      lr: float, learning rate. 
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    lr = args.lr
    
    # Load data. 
    t1 = time.time()
    tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "data.h5")
    te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "%ddb" % int(te_snr), "data.h5")
    (tr_x, tr_y) = pp_data.load_hdf5(tr_hdf5_path)
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    print(tr_x.shape, tr_y.shape)
    print(te_x.shape, te_y.shape)
    print("Load data time: %s s" % (time.time() - t1,))
    
    batch_size = 500
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))
    
    # Scale data. 
    if True:
        t1 = time.time()
        scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "scaler.p")
        scaler = pickle.load(open(scaler_path, 'rb'))
        tr_x = pp_data.scale_on_3d(tr_x, scaler)
        tr_y = pp_data.scale_on_2d(tr_y, scaler)
        te_x = pp_data.scale_on_3d(te_x, scaler)
        te_y = pp_data.scale_on_2d(te_y, scaler)
        print("Scale data time: %s s" % (time.time() - t1,))
        
    # Debug plot. 
    if False:
        plt.matshow(tr_x[0 : 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
        plt.show()
        pause
        
    # Build model
    (_, n_concat, n_freq) = tr_x.shape
    n_hid = 2048
    
    model = Sequential()
    model.add(Flatten(input_shape=(n_concat, n_freq)))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_freq, activation='linear'))
    model.summary()
    
    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(lr=lr))

    # Data generator. 
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    
    # Directories for saving models and training stats
    model_dir = os.path.join(workspace, "models", "%ddb" % int(tr_snr))
    pp_data.create_folder(model_dir)
    
    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    pp_data.create_folder(stats_dir)
    
    # Print loss before training. 
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
    te_loss = eval(model, eval_te_gen, te_x, te_y)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
    
    # Save out training stats. 
    stat_dict = {'iter': iter, 
                    'tr_loss': tr_loss, 
                    'te_loss': te_loss, }
    stat_path = os.path.join(stats_dir, "%diters.p" % iter)
    cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    
    # Train. 
    t1 = time.time()
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        loss = model.train_on_batch(batch_x, batch_y)
        iter += 1
        
        # Validate and save training stats. 
        if iter % 1000 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)
            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
            
            # Save out training stats. 
            stat_dict = {'iter': iter, 
                         'tr_loss': tr_loss, 
                         'te_loss': te_loss, }
            stat_path = os.path.join(stats_dir, "%diters.p" % iter)
            cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
            
        # Save model. 
        if iter % 5000 == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            model.save(model_path)
            print("Saved model to %s" % model_path)
        
        if iter == 10001:
            break
            
    print("Training time: %s s" % (time.time() - t1,))

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
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    n_concat = args.n_concat
    iter = args.iteration
    
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    scale = True
    
    # Load model. 
    model_path = os.path.join(workspace, "models", "%ddb" % int(tr_snr), "md_%diters.h5" % iter)
    model = load_model(model_path)
    
    # Load scaler. 
    scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    # Load test data. 
    feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "%ddb" % int(te_snr))
    names = os.listdir(feat_dir)

    for (cnt, na) in enumerate(names):
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_cmplx_x)
        
        # Process data. 
        n_pad = (n_concat - 1) / 2
        mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
        mixed_x = pp_data.log_sp(mixed_x)
        speech_x = pp_data.log_sp(speech_x)
        
        # Scale data. 
        if scale:
            mixed_x = pp_data.scale_on_2d(mixed_x, scaler)
            speech_x = pp_data.scale_on_2d(speech_x, scaler)
        
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        
        # Predict. 
        pred = model.predict(mixed_x_3d)
        print(cnt, na)
        
        # Inverse scale. 
        if scale:
            mixed_x = pp_data.inverse_scale_on_2d(mixed_x, scaler)
            speech_x = pp_data.inverse_scale_on_2d(speech_x, scaler)
            pred = pp_data.inverse_scale_on_2d(pred, scaler)
        
        # Debug plot. 
        if args.visualize:
            fig, axs = plt.subplots(3,1, sharex=False)
            axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
            axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
            axs[1].set_title("Clean speech log spectrogram")
            axs[2].set_title("Enhanced speech log spectrogram")
            for j1 in xrange(3):
                axs[j1].xaxis.tick_bottom()
            plt.tight_layout()
            plt.show()

        # Recover enhanced wav. 
        pred_sp = np.exp(pred)
        s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
        s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude 
                                                        # change after spectrogram and IFFT. 
        
        # Write out enhanced wav. 
        out_path = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr), "%s.enh.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--tr_snr', type=float, required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)
    parser_train.add_argument('--lr', type=float, required=True)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    
    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'calculate_pesq':
        calculate_pesq(args)
    else:
        raise Exception("Error!")