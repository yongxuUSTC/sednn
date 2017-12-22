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

def load_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)     # (n_segs, n_concat, n_freq)
        y = np.array(y)     # (n_segs, n_freq)
        
    return x, y

def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def eval(model, gen, x, y):
    pred_all, y_all = [], []
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    loss = np_mean_absolute_error(y_all, pred_all)
    return loss
    

def train(args):
    workspace = args.workspace
    tr_snr = 0
    te_snr = 0
    
    # Load data. 
    t1 = time.time()
    tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "data.h5")
    te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "%ddb" % int(te_snr), "data.h5")
    (tr_x, tr_y) = load_data(tr_hdf5_path)
    (te_x, te_y) = load_data(te_hdf5_path)
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
        
    # Debug. 
    if False:
        N = 1000
        plt.matshow(tr_x[0 + N : 1000 + N, 0, :].T, origin='lower', aspect='auto', cmap='jet')
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
                  optimizer=Adam(lr=1e-4))
                  
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
    te_loss = eval(model, eval_te_gen, te_x, te_y)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
    
    model_dir = os.path.join(workspace, "models", "%ddb" % int(tr_snr))
    pp_data.create_folder(model_dir)
    
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        loss = model.train_on_batch(batch_x, batch_y)
        
        iter += 1
        
        # Validate. 
        if iter % 1000 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)
            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
            
        # Save model. 
        if iter % 5000 == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            model.save(model_path)
            print("Saved model to %s" % model_path)
        
        if iter == 100000:
            break
        

def inference(args):
    workspace = args.workspace
    tr_snr = 0
    te_snr = 0
    iter = 100000
    n_concat = args.n_concat
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

    for na in names:
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
        print(na)
        print(pred.shape, speech_x.shape)
        
        # Inverse scale. 
        if scale:
            mixed_x = pp_data.inverse_scale_on_2d(mixed_x, scaler)
            speech_x = pp_data.inverse_scale_on_2d(speech_x, scaler)
            pred = pp_data.inverse_scale_on_2d(pred, scaler)
        
        # Debug. 
        if True:
            fig, axs = plt.subplots(3,1, sharex=True)
            axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
            plt.show()

        # Recover enhanced wav. 
        pred_sp = np.exp(pred)
        s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
        
        # Write out enhanced wav. 
        out_path = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr), "%s.enh.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)
        
def evaluate(args):
    workspace = args.workspace
    speech_dir = args.speech_dir
    te_snr = 0
    
    enh_dir = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr))
    names = os.listdir(enh_dir)
    for na in names:
        enh_path = os.path.join(enh_dir, na)
        
        speech_na = na.split('.')[0]
        speech_path = os.path.join(speech_dir, "%s.WAV" % speech_na)
        
        cmd = ' '.join(["./pesq/pesq", speech_path, enh_path, "+16000"])
        os.system(cmd)
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    
    parser_evaluate = subparsers.add_parser('evaluate')
    parser_evaluate.add_argument('--workspace', type=str, required=True)
    parser_evaluate.add_argument('--speech_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        raise Exception("Error!")