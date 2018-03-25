"""
Created: 2018.02.24
"""
import numpy as np
import time
import os
import argparse
from sklearn import preprocessing
import matplotlib.pyplot as plt
import cPickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

import prepare_data as pp_data
import config as cfg
import stft


class MNIST(data.Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.LongTensor(y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return len(self.x)

class DNN(nn.Module):
    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)
        
    def forward(self, x):
        drop_p = 0.2
        (_, stack_num, n_freq) = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        
        return x5
 

def transform(x, type):
    return pp_data.log(x, type)
    
    
def inv_transform(x, type):
    return pp_data.exp(x, type)
         
         
def calculate_scalar(args):
    workspace = args.workspace
    stack_num = args.stack_num
    hop_frames = args.hop_frames
    filename = args.filename
    audio_type = 'speech'
    
    hdf5_file = os.path.join(args.workspace, "features", "cmplx_spectrogram.h5")
    data_type = 'train'
    batch_size = 500
    data_loader = pp_data.DataLoader(hdf5_file, data_type, audio_type, stack_num, hop_frames, center_only=True, batch_size=batch_size)
    
    
    x_all = []
    iter = 0
    max_iter = 100
    
    for (batch_x, batch_y) in data_loader.generate():
        x_all.append(batch_x)
        
        iter += 1
        if iter == max_iter:
            break
            
    x_all = np.concatenate(x_all, axis=0)
    
    x_all = np.abs(x_all)
    x_all = transform(x_all, type='numpy')
    (mean_, std_) = pp_data.calculate_scalar(x_all)
    
    out_path = os.path.join(workspace, "scalars", filename, "scalar.p")
    pp_data.create_folder(os.path.dirname(out_path))
    cPickle.dump((mean_, std_), open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    print("Scalar saved to %s" % out_path)
        

def move_data_to_gpu(x, cuda, volatile=False):
    x = torch.Tensor(x)
    if cuda:
        x = x.cuda()
    x = Variable(x, volatile=volatile)
    return x


def forward(model, x, mean, std, dft, cuda, volatile=False):
    
    x = np.abs(x)
    x = move_data_to_gpu(x, cuda, volatile)
    
    # (x_real, x_imag) = dft.rdft(x)
    # x = torch.sqrt(x_real ** 2 + x_imag ** 2)
    
    import crash
    pause
    
    x = transform(x, type='torch')
    x = pp_data.scale(x, mean, std)
    
    import crash
    pause

    output = model(x)
    
    
    output = pp_data.inv_scale(output, mean, std)
    output = inv_transform(output, type='torch')

    return output


def mse_loss(output, target):
    loss = torch.mean(torch.abs(output - target))
    return loss


def evaluate(model, data_loader, mean_, std_, dft, cuda):
    iter = 0
    
    output_all = []
    target_all = []

    max_iter = 200
    
    for (batch_x, batch_y) in data_loader.generate():
        output = forward(model, batch_x, mean_, std_, dft, cuda, volatile=True)
        # output = pp_data.inv_scale(output, mean_, std_)
        # output = inv_transform(output, type='torch')
        output_all.append(output)
        
        batch_y = np.abs(batch_y)
        batch_y = move_data_to_gpu(batch_y, cuda)
        # batch_y = transform(batch_y, type='torch')
        # batch_y = pp_data.scale(batch_y, mean_, std_)
        
        target_all.append(batch_y)
        
        iter += 1
        
        if iter == max_iter:
            break

    output_all = torch.cat(output_all, dim=0)
    target_all = torch.cat(target_all, dim=0)
    loss = mse_loss(output_all, target_all)
    
    return loss
    
def train(args):
    workspace = args.workspace
    audio_type = args.audio_type
    stack_num = args.stack_num
    hop_frames = args.hop_frames
    filename = args.filename
    cuda = args.use_cuda and torch.cuda.is_available()
    fft_size = cfg.fft_size
    print("cuda:", cuda)
    
    hdf5_file = os.path.join(args.workspace, "features", "cmplx_spectrogram.h5")
    data_type = 'train'
    
    t1 = time.time()
    batch_size = 500
    shuffle = False
    load_raw = False
    data_loader = pp_data.DataLoader(hdf5_file, data_type, audio_type, stack_num, hop_frames, center_only=True, batch_size=batch_size, shuffle=shuffle, load_raw=load_raw)
    eval_tr_data_loader = pp_data.DataLoader(hdf5_file, 'train', audio_type, stack_num, hop_frames, center_only=True, batch_size=batch_size, shuffle=shuffle, load_raw=load_raw)
    eval_te_data_loader = pp_data.DataLoader(hdf5_file, 'test', audio_type, stack_num, hop_frames, center_only=True, batch_size=batch_size, shuffle=shuffle, load_raw=load_raw)
    print("Load time: %s" % (time.time() - t1))
    
    # Load scalar
    scalar_path = os.path.join(workspace, "scalars", filename, "scalar.p")
    (mean_, std_) = cPickle.load(open(scalar_path, 'rb'))
    mean_ = move_data_to_gpu(mean_, cuda)
    std_ = move_data_to_gpu(std_, cuda)
    
    # Model
    n_freq = 257
    model = DNN(stack_num, n_freq)
    
    if cuda:
        model.cuda()
        
    dft = pp_data.DFT(fft_size, cuda)
        
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    # Train
    iter = 0
    model_dir = os.path.join(workspace, "models", filename, audio_type)
    pp_data.create_folder(model_dir)
    t_train = time.time()
    
    for (batch_x, batch_y) in data_loader.generate():
    
        output = forward(model, batch_x, mean_, std_, dft, cuda)
        
        
        batch_y = np.abs(batch_y)
        batch_y = move_data_to_gpu(batch_y, cuda)
        # batch_y = transform(batch_y, type='torch')
        # batch_y = pp_data.scale(batch_y, mean_, std_)
        
        loss = mse_loss(output, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iter += 1
        
        # Evaluate. 
        loss_ary = []
        if iter % 500 == 0:
            t_eval = time.time()
            tr_loss = evaluate(model, eval_tr_data_loader, mean_, std_, dft, cuda)
            # tr_loss = -1
            te_loss = evaluate(model, eval_te_data_loader, mean_, std_, dft, cuda)
            print("Iter: %d, train err: %f, test err: %f, train time: %s, eval time: %s" % \
                    (iter, tr_loss, te_loss, time.time() - t_train, time.time() - t_eval))
            t_train = time.time()
        
        # Save model. 
        if iter % 5000 == 0:
            save_out_dict = {'iter': iter, 
                            'state_dict': model.state_dict(), 
                            'optimizer': optimizer.state_dict(), 
                            'te_loss': loss, }
            save_out_path = os.path.join(model_dir, "md_%d_iters.tar" % iter)
            torch.save(save_out_dict, save_out_path)
            print("Save model to %s" % save_out_path)

        t1 = time.time()
    
    
def inference(args):
    workspace = args.workspace
    iter = args.iteration
    stack_num = args.stack_num
    filename = args.filename
    mini_num = args.mini_num
    visualize = args.visualize
    cuda = args.use_cuda and torch.cuda.is_available()
    print("cuda:", cuda)
    audio_type = 'speech'
    
    sample_rate = cfg.sample_rate
    fft_size = cfg.fft_size
    hop_size = cfg.hop_size
    window_type = cfg.window_type

    if window_type == 'hamming':
        window = np.hamming(fft_size)

    # Audio
    audio_dir = "/vol/vssp/msos/qk/workspaces/speech_enhancement/mixed_audios/spectrogram/test/0db"
    # audio_dir = "/user/HS229/qk00006/my_code2015.5-/python/pub_speech_enhancement/mixture2clean_dnn/workspace/mixed_audios/spectrogram/test/0db"
    names = os.listdir(audio_dir)
    
    # Load model
    model_path = os.path.join(workspace, "models", filename, audio_type, "md_%d_iters.tar" % iter)
    n_freq = 257
    model = DNN(stack_num, n_freq)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()
        
    # Load scalar
    scalar_path = os.path.join(workspace, "scalars", filename, "scalar.p")
    (mean_, std_) = cPickle.load(open(scalar_path, 'rb'))
    mean_ = move_data_to_gpu(mean_, cuda, volatile=True)
    std_ = move_data_to_gpu(std_, cuda, volatile=True)
    
    if mini_num > 0:
        n_every = len(names) / mini_num
    else:
        n_every = 1
        
    out_wav_dir = os.path.join(workspace, "enh_wavs", filename)
    pp_data.create_folder(out_wav_dir)
    
    dft = pp_data.DFT(fft_size)
        
    for (cnt, name) in enumerate(names):
        if cnt % n_every == 0:
            audio_path = os.path.join(audio_dir, name)
            (audio, _) = pp_data.read_audio(audio_path, sample_rate)
            
            audio = pp_data.normalize(audio)
            cmplx_sp = pp_data.calc_sp(audio, fft_size, hop_size, window)
            x = np.abs(cmplx_sp)
            
            # Process data. 
            n_pad = (stack_num - 1) / 2
            x = pp_data.pad_with_border(x, n_pad)
            x = pp_data.mat_2d_to_3d(x, stack_num, hop=1)
            
            output = forward(model, x, mean_, std_, cuda, dft)
            
            # output = pp_data.inv_scale(output, mean_, std_)
            # output = inv_transform(output, type='torch')
            
            pred_mag_sp = output.data.cpu().numpy()
            
            pred_cmplx_sp = stft.real_to_complex(pred_mag_sp, cmplx_sp)
            frames = stft.istft(pred_cmplx_sp)
            
            cola_constant = stft.get_cola_constant(hop_size, window)
            seq = stft.overlap_add(frames, hop_size, cola_constant)
            seq = seq[0 : len(audio)]
            
            
            # Write out wav
            out_wav_path = os.path.join(out_wav_dir, name)
            pp_data.write_audio(out_wav_path, seq, sample_rate)
            print("Write out wav to: %s" % out_wav_path)
            
            if visualize:
                vmin = -5.
                vmax = 5.
                fig, axs = plt.subplots(2,1, sharex=True)
                axs[0].matshow(np.log(np.abs(cmplx_sp)).T, origin='lower', aspect='auto', cmap='jet')
                axs[1].matshow(np.log(np.abs(output)).T, origin='lower', aspect='auto', cmap='jet')
                plt.show()
            
            
def inference_wiener(args):
    workspace = args.workspace
    iter = args.iteration
    stack_num = args.stack_num
    filename = args.filename
    mini_num = args.mini_num
    visualize = args.visualize
    cuda = args.use_cuda and torch.cuda.is_available()
    print("cuda:", cuda)
    
    sample_rate = cfg.sample_rate
    fft_size = cfg.fft_size
    hop_size = cfg.hop_size
    window_type = cfg.window_type

    if window_type == 'hamming':
        window = np.hamming(fft_size)

    # Audio
    audio_dir = "/vol/vssp/msos/qk/workspaces/speech_enhancement/mixed_audios/spectrogram/test/0db"
    # audio_dir = "/user/HS229/qk00006/my_code2015.5-/python/pub_speech_enhancement/mixture2clean_dnn/workspace/mixed_audios/spectrogram/test/0db"
    names = os.listdir(audio_dir)
    
    # Load model. 
    target_type = ['speech', 'noise']
    model_dict = {}
    for e in target_type:
        n_freq = 257
        model = DNN(stack_num, n_freq)
        model_path = os.path.join(workspace, "models", filename, e, "md_%d_iters.tar" % iter)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Move model to GPU. 
        if cuda:
            model.cuda()
        model.eval()
        
        model_dict[e] = model
        
    # Load scalar
    scalar_path = os.path.join(workspace, "scalars", filename, "scalar.p")
    (mean_, std_) = cPickle.load(open(scalar_path, 'rb'))
    mean_ = move_data_to_gpu(mean_, cuda, volatile=True)
    std_ = move_data_to_gpu(std_, cuda, volatile=True)
    
    if mini_num > 0:
        n_every = len(names) / mini_num
    else:
        n_every = 1
        
    out_wav_dir = os.path.join(workspace, "enh_wavs", filename)
    pp_data.create_folder(out_wav_dir)
        
    for (cnt, name) in enumerate(names):
        if cnt % n_every == 0:
            audio_path = os.path.join(audio_dir, name)
            (audio, _) = pp_data.read_audio(audio_path, sample_rate)
            
            audio = pp_data.normalize(audio)
            cmplx_sp = pp_data.calc_sp(audio, fft_size, hop_size, window)
            x = np.abs(cmplx_sp)
            
            # Process data. 
            n_pad = (stack_num - 1) / 2
            x = pp_data.pad_with_border(x, n_pad)
            x = pp_data.mat_2d_to_3d(x, stack_num, hop=1)
            
            # Predict. 
            pred_dict = {}
            for e in target_type:
                pred = forward(model_dict[e], x, mean_, std_, cuda)
                pred = pred.data.cpu().numpy()
                pred_dict[e] = pred
            print(cnt, name)
            
            # Wiener filter. 
            pred_mag_sp = pred_dict['speech'] / (pred_dict['speech'] + pred_dict['noise']) * np.abs(cmplx_sp)
            
            pred_cmplx_sp = stft.real_to_complex(pred_mag_sp, cmplx_sp)
            frames = stft.istft(pred_cmplx_sp)
            
            cola_constant = stft.get_cola_constant(hop_size, window)
            seq = stft.overlap_add(frames, hop_size, cola_constant)
            seq = seq[0 : len(audio)]
            
            
            # Write out wav
            out_wav_path = os.path.join(out_wav_dir, name)
            pp_data.write_audio(out_wav_path, seq, sample_rate)
            print("Write out wav to: %s" % out_wav_path)
            
            if visualize:
                vmin = -5.
                vmax = 5.
                fig, axs = plt.subplots(3,1, sharex=True)
                axs[0].matshow(np.log(np.abs(cmplx_sp)).T, origin='lower', aspect='auto', cmap='jet')
                axs[1].matshow(np.log(np.abs(pred_dict['speech'])).T, origin='lower', aspect='auto', cmap='jet')
                axs[2].matshow(np.log(np.abs(pred_dict['noise'])).T, origin='lower', aspect='auto', cmap='jet')
                plt.show()
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_calculate_scalar = subparsers.add_parser('calculate_scalar')
    parser_calculate_scalar.add_argument('--workspace', type=str, required=True)
    parser_calculate_scalar.add_argument('--stack_num', type=int, required=True)
    parser_calculate_scalar.add_argument('--hop_frames', type=int, required=True)

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--use_cuda', action='store_true', default=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--audio_type', type=str, required=True)
    parser_train.add_argument('--stack_num', type=int, required=True)
    parser_train.add_argument('--hop_frames', type=int, required=True)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--use_cuda', action='store_true', default=True)
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--stack_num', type=int, required=True)
    parser_inference.add_argument('--mini_num', type=int, default=-1)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    
    parser_inference_wiener = subparsers.add_parser('inference_wiener')
    parser_inference_wiener.add_argument('--use_cuda', action='store_true', default=True)
    parser_inference_wiener.add_argument('--workspace', type=str, required=True)
    parser_inference_wiener.add_argument('--iteration', type=int, required=True)
    parser_inference_wiener.add_argument('--stack_num', type=int, required=True)
    parser_inference_wiener.add_argument('--mini_num', type=int, default=-1)
    parser_inference_wiener.add_argument('--visualize', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = pp_data.get_filename(__file__)
    
    if args.mode == "calculate_scalar":
        calculate_scalar(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)
    elif args.mode == "inference_wiener":
        inference_wiener(args)
    else:
        raise Exception("Error!")