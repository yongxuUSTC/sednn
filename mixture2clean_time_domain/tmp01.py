"""
2018.03.08
"""
import numpy as np
import time
import os
import argparse
from sklearn import preprocessing
import matplotlib.pyplot as plt
import cPickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

import prepare_data as pp_data
import stft
import config as cfg
import mu_law


class MNIST(data.Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.LongTensor(y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return len(self.x)


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
        
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, filter_len) = layer.weight.size()
        n = n_in * filter_len
        
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()
    
    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)
    layer.bias.data.copy_(torch.zeros(n_out))  
        
def init_bn(bn):
    bn.weight.data.copy_(torch.ones(bn.weight.size()))
    

class CNN(nn.Module):
    def __init__(self, filter_len):
        super(CNN, self).__init__()
        
        n_hid = 1024
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=filter_len, stride=1, padding=0, dilation=1, groups=1, bias=True)

        init_layer(self.conv1)
        
        
    def forward(self, x):
        
        (batch_size, input_len) = x.size()
        x = x.view(batch_size, 1, input_len)
        x = self.conv1(x)
        
        x = x.view(batch_size, x.size(-1))
        
        return x
 
    
def eval(model, data_loader, cuda):
    model.eval()
    pred_all = []
    y_all = []
    for (batch_x, batch_y) in data_loader:
        batch_x = Variable(batch_x, volatile=True)
        if cuda:
            batch_x = batch_x.cuda(async=True)
        pred = model(batch_x)
        pred = pred.data.cpu().numpy()
        pred_all.append(pred)
        y_all.append(batch_y)
        
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    err = pp_data.categorical_error(pred_all, y_all)
    return err
         
         

def transform(x, type):
    return pp_data.log(x, type)
    
    
def inv_transform(x, type):
    return pp_data.exp(x, type)
    

def move_data_to_gpu(x, cuda, volatile=False):
    x = torch.Tensor(x)
    if cuda:
        x = x.cuda()
    x = Variable(x, volatile=volatile)
    return x


def forward(model, x, cuda, volatile=False):
    
    x = move_data_to_gpu(x, cuda, volatile)
    
    output = model(x)
    
    return output


def mse_loss(output, target):
    loss = torch.mean(torch.abs(output - target) ** 2)
    return loss


def l1_loss(output, target):
    loss = torch.mean(torch.abs(output - target))
    return loss


def evaluate(model, data_loader, cuda):
    iter = 0
    
    output_all = []
    target_all = []

    max_iter = 20
    
    for (batch_x, batch_y) in data_loader.generate():
        output = forward(model, batch_x, cuda, volatile=True)
        output = output.data.cpu()
        output_all.append(output)
        
        batch_y = torch.Tensor(batch_y)
        target_all.append(batch_y)
        
        iter += 1
        
        if iter == max_iter:
            break

    output_all = torch.cat(output_all, dim=0)
    target_all = torch.cat(target_all, dim=0)
    loss = l1_loss(output_all, target_all)
    
    return loss
    
def train(args):
    workspace = args.workspace
    filename = args.filename
    cuda = args.use_cuda and torch.cuda.is_available()
    is_mulaw = args.mulaw
    print("cuda:", cuda)
    
    hdf5_file = os.path.join(args.workspace, "features", "data.h5")
    data_type = 'train'
    target_type = 'speech'
    filter_len=513
    out_len = 300
    hop_samples=120
    
    t1 = time.time()
    data_loader = pp_data.DataLoader(hdf5_file, data_type, target_type, filter_len, out_len, hop_samples, batch_size=100, shuffle=True)
    eval_tr_data_loader = pp_data.DataLoader(hdf5_file, 'train', target_type, filter_len, out_len, hop_samples, batch_size=100, shuffle=True)
    eval_te_data_loader = pp_data.DataLoader(hdf5_file, 'test', target_type, filter_len, out_len, hop_samples, batch_size=100, shuffle=True)
    print("Load time: %s" % (time.time() - t1))

    # Model
    model = CNN(filter_len)
    
    if cuda:
        model.cuda()
        
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    # Train
    iter = 0
    model_dir = os.path.join(workspace, "models", filename)
    pp_data.create_folder(model_dir)
    t_train = time.time()
    
    mulaw = mu_law.MuLaw(mu=256)
    
    for (batch_x, batch_y) in data_loader.generate():
        
        if is_mulaw:
            batch_x = mulaw.transform(batch_x)
            batch_y = mulaw.transform(batch_y)
        
        if False:
            tmp = np.zeros_like(batch_x)
            tmp[:, 256 : 256 + batch_y.shape[1]] = batch_y
            
            for i1 in xrange(batch_x.shape[0]):
                plt.plot(batch_x[i1], 'b')
                plt.plot(tmp[i1], 'r')
                plt.axis([0, 800, -1, 1])
                plt.show()
        
        output = forward(model, batch_x, cuda)
        
        batch_y = move_data_to_gpu(batch_y, cuda)
        
        loss = l1_loss(output, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iter += 1
        
        # Evaluate. 
        loss_ary = []
        if iter % 100 == 0:
            t_eval = time.time()
            tr_loss = evaluate(model, eval_tr_data_loader, cuda)
            te_loss = evaluate(model, eval_te_data_loader, cuda)
            print("Iter: %d, train err: %f, test err: %f, train time: %s, eval time: %s" % \
                    (iter, tr_loss, te_loss, time.time() - t_train, time.time() - t_eval))
            t_train = time.time()
        
        # Save model. 
        if iter % 1000 == 0:
            save_out_dict = {'iter': iter, 
                            'state_dict': model.state_dict(), 
                            'optimizer': optimizer.state_dict(), 
                            'te_loss': loss, }
            save_out_path = os.path.join(model_dir, "md_%d_iters.tar" % iter)
            torch.save(save_out_dict, save_out_path)
            print("Save model to %s" % save_out_path)

    
    
def inference(args):
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename
    mini_num = args.mini_num
    visualize = args.visualize
    cuda = args.use_cuda and torch.cuda.is_available()
    is_mulaw = args.mulaw
    print("cuda:", cuda)
    
    filter_len = 513
    
    sample_rate = cfg.sample_rate

    # Audio
    audio_dir = "/vol/vssp/msos/qk/workspaces/speech_enhancement/mixed_audios/spectrogram/test/0db"
    names = os.listdir(audio_dir)
    
    # Load model
    model_path = os.path.join(workspace, "models", filename, "md_{}_iters.tar".format(iteration))
    
    model = CNN(filter_len)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    out_wav_dir = os.path.join(workspace, "enh_wavs", filename)
    pp_data.create_folder(out_wav_dir)
    
    if cuda:
        model.cuda()

    if mini_num > 0:
        n_every = len(names) / mini_num
    else:
        n_every = 1
        
    mulaw = mu_law.MuLaw(mu=256)
        
    for (cnt, name) in enumerate(names):
        if cnt % n_every == 0:
            audio_path = os.path.join(audio_dir, name)
            (audio, _) = pp_data.read_audio(audio_path, sample_rate)
            
            audio = pp_data.normalize(audio)
            
            # Mu-law transform
            if is_mulaw:
                audio = mulaw.transform(audio)
            
            # Process data. 
            n_pad = (filter_len - 1) / 2
            x = pp_data.pad_with_border(audio, n_pad)
            x = x[np.newaxis, :]
            
            output = forward(model, x, cuda, volatile=True)
            output = output.data.cpu().numpy()
            seq = output[0]
            
            # Mu-law inverse transform
            if is_mulaw:
                audio = mulaw.inverse_transform(audio)
                seq = mulaw.inverse_transform(seq)
            
            # Write out wav
            out_wav_path = os.path.join(out_wav_dir, name)
            pp_data.write_audio(out_wav_path, seq, sample_rate)
            print("Write out wav to: %s" % out_wav_path)
            
            if visualize:
                mix_sp = stft.stft(audio, 512, 256, np.hamming(512), mode='magnitude')
                enh_sp = stft.stft(seq, 512, 256, np.hamming(512), mode='magnitude')
                
                fig, axs = plt.subplots(2,2, sharex=False)
                axs[0, 0].plot(audio)
                axs[0, 1].plot(seq)
                axs[0, 0].set_ylim([-1, 1])
                axs[0, 1].set_ylim([-1, 1])
                axs[1, 0].matshow(np.log(np.abs(mix_sp)).T, origin='lower', aspect='auto', cmap='jet', vmin=-5, vmax=5)
                axs[1, 1].matshow(np.log(np.abs(enh_sp)).T, origin='lower', aspect='auto', cmap='jet', vmin=-5, vmax=5)
                plt.show()

            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--use_cuda', action='store_true', default=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--mulaw', action='store_true', default=False)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--use_cuda', action='store_true', default=True)
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--mulaw', action='store_true', default=False)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--mini_num', type=int, default=-1)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = pp_data.get_filename(__file__)
    
    print(args)
    
    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)
    else:
        raise Exception("Error!")