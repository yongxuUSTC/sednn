"""
Summary:  mnist dnn pytorch example. 
          te_err around 2%. 
Author:   Qiuqiang Kong
Usage:    $ CUDA_VISIBLE_DEVICES=1 python test9.py train --init_type=glorot_uniform --optimizer=adam --loss=softmax --lr=1e-3
          $ python mnist_dnn_pt.py train --init_type=glorot_uniform --optimizer=adam --loss=softmax --lr=1e-4 --resume_model_path="models/md_3000iters.tar"
Created:  2017.12.09
Modified: 2017.12.12
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
        
        # self.bn1 = nn.
        
    def forward(self, x):
        drop_p = 0.2
        (_, stack_num, n_freq) = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        
        return x5
 
    
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
         
         
def calculate_scalar(args):
    workspace = args.workspace
    stack_num = args.stack_num
    hop_frames = args.hop_frames
    filename = args.filename
    
    
    hdf5_file = os.path.join(args.workspace, "features", "cmplx_spectrogram.h5")
    data_type = 'train'
    batch_size = 100
    data_loader = pp_data.DataLoader(hdf5_file, data_type, stack_num, hop_frames, center_only=True, batch_size=batch_size)
    
    
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


def forward(model, x, mean, std, cuda, volatile=False):
    x = np.abs(x)
    x = move_data_to_gpu(x, cuda, volatile)
    
    x = transform(x, type='torch')
    x = pp_data.scale(x, mean, std)

    output = model(x)
    
    output = pp_data.inv_scale(output, mean, std)
    output = inv_transform(output, type='torch')

    return output


def mse_loss(output, target):
    loss = torch.mean(torch.abs(output - target) ** 2)
    return loss


def evaluate(model, data_loader, mean_, std_, cuda):
    iter = 0
    
    output_all = []
    target_all = []

    max_iter = 20
    
    for (batch_x, batch_y) in data_loader.generate():
        output = forward(model, batch_x, mean_, std_, cuda, volatile=True)
        output = output.data.cpu()
        output_all.append(output)
        
        batch_y = np.abs(batch_y)
        batch_y = torch.Tensor(batch_y)
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
    stack_num = args.stack_num
    hop_frames = args.hop_frames
    filename = args.filename
    cuda = args.use_cuda and torch.cuda.is_available()
    print("cuda:", cuda)
    
    hdf5_file = os.path.join(args.workspace, "features", "cmplx_spectrogram.h5")
    data_type = 'train'
    
    t1 = time.time()
    data_loader = pp_data.DataLoader(hdf5_file, data_type, stack_num, hop_frames, center_only=True, batch_size=100)
    eval_tr_data_loader = pp_data.DataLoader(hdf5_file, 'train', stack_num, hop_frames, center_only=True, batch_size=100)
    eval_te_data_loader = pp_data.DataLoader(hdf5_file, 'test', stack_num, hop_frames, center_only=True, batch_size=100)
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
        
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    # Train
    iter = 0
    model_dir = os.path.join(workspace, "models", filename)
    pp_data.create_folder(model_dir)
    t_train = time.time()
    
    for (batch_x, batch_y) in data_loader.generate():
        
        output = forward(model, batch_x, mean_, std_, cuda)
        
        
        batch_y = np.abs(batch_y)
        batch_y = move_data_to_gpu(batch_y, cuda)
        
        loss = mse_loss(output, batch_y)
        
        if iter%1000==0:
            fig, axs = plt.subplots(3,1, sharex=True)
            axs[0].matshow(np.log((np.abs(batch_x[:, 0, :]))).T, origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(np.log((np.abs(batch_y.data.cpu().numpy()))).T, origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(np.log((np.abs(output.data.cpu().numpy()))).T, origin='lower', aspect='auto', cmap='jet')
            plt.show()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iter += 1
        
        # Evaluate. 
        loss_ary = []
        if iter % 100 == 0:
            t_eval = time.time()
            tr_loss = evaluate(model, eval_tr_data_loader, mean_, std_, cuda)
            te_loss = evaluate(model, eval_te_data_loader, mean_, std_, cuda)
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

        t1 = time.time()
    
    
def inference(args):
    workspace = args.workspace
    model_name = args.model_name
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
    names = os.listdir(audio_dir)
    
    # Load model
    model_path = os.path.join(workspace, "models", filename, model_name)
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
        
    for (cnt, name) in enumerate(names):
        if cnt % n_every == 0:
            audio_path = os.path.join(audio_dir, name)
            (audio, _) = pp_data.read_audio(audio_path, sample_rate)
            
            audio = pp_data.normalize(audio)
            sp = pp_data.calc_sp(audio, fft_size, hop_size, window)
            x = np.abs(sp)
            
            # Process data. 
            n_pad = (stack_num - 1) / 2
            x = pp_data.pad_with_border(x, n_pad)
            x = pp_data.mat_2d_to_3d(x, stack_num, hop=1)
            
            output = forward(model, x, mean_, std_, cuda)
            output = output.data.cpu().numpy()
            
            print(output.shape)
            if visualize:
                fig, axs = plt.subplots(2,1, sharex=True)
                axs[0].matshow(np.log(np.abs(sp)).T, origin='lower', aspect='auto', cmap='jet')
                axs[1].matshow(np.log(np.abs(output)).T, origin='lower', aspect='auto', cmap='jet')
                plt.show()
            
            import crash
            pause
            
    
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
    parser_train.add_argument('--stack_num', type=int, required=True)
    parser_train.add_argument('--hop_frames', type=int, required=True)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--use_cuda', action='store_true', default=True)
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--model_name', type=str, required=True)
    parser_inference.add_argument('--stack_num', type=int, required=True)
    parser_inference.add_argument('--mini_num', type=int, default=-1)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = pp_data.get_filename(__file__)
    
    if args.mode == "calculate_scalar":
        calculate_scalar(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)
    else:
        raise Exception("Error!")