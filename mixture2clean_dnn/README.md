# Speech enhancement using deep neural networks (Keras implementation)

This code uses deep neural network (DNN) to do speech enhancement. This code is a Keras implementation of The paper:

[1] Xu, Y., Du, J., Dai, L.R. and Lee, C.H., 2015. A regression approach to speech enhancement based on deep neural networks. IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP), 23(1), pp.7-19.

Original C++ implementation is here (https://github.com/yongxuUSTC/DNN-for-speech-enhancement) by Yong Xu (yong.xu@surrey.ac.uk). This Keras re-implementation is done by Qiuqiang Kong (q.kong@surrey.ac.uk)

## Run on mini data. 
It is suggest to use mini data for a quick run before using full data. We already prepared mini data in this repo. You may run the code as follows, 

1. pip install -r requirements.txt

2. Download the PESQ evaluation tool (https://www.itu.int/rec/T-REC-P.862-200102-I/en) and compile the code by $ gcc -o pesq pesq_tool/*.c -lm

Copy the compiled executable pesq to mixture2clean_dnn/

3. Run ./runme.sh, then mixing data, training, inference and evaluation will be executed. You may also run the commands in runme.sh line by line to ensure every steps are correctly runned. 

If all the steps are successful, you may get results printed on the screen. Notice only mini data is used for training. Better results can be obtained using more data for training. 

<pre>
Noise(0dB)   PESQ
----------------------
n64     1.36 +- 0.05
n71     1.35 +- 0.18
----------------------
Avg.    1.35 +- 0.12
</pre>

## Run on TIMIT and 115 noises
You may replace the mini data with your own data. We listed the data need to be prepared in meta_data/ to re-run the experiments in [1]. The data contains:

Training:
Speech: TIMIT 4620 training sentences. 
Noise: 115 kinds of noises (http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/115noises.html)

Testing:
Speech: TIMIT 168 testing sentences (selected 10% from 1680 testing sentences)
Noise: Noise 92 (http://www.speech.cs.cmu.edu/comp.speech/Section1/Data/noisex.html)

Some of the dataset are not published. Instead, you could collect your own data. 

1. Download and prepare data. 

2. Set MINIDATA=0 in runme.sh. Modify WORKSPACE, TR_SPEECH_DIR, TR_NOISE_DIR, TE_SPEECH_DIR, TE_NOISE_DIR in runme.sh file. 

3. Run ./runme.sh

If all the steps are successful, you may get results printed on the screen. The training takes a few miniutes to train 10,000 iterations on a TitanX GPU. The training and testing loss looks like:

<pre>
Iteration: 0, tr_loss: 1.228049, te_loss: 1.252313
Iteration: 1000, tr_loss: 0.533825, te_loss: 0.677872
Iteration: 2000, tr_loss: 0.505751, te_loss: 0.678816
Iteration: 3000, tr_loss: 0.483631, te_loss: 0.666576
Iteration: 4000, tr_loss: 0.480287, te_loss: 0.675403
Iteration: 5000, tr_loss: 0.457020, te_loss: 0.676319
Saved model to /vol/vssp/msos/qk/workspaces/speech_enhancement/models/0db/md_5000iters.h5
Iteration: 6000, tr_loss: 0.461330, te_loss: 0.673847
Iteration: 7000, tr_loss: 0.445159, te_loss: 0.668545
Iteration: 8000, tr_loss: 0.447244, te_loss: 0.680740
Iteration: 9000, tr_loss: 0.427652, te_loss: 0.678236
Iteration: 10000, tr_loss: 0.421219, te_loss: 0.663294
Saved model to /vol/vssp/msos/qk/workspaces/speech_enhancement/models/0db/md_10000iters.h5
Training time: 202.551192045 s
</pre>

The final PESQ looks like:

<pre>
Noise(0dB)            PESQ
---------------------------------
pink             2.01 +- 0.23
buccaneer1       1.88 +- 0.25
factory2         2.21 +- 0.21
hfchannel        1.63 +- 0.24
factory1         1.93 +- 0.23
babble           1.81 +- 0.28
m109             2.13 +- 0.25
leopard          2.49 +- 0.23
volvo            2.83 +- 0.23
buccaneer2       2.03 +- 0.25
white            2.00 +- 0.21
f16              1.86 +- 0.24
destroyerops     1.99 +- 0.23
destroyerengine  1.86 +- 0.23
machinegun       2.55 +- 0.27
---------------------------------
Avg.             2.08 +- 0.24
</pre>


## Visualization
In the inference step, you may add --visualize to the arguments to plot the mixture, clean and enhanced speech log magnitude spectrogram. 

![alt text](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_sp.png)

## Bugs report:
1. PESQ dose not support long path/folder name, so please shorten your path/folder name. Or you will get a wrong/low PESQ score (or you can modify the PESQ source code to enlarge the size of the path name variable)

2. For larger dataset which can not be loaded into the momemory at one time, you can 1. prepare your training scp list ---> 2. random your training scp list ---> 3. split your triaining scp list into several parts ---> 4. read each part for training one by one
