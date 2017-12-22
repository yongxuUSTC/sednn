### Speech enhancement using deep neural networks (Keras implementation)

This code uses deep neural network (DNN) to do speech enhancement. This code is a Keras implementation of The paper:

[1] Xu, Y., Du, J., Dai, L.R. and Lee, C.H., 2015. A regression approach to speech enhancement based on deep neural networks. IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP), 23(1), pp.7-19.

Original C++ implementation is here (https://github.com/yongxuUSTC/DNN-for-speech-enhancement). This Keras re-implementation is done by Qiuqiang Kong (q.kong@surrey.ac.uk)

## Usage. 
It is suggest to use mini data for a quick run before using full data. We already prepared mini data in this repo. You may run the code as follows, 

1. pip install -r requirements.txt

2. Compile the PESQ evaluation tool using $ gcc -o pesq pesq_tool/*.c -lm

3. Run ./runme.sh, then mixing data, training, inference and evaluation will be executed. You may also run the commands in runme.sh line by line. 

If all the steps are successful, you may get results printed on the screen. Notice only mini data is used for training. Better results can be obtained using more data for training. 

Noise   PESQ
# ----------------------
n64     1.36 +- 0.05
n71     1.35 +- 0.18
# ----------------------
Avg.    1.35 +- 0.12

## Use your own data. 
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

If all the steps are successful, you may get results printed on the screen. The training takes a few miniutes to train 10,000 iterations on a TitanX GPU. 

## Visualization
In the inference step, you may add --visualize to the arguments to plot the mixture, clean and enhanced speech log Melspectrogram. 

![alt text](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_mel_sp.png)
