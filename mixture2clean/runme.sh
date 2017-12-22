#!/bin/bash
WORKSPACE="/vol/vssp/msos/qk/workspaces/speech_enhancement"

TR_SPEECH_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/timit_wavs/train"
TR_NOISE_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/nosie_wavs/train"

TE_SPEECH_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/timit_wavs/subtest"
TE_NOISE_DIR="/vol/vssp/msos/qk/Datasets/NOISE92_16k"

# Create mixture csv. 
python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --magnification=1
python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test

# Calculate mixture features. 
python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=0
python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=0

# Pack features. 
python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=train --snr=0 --n_concat=7 --n_hop=3
python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --snr=0 --n_concat=7 --n_hop=3

# Compute scaler. 
python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=train --snr=0

# Train. 
CUDA_VISIBLE_DEVICES=3 python main_dnn.py train --workspace=$WORKSPACE

# Inference. 
CUDA_VISIBLE_DEVICES=3 python main_dnn.py inference --workspace=$WORKSPACE --n_concat=7

# Evaluate. 
python main_dnn.py evaluate --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR
