#!/bin/bash

MINIDATA=1
if [ $MINIDATA -eq 1 ]; then
  WORKSPACE="workspace"
  mkdir $WORKSPACE
  TR_SPEECH_DIR="mini_data/train_speech"
  TR_NOISE_DIR="mini_data/train_noise"
  TE_SPEECH_DIR="mini_data/test_speech"
  TE_NOISE_DIR="mini_data/test_noise"
  echo "Using mini data. "
else
  WORKSPACE="/vol/vssp/msos/qk/workspaces/pub_speech_enhancement/mixture2clean_dnn_pytorch"
  TR_SPEECH_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/timit_wavs/train"
  TR_NOISE_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/nosie_wavs/train"
  TE_SPEECH_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/timit_wavs/subtest"
  TE_NOISE_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/nosie_wavs/test"
  echo "Using full data. "
fi

python prepare_data.py calculate_features --workspace=$WORKSPACE --tr_speech_dir=$TR_SPEECH_DIR --tr_noise_dir=$TR_NOISE_DIR --te_speech_dir=$TE_SPEECH_DIR --te_noise_dir=$TE_NOISE_DIR

STACK_NUM=7
HOP_FRAMES=1

CUDA_VISIBLE_DEVICES=1 python xxx.py calculate_scalar --workspace=$WORKSPACE --stack_num=$STACK_NUM --hop_frames=$HOP_FRAMES

CUDA_VISIBLE_DEVICES=1 python xxx.py train --workspace=$WORKSPACE --audio_type=speech --stack_num=$STACK_NUM --hop_frames=$HOP_FRAMES

CUDA_VISIBLE_DEVICES=1 python xxx.py inference --workspace=$WORKSPACE --iteration=10000 --stack_num=$STACK_NUM --mini_num=50 --visualize

# Calculate PESQ of all enhanced speech. 
python evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --filename=xxx

# Calculate overall stats. 
python evaluate.py get_stats

