#!/bin/bash
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
#
# Apache 2.0.
#
# This recipe demonstrates the use of x-vectors for speaker diarization.
# The scripts are based on the recipe in ../v1/run.sh, but clusters x-vectors
# instead of i-vectors.  It is similar to the x-vector-based diarization system
# described in "Diarization is Hard: Some Experiences and Lessons Learned for
# the JHU Team in the Inaugural DIHARD Challenge" by Sell et al.  The main
# difference is that we haven't implemented the VB resegmentation yet.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
sre_root=/mnt/nas_data/sredata/LDC2009E100_sre96_08
swbd_root=/mnt/nas_data/sredata/LDC2016E81_SRE16_SWB/LDC2016E81
callhome_root=/mnt/nas_data/sredata/LDC2009E100_sre96_08/SRE00/r65_8_1
stage=3
nnet_dir=exp/xvector_nnet_1a/
JB_path=exp/JB
JB_para_path=$JB_path/JB_parameter

iter_count=300 #Joint Bayesian训练的迭代次数

# Prepare datasets
if [ $stage -le 0 ]; then
  # Prepare a collection of NIST SRE data. This will be used to train,
  # x-vector DNN and PLDA model.
  echo "0.1 Preparing sre..."
  local/make_sre.sh $sre_root data
fi

# Prepare features
if [ $stage -le 1 ]; then
  # The script local/make_callhome.sh splits callhome into two parts, called
  # callhome1 and callhome2.  Each partition is treated like a held-out
  # dataset, and used to estimate various quantities needed to perform
  # diarization on the other part (and vice versa).
  echo "stage1.1: making mfcc on training dataset..."

  #如果没有跑train_DNN.py则跑下面的脚本
#   for name in sre; do
#     steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 \
#       --cmd "$train_cmd" --write-utt2num-frames true \
#       data/$name exp/make_mfcc $mfccdir
#     utils/fix_data_dir.sh data/$name
#   done
#   for name in sre; do
#     sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
#       data/$name exp/make_vad $vaddir
#     utils/fix_data_dir.sh data/$name
#   done

  #如果跑train_DNN.py生成了data/train文件则跑下面的脚本
  # The sre dataset is a subset of train
  cp data/train/{feats,vad}.scp data/sre/
  utils/fix_data_dir.sh data/sre

  # This writes features to disk after applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  for name in sre; do
    local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
    cp data/$name/vad.scp data/${name}_cmn/
    if [ -f data/$name/segments ]; then
      cp data/$name/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done

  echo "0.01" > data/sre_cmn/frame_shift
  # Create segments to extract x-vectors from for PLDA training data.
  # The segments are created using an energy-based speech activity
  # detection (SAD) system, but this is not necessary.  You can replace
  # this with segments computed from your favorite SAD.
  diarization/vad_to_segments.sh --nj 40 --cmd "$train_cmd" \
    data/sre_cmn data/sre_cmn_segmented
fi

# Extract x-vectors
if [ $stage -le 2 ]; then
  # Reduce the amount of training data for the PLDA,
  utils/subset_data_dir.sh data/sre_cmn_segmented 128000 data/sre_cmn_segmented_128k
  # Extract x-vectors for the SRE, which is our PLDA training
  # data.  A long period is used here so that we don't compute too
  # many x-vectors for each recording.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
    --nj 40 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
    --hard-min true $nnet_dir \
    data/sre_cmn_segmented_128k $nnet_dir/xvectors_sre_segmented_128k
fi

if [ $stage -le 3 ]; then
    mkdir -p ${JB_path}
    mkdir -p ${JB_para_path}
    train_path=$JB_path/xvectors_sre_segmented_128k
    mkdir -p ${train_path}

    ivector-subtract-global-mean scp:$nnet_dir/xvectors_sre_segmented_128k/xvector.scp ark:- | \
    # transform-vec ../v2/$nnet_dir/xvectors_Hub4m97_1/transform.mat ark:- ark:- | \
    ivector-normalize-length ark:- ark,scp:$train_path/xvector.ark,$train_path/xvector.scp
    cp $nnet_dir/xvectors_sre_segmented_128k/utt2spk $train_path/utt2spk 

    python JB/train.py "$train_path/xvector.scp" "$train_path/utt2spk" $JB_para_path $iter_count
fi
