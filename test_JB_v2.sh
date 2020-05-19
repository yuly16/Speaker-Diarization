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
Hub4m97_root=/mnt/nas_workspace2/spmiData/Hub4m97
Deliver_root=/mnt/nas_data/Deliver2Sinovoice_2014Nov/data_20140311_third
stage=7
nnet_dir=exp/xvector_nnet_1a/

JB_path=exp/JB
JB_para_path=$JB_path/JB_parameter

dataset=Deliver
xvec_path=$JB_path/xvectors_${dataset}
mkdir -p $xvec_path

# Prepare datasets
if [ $stage -le 0 ]; then
  if [ $dataset == Deliver ]; then
    # 生成kaldi格式的数据集,并将该数据集分割成recording相等的两部分.每一个部分有29个recording.
    echo "0 Preparing Deliver..."
    python dataloader/make_Deliver.py $Deliver_root data
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 16 \
        --cmd "$train_cmd" --write-utt2num-frames true \
        data/$dataset exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$dataset

    sid/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
        data/$dataset exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$dataset
  fi
fi

# Prepare features
if [ $stage -le 1 ]; then
  echo "1 making mfcc on dataset..."
  for name in $dataset; do
    local/nnet3/xvector/prepare_feats.sh --nj 16 --cmd "$train_cmd" \
        data/${name} data/${name}_cmn exp/${name}_cmn
    cp data/${name}/vad.scp data/${name}_cmn/
    if [ -f data/${name}/segments ]; then
        cp data/${name}/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done
fi

# Extract x-vectors
if [ $stage -le 2 ]; then

  echo "2 extract x-vectors on dataset..."
  # Extract x-vectors for the two partitions of callhome.
  for name in $dataset; do
    diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
      --nj 20 --window 1.5 --period 0.75 --apply-cmn false \
      --min-segment 0.5 $nnet_dir \
      data/${name}_cmn $nnet_dir/xvectors_${name}
  done
fi

# whitening
if [ $stage -le 3 ]; then
  echo "3 scaling dataset..."
####################################################

  #开发集
  ivector-subtract-global-mean $nnet_dir/xvectors_${dataset}/mean.vec scp:$nnet_dir/xvectors_${dataset}/xvector.scp ark:- | \
  ivector-normalize-length ark:- ark,scp:$xvec_path/xvector.ark,$xvec_path/xvector.scp


  mkdir -p $JB_path/log
  $train_cmd $JB_path/log/transform.log \
  est-pca --read-vectors=true --normalize-mean=false \
    --normalize-variance=true --dim=-1 \
    scp:$xvec_path/xvector.scp $xvec_path/transform.mat || exit 1;


  #开发集变换
  ivector-subtract-global-mean $nnet_dir/xvectors_${dataset}/mean.vec scp:$nnet_dir/xvectors_${dataset}/xvector.scp ark:- | \
  ivector-normalize-length ark:- ark:- | \
  transform-vec $xvec_path/transform.mat ark:- ark,scp:$xvec_path/xvector.ark,$xvec_path/xvector.scp  


  cp $nnet_dir/xvectors_${dataset}/utt2spk $xvec_path/utt2spk 
  cp $nnet_dir/xvectors_${dataset}/spk2utt $xvec_path/spk2utt 
  cp $nnet_dir/xvectors_${dataset}/segments $xvec_path/segments 


fi

# calculate score
if [ $stage -le 4 ]; then
  echo "4 calculating score on dataset..."
  python JB/test.py $xvec_path $JB_para_path $xvec_path/transform.mat $xvec_path

fi

if [ $stage -le 7 ]; then
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
    --reco2num-spk data/$dataset/reco2num_spk \
    $xvec_path $xvec_path/JB_scores_num_spk

  mkdir -p $JB_path/results
  #Now combine the results for Hub4m97_1 and Hub4m97_2 and evaluate it together.
  cat $xvec_path/JB_scores_num_spk/rttm \
    | md-eval.pl -1 -c 0.25 -r data/$dataset/ref.rttm -s - 2> $JB_path/results/num_spk.log \
    > $JB_path/results/DER_num_spk.txt
  # cat utils_others/ref_method2.rttm \
  #   | md-eval.pl -1 -c 0.25 -r data/$dataset/ref.rttm -s - 2> $JB_path/results/num_spk.log \
  #   > $JB_path/results/DER_num_spk_method2.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $JB_path/results/DER_num_spk.txt)
  echo "Using the oracle number of speakers, DER: $der%"
fi
